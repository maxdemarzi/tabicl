// Worst-case optimal join, compiled.
//
// Follows the strategy Umbra adopts in Freitag, Bandle, Schmidt, Kemper & Neumann,
// "Adopting Worst-Case Optimal Joins in Relational Database Systems" (VLDB 2020),
// Algorithm 3: at each join level, iterate the *smallest* participating relation's
// candidate set and probe every other participant, rather than advancing sorted
// iterators in lockstep the way leapfrog triejoin does. Work at a level is then
// bounded by the most selective relation instead of by a coordinated seek across all
// of them.
//
// Four deviations from a textbook write-up, each one because the obvious version is
// slower in practice:
//
//   1. Candidate sets are sorted vectors probed by binary search, NOT hash sets.
//      A hash set per distinct join key means hundreds of thousands of tiny
//      bucket-array allocations, and construction/destruction -- not lookup --
//      dominates. O(log n) probes with no allocation beat O(1) probes that each cost
//      a malloc.
//
//   2. One direction per relation. With the variable ordering fixed up front, each
//      relation's attributes have fixed anchor/dependent roles, so only one index
//      direction is ever consulted. Building both halves memory and construction for
//      nothing. (COLT/Free Join.)
//
//   3. No per-node heap allocation. The inner loop runs once per search-tree node;
//      anything allocating there shows up immediately in a profile. Saved state uses
//      a preallocated stack buffer.
//
//   5. Parallel, with a cost model before a scheduler. Three things had to be right,
//      in order:
//        (a) Cost of a candidate is the PRODUCT of the touching relations' bucket
//            sizes, not the min. Once the next variable is bound, relations that
//            share only the bound variable form a cross product, so min badly
//            underweights a symmetric hub vertex -- which can alone be a third of
//            all work. A scheduler is only as good as its estimates.
//        (b) Scheduling is dynamic, not striped or statically partitioned. Round
//            robin over value-sorted candidates is systematically unfair because
//            degree correlates with value globally; static LPT still strands a
//            thread whose batch outruns its estimate. A shared atomic cursor over a
//            cost-descending list lets every other thread keep draining.
//        (c) Even perfect scheduling cannot beat one oversized item, since an item
//            runs start-to-finish on one thread. Anything above a fair share
//            (total / threads) is pre-split into finer sub-items, recursively, so no
//            single item can cap the speedup.
//
//   4. The last variable is resolved in bulk. Once every other variable is bound, the
//      remaining answers are exactly the intersection of the participants' candidate
//      sets -- computable as a sorted merge over the whole set at once, instead of a
//      recursion frame plus a probe per output tuple.
//
// Storage is CSR-style flat arrays (values + child ids + per-node offsets) rather
// than vector-of-vector, so a level is contiguous rather than a pointer chase.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <thread>
#include <climits>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using i64 = std::int64_t;

namespace {

constexpr int kMaxRelations = 64;

// A relation as a trie over the global variable ordering restricted to its own
// attributes, stored CSR-style. For depth d, node n owns the half-open value range
// [start[d][n], start[d][n + 1]) inside vals[d]; child[d][i] is the node index at
// depth d+1 reached by vals[d][i].
struct Trie {
	int arity = 0;
	std::vector<std::vector<i64>> vals;
	std::vector<std::vector<int>> child;
	std::vector<std::vector<int>> start;

	void build(const i64 *data, std::size_t rows, int k) {
		arity = k;
		vals.assign(k, {});
		child.assign(k, {});
		start.assign(k, {});

		// One lexicographic sort up front; the trie is then just the run structure of
		// the sorted rows, so no per-node hash table is ever built.
		std::vector<std::size_t> order(rows);
		std::iota(order.begin(), order.end(), std::size_t{0});
		std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
			for (int d = 0; d < k; ++d) {
				const i64 x = data[a * k + d], y = data[b * k + d];
				if (x != y) {
					return x < y;
				}
			}
			return false;
		});

		std::vector<std::pair<std::size_t, std::size_t>> ranges{{0, rows}};
		for (int d = 0; d < k; ++d) {
			start[d].push_back(0);
			std::vector<std::pair<std::size_t, std::size_t>> next;
			for (const auto &range : ranges) {
				std::size_t i = range.first;
				while (i < range.second) {
					const i64 v = data[order[i] * k + d];
					std::size_t j = i;
					while (j < range.second && data[order[j] * k + d] == v) {
						++j;
					}
					vals[d].push_back(v);
					child[d].push_back(static_cast<int>(next.size()));
					next.emplace_back(i, j);
					i = j;
				}
				start[d].push_back(static_cast<int>(vals[d].size()));
			}
			ranges.swap(next);
		}
	}

	inline int lo(int d, int node) const {
		return start[d][node];
	}
	inline int hi(int d, int node) const {
		return start[d][node + 1];
	}
};

} // namespace

namespace {

// Per-worker search state. The tries are read-only once built, so workers share them
// and own only their cursor stack and output buffer.
struct JoinState {
	const std::vector<Trie> *tries = nullptr;
	const std::vector<std::vector<std::pair<int, int>>> *by_var = nullptr;
	const std::vector<std::vector<int>> *lower_bounds = nullptr;
	int n_vars = 0;
	std::size_t max_results = 0;

	std::vector<int> node;
	std::vector<i64> binding;
	std::vector<i64> out;
	std::vector<i64> acc, tmp;
	bool capped = false;

	// Aggregation mode (FAQ): accumulate during elimination instead of emitting
	// tuples. `counts[v]` is how many results contain v in any position.
	bool counting = false;
	i64 total = 0;
	std::vector<i64> counts;

	void recurse(int level);
};

void JoinState::recurse(int level) {
	if (capped) {
		return;
	}
	const auto &participants = (*by_var)[level];
	const auto &T = *tries;

	i64 floor_value = INT64_MIN;
	bool has_floor = false;
	for (int j : (*lower_bounds)[level]) {
		floor_value = std::max(floor_value, binding[j] + 1);
		has_floor = true;
	}

	// ---- Bulk resolution of the final variable. ----
	if (level == n_vars - 1) {
		int lead = 0, best = INT_MAX;
		for (std::size_t p = 0; p < participants.size(); ++p) {
			const int rel = participants[p].first, depth = participants[p].second;
			const int width = T[rel].hi(depth, node[rel]) - T[rel].lo(depth, node[rel]);
			if (width < best) {
				best = width;
				lead = static_cast<int>(p);
			}
		}
		const int lrel = participants[lead].first, ldep = participants[lead].second;
		const i64 *lbeg = T[lrel].vals[ldep].data() + T[lrel].lo(ldep, node[lrel]);
		const i64 *lend = T[lrel].vals[ldep].data() + T[lrel].hi(ldep, node[lrel]);
		if (has_floor) {
			lbeg = std::lower_bound(lbeg, lend, floor_value);
		}

		acc.assign(lbeg, lend);
		for (std::size_t p = 0; p < participants.size() && !acc.empty(); ++p) {
			if (static_cast<int>(p) == lead) {
				continue;
			}
			const int rel = participants[p].first, depth = participants[p].second;
			const i64 *obeg = T[rel].vals[depth].data() + T[rel].lo(depth, node[rel]);
			const i64 *oend = T[rel].vals[depth].data() + T[rel].hi(depth, node[rel]);
			tmp.clear();
			std::set_intersection(acc.begin(), acc.end(), obeg, oend, std::back_inserter(tmp));
			acc.swap(tmp);
		}

		if (counting) {
			// |I| completions are known without visiting them, so the already-bound
			// variables are credited in O(1) rather than once per tuple, and nothing is
			// materialised. Only the final variable needs a pass, and only because
			// per-value counts were asked for.
			const i64 n_completions = static_cast<i64>(acc.size());
			total += n_completions;
			for (int v = 0; v < level; ++v) {
				counts[binding[v]] += n_completions;
			}
			for (i64 v : acc) {
				counts[v] += 1;
			}
			return;
		}

		for (i64 v : acc) {
			binding[level] = v;
			out.insert(out.end(), binding.begin(), binding.end());
			if (max_results && out.size() / static_cast<std::size_t>(n_vars) >= max_results) {
				capped = true;
				return;
			}
		}
		return;
	}

	// ---- Interior level: lead with the smallest candidate set, probe the rest. ----
	int lead = 0, best = INT_MAX;
	for (std::size_t p = 0; p < participants.size(); ++p) {
		const int rel = participants[p].first, depth = participants[p].second;
		const int width = T[rel].hi(depth, node[rel]) - T[rel].lo(depth, node[rel]);
		if (width < best) {
			best = width;
			lead = static_cast<int>(p);
		}
	}

	const int lrel = participants[lead].first, ldep = participants[lead].second;
	const int lbegin = T[lrel].lo(ldep, node[lrel]);
	const int lfinish = T[lrel].hi(ldep, node[lrel]);
	const std::vector<i64> &lvals = T[lrel].vals[ldep];

	int cursor = lbegin;
	if (has_floor) {
		cursor = static_cast<int>(
		    std::lower_bound(lvals.begin() + lbegin, lvals.begin() + lfinish, floor_value) - lvals.begin());
	}

	int saved[kMaxRelations];
	for (; cursor < lfinish; ++cursor) {
		const i64 value = lvals[cursor];
		bool ok = true;
		int n_saved = 0;
		for (std::size_t p = 0; p < participants.size(); ++p) {
			if (static_cast<int>(p) == lead) {
				continue;
			}
			const int rel = participants[p].first, depth = participants[p].second;
			const std::vector<i64> &vv = T[rel].vals[depth];
			const int a = T[rel].lo(depth, node[rel]), b = T[rel].hi(depth, node[rel]);
			const auto it = std::lower_bound(vv.begin() + a, vv.begin() + b, value);
			if (it == vv.begin() + b || *it != value) {
				ok = false;
				break;
			}
			saved[n_saved++] = node[rel];
			node[rel] = T[rel].child[depth][it - vv.begin()];
		}

		if (ok) {
			const int saved_lead = node[lrel];
			node[lrel] = T[lrel].child[ldep][cursor];
			binding[level] = value;
			recurse(level + 1);
			node[lrel] = saved_lead;
		}

		int k = 0;
		for (std::size_t p = 0; p < participants.size() && k < n_saved; ++p) {
			if (static_cast<int>(p) == lead) {
				continue;
			}
			node[participants[p].first] = saved[k++];
		}
		if (capped) {
			return;
		}
	}
}

} // namespace

namespace {

// A unit of schedulable work: a prefix of bound variables, the trie cursors that
// prefix implies, and an estimate of what remains beneath it.
struct WorkItem {
	std::vector<i64> prefix;
	std::vector<int> node;
	double cost = 0.0;
	int level = 0;
};

// Cost = PRODUCT of the candidate-set widths at this item's level, not the minimum.
// Relations sharing only an already-bound variable fan out multiplicatively, so a min
// estimate understates exactly the hub vertices that dominate a run.
inline double estimate(const std::vector<Trie> &T, const std::vector<std::pair<int, int>> &parts,
                       const std::vector<int> &node) {
	double cost = 1.0;
	for (const auto &pd : parts) {
		const int width = T[pd.first].hi(pd.second, node[pd.first]) - T[pd.first].lo(pd.second, node[pd.first]);
		cost *= static_cast<double>(width > 0 ? width : 1);
	}
	return cost;
}

// Bind one more variable, turning an item into its children. Mirrors the interior
// level of the search, but emits work instead of recursing.
void expand(const std::vector<Trie> &T, const std::vector<std::vector<std::pair<int, int>>> &by_var,
            const std::vector<std::vector<int>> &lower_bounds, const WorkItem &item, std::vector<WorkItem> &out) {
	const int level = item.level;
	const auto &parts = by_var[level];

	i64 floor_value = INT64_MIN;
	bool has_floor = false;
	for (int j : lower_bounds[level]) {
		floor_value = std::max(floor_value, item.prefix[j] + 1);
		has_floor = true;
	}

	int lead = 0, best = INT_MAX;
	for (std::size_t p = 0; p < parts.size(); ++p) {
		const int rel = parts[p].first, depth = parts[p].second;
		const int width = T[rel].hi(depth, item.node[rel]) - T[rel].lo(depth, item.node[rel]);
		if (width < best) {
			best = width;
			lead = static_cast<int>(p);
		}
	}
	const int lrel = parts[lead].first, ldep = parts[lead].second;
	const std::vector<i64> &lvals = T[lrel].vals[ldep];
	const int lbegin = T[lrel].lo(ldep, item.node[lrel]), lfinish = T[lrel].hi(ldep, item.node[lrel]);

	int cursor = lbegin;
	if (has_floor) {
		cursor = static_cast<int>(
		    std::lower_bound(lvals.begin() + lbegin, lvals.begin() + lfinish, floor_value) - lvals.begin());
	}

	for (; cursor < lfinish; ++cursor) {
		const i64 value = lvals[cursor];
		std::vector<int> child = item.node;
		bool ok = true;
		for (std::size_t p = 0; p < parts.size(); ++p) {
			if (static_cast<int>(p) == lead) {
				continue;
			}
			const int rel = parts[p].first, depth = parts[p].second;
			const std::vector<i64> &vv = T[rel].vals[depth];
			const int a = T[rel].lo(depth, item.node[rel]), b = T[rel].hi(depth, item.node[rel]);
			const auto it = std::lower_bound(vv.begin() + a, vv.begin() + b, value);
			if (it == vv.begin() + b || *it != value) {
				ok = false;
				break;
			}
			child[rel] = T[rel].child[depth][it - vv.begin()];
		}
		if (!ok) {
			continue;
		}
		child[lrel] = T[lrel].child[ldep][cursor];

		WorkItem next;
		next.prefix = item.prefix;
		next.prefix[level] = value;
		next.node = std::move(child);
		next.level = level + 1;
		next.cost = next.level < static_cast<int>(by_var.size()) ? estimate(T, by_var[next.level], next.node) : 1.0;
		out.push_back(std::move(next));
	}
}

} // namespace

// Shared driver. `counting` selects FAQ-style aggregation: the search is identical,
// only what happens at a completed prefix differs.
static void run_join(const std::vector<py::array_t<i64>> &relations, const std::vector<std::vector<int>> &var_ids,
                     int n_vars, const std::vector<std::pair<int, int>> &less_than, std::size_t max_results,
                     int threads, bool counting, std::vector<i64> &merged, i64 &grand_total,
                     std::vector<i64> &merged_counts) {
	const int n_rel = static_cast<int>(relations.size());
	if (n_rel > kMaxRelations) {
		throw std::runtime_error("too many relations");
	}
	if (n_vars <= 0) {
		throw std::runtime_error("n_vars must be positive");
	}

	std::vector<Trie> tries(n_rel);
	std::vector<const i64 *> ptrs(n_rel);
	std::vector<std::size_t> rows(n_rel);
	std::vector<int> arity(n_rel);
	for (int i = 0; i < n_rel; ++i) {
		auto buf = relations[i].request();
		if (buf.ndim != 2) {
			throw std::runtime_error("each relation must be 2-D");
		}
		arity[i] = static_cast<int>(buf.shape[1]);
		if (arity[i] != static_cast<int>(var_ids[i].size())) {
			throw std::runtime_error("var_ids length must match relation arity");
		}
		ptrs[i] = static_cast<const i64 *>(buf.ptr);
		rows[i] = static_cast<std::size_t>(buf.shape[0]);
	}

	std::vector<std::vector<std::pair<int, int>>> by_var(n_vars);
	for (int i = 0; i < n_rel; ++i) {
		for (int d = 0; d < arity[i]; ++d) {
			by_var[var_ids[i][d]].emplace_back(i, d);
		}
	}
	for (int v = 0; v < n_vars; ++v) {
		if (by_var[v].empty()) {
			throw std::runtime_error("variable appears in no relation");
		}
	}

	std::vector<std::vector<int>> lower_bounds(n_vars);
	for (const auto &lt : less_than) {
		if (lt.first >= lt.second) {
			throw std::runtime_error("less_than requires the lesser variable to come first");
		}
		lower_bounds[lt.second].push_back(lt.first);
	}

	// Counter array is indexed by value, so it must span the value domain.
	i64 domain = 0;
	if (counting) {
		for (int i = 0; i < n_rel; ++i) {
			const i64 *d = ptrs[i];
			const std::size_t n = rows[i] * static_cast<std::size_t>(arity[i]);
			for (std::size_t j = 0; j < n; ++j) {
				domain = std::max(domain, d[j]);
			}
		}
		++domain;
		if (domain <= 0) {
			domain = 1;
		}
	}

	{
		// The join touches no Python objects, so hold nothing while it runs.
		py::gil_scoped_release release;

		for (int i = 0; i < n_rel; ++i) {
			tries[i].build(ptrs[i], rows[i], arity[i]);
		}

		auto fresh = [&]() {
			JoinState st;
			st.tries = &tries;
			st.by_var = &by_var;
			st.lower_bounds = &lower_bounds;
			st.n_vars = n_vars;
			st.max_results = max_results;
			st.node.assign(n_rel, 0);
			st.binding.assign(n_vars, 0);
			st.counting = counting;
			if (counting) {
				st.counts.assign(static_cast<std::size_t>(domain), 0);
			}
			return st;
		};

		unsigned hw = std::thread::hardware_concurrency();
		int n_threads = threads > 0 ? threads : static_cast<int>(hw ? hw : 1u);
		// Capping results makes "which tuples" order-dependent, so keep that serial
		// rather than return a nondeterministic subset.
		if (max_results || n_vars < 2) {
			n_threads = 1;
		}

		if (n_threads <= 1) {
			JoinState st = fresh();
			st.recurse(0);
			merged.swap(st.out);
			grand_total = st.total;
			merged_counts.swap(st.counts);
		} else {
			// Seed the work list by binding variable 0.
			WorkItem root;
			root.prefix.assign(n_vars, 0);
			root.node.assign(n_rel, 0);
			root.level = 0;
			std::vector<WorkItem> items;
			expand(tries, by_var, lower_bounds, root, items);

			// Pre-split anything above a fair share, recursively. Without this one
			// dominant item runs alone on one thread and caps the whole speedup, no
			// matter how well everything else is packed.
			constexpr int kMaxSplitRounds = 8;
			constexpr std::size_t kMaxItems = 1u << 20;
			for (int round = 0; round < kMaxSplitRounds; ++round) {
				double total = 0.0;
				for (const auto &it : items) {
					total += it.cost;
				}
				const double fair = total / static_cast<double>(n_threads);
				bool split_any = false;
				std::vector<WorkItem> next;
				next.reserve(items.size());
				for (auto &it : items) {
					if (it.cost > fair && it.level < n_vars - 1 && next.size() < kMaxItems) {
						expand(tries, by_var, lower_bounds, it, next);
						split_any = true;
					} else {
						next.push_back(std::move(it));
					}
				}
				items.swap(next);
				if (!split_any) {
					break;
				}
			}

			// Largest first: starting the biggest item earliest minimises the chance it
			// is still running once every other thread has drained the queue.
			std::sort(items.begin(), items.end(),
			          [](const WorkItem &a, const WorkItem &b) { return a.cost > b.cost; });

			std::vector<JoinState> workers;
			workers.reserve(n_threads);
			for (int t = 0; t < n_threads; ++t) {
				workers.push_back(fresh());
			}

			// A single shared cursor, claimed atomically -- dynamic, so a thread whose
			// real cost exceeds its estimate strands nothing.
			std::atomic<std::size_t> next_item{0};
			auto run = [&](int tid) {
				JoinState &st = workers[tid];
				for (;;) {
					const std::size_t idx = next_item.fetch_add(1, std::memory_order_relaxed);
					if (idx >= items.size()) {
						break;
					}
					const WorkItem &it = items[idx];
					st.node = it.node;
					for (int v = 0; v < it.level; ++v) {
						st.binding[v] = it.prefix[v];
					}
					st.recurse(it.level);
				}
			};

			std::vector<std::thread> pool;
			pool.reserve(n_threads - 1);
			for (int t = 1; t < n_threads; ++t) {
				pool.emplace_back(run, t);
			}
			run(0);
			for (auto &th : pool) {
				th.join();
			}

			if (counting) {
				merged_counts.assign(static_cast<std::size_t>(domain), 0);
				for (const auto &w : workers) {
					grand_total += w.total;
					for (std::size_t v = 0; v < merged_counts.size(); ++v) {
						merged_counts[v] += w.counts[v];
					}
				}
			} else {
				std::size_t total_out = 0;
				for (const auto &w : workers) {
					total_out += w.out.size();
				}
				merged.reserve(total_out);
				for (auto &w : workers) {
					merged.insert(merged.end(), w.out.begin(), w.out.end());
				}
			}
		}
	}

}

static py::array_t<i64> wcoj_hash_join(const std::vector<py::array_t<i64>> &relations,
                                       const std::vector<std::vector<int>> &var_ids, int n_vars,
                                       const std::vector<std::pair<int, int>> &less_than, std::size_t max_results,
                                       int threads) {
	std::vector<i64> merged, counts;
	i64 total = 0;
	run_join(relations, var_ids, n_vars, less_than, max_results, threads, false, merged, total, counts);

	const std::size_t n_out = merged.size() / static_cast<std::size_t>(n_vars);
	py::array_t<i64> result({n_out, static_cast<std::size_t>(n_vars)});
	if (n_out) {
		std::copy(merged.begin(), merged.end(), static_cast<i64 *>(result.request().ptr));
	}
	return result;
}

// FAQ-style aggregation: returns (total results, per-value occurrence counts) without
// ever materialising a result tuple.
static py::tuple wcoj_count(const std::vector<py::array_t<i64>> &relations,
                            const std::vector<std::vector<int>> &var_ids, int n_vars,
                            const std::vector<std::pair<int, int>> &less_than, int threads) {
	std::vector<i64> merged, counts;
	i64 total = 0;
	run_join(relations, var_ids, n_vars, less_than, 0, threads, true, merged, total, counts);

	py::array_t<i64> out(static_cast<py::ssize_t>(counts.size()));
	if (!counts.empty()) {
		std::copy(counts.begin(), counts.end(), static_cast<i64 *>(out.request().ptr));
	}
	return py::make_tuple(total, out);
}

PYBIND11_MODULE(_wcoj_native, m) {
	m.doc() = "Worst-case optimal join (Umbra, VLDB 2020, Algorithm 3)";
	m.def("wcoj_hash_join", &wcoj_hash_join, py::arg("relations"), py::arg("var_ids"), py::arg("n_vars"),
	      py::arg("less_than"), py::arg("max_results") = 0, py::arg("threads") = 0);
	m.def("wcoj_count", &wcoj_count, py::arg("relations"), py::arg("var_ids"), py::arg("n_vars"),
	      py::arg("less_than"), py::arg("threads") = 0);
}
