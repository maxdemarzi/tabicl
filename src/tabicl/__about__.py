# PEP 440 local version identifier. The base is upstream's released 2.1.1, which this fork
# is built on; `+scaling.N` marks the scaling-upgrades work layered on top. Without it an
# install from this branch reports plain "2.1.1" and is indistinguishable from the real
# PyPI release despite ~400 commits of divergence, including changes under _model/ — which
# would make any environment record of a measurement ambiguous.
#
# A local version is also unpublishable to PyPI by construction, so this fork cannot
# collide with the upstream `tabicl` namespace by accident.
__version__ = "2.1.1+scaling.1"
