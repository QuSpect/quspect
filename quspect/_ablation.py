"""Ablation switches for studying the transpiler's components (research only).

The environment variable ``QUSPECT_ABLATE`` holds a comma-separated list of
component names; each listed component is removed from :func:`transpile`
(and from the helpers it calls).  Unset or empty — the default — changes
nothing: every switch is off and the output is bit-identical to the
pipeline without this module.

Components (see the paper's compiler-techniques section):

``vf2``
    Exact (SWAP-free) layout stage: the trivial-layout check and the VF2++
    subgraph-monomorphism search (as ``exact_layout=False``).
``qap``
    Low-dilation (QAP) layout starts of SABRE (as ``dilation_layouts=0``).
``mirror``
    Mirror-gate-aware routing (MIRAGE mirrors, mirror-cost SWAP charging and
    the mirror-scaled SWAP-fusion bonus); block nodes stay.  The level-3
    no-mirror candidate family is then identical to the main family and is
    skipped.  The SU(4) wide lookahead stays enabled.
``comm``
    Level-3 commutation-aware candidate family (``comm/...``).
``lns``
    Window LNS of the routing (``refine_plan``: ``lns/``, ``lns2/``) and its
    equal-cost depth sweep (``deepen_plan``: ``lnsd/``), at levels 2 and 3.
``kendall``
    Layered (Kendall-tau) chain routing on paths and rings (``layers/...``).
``rollout``
    Truncated-rollout routing of SU(4) circuits (``rollout/...``).
``select``
    Level-3 candidate search: level 3 runs the level-2 routing branch
    (single SABRE result, wide lookahead / layered alternatives and
    level-2 LNS) and the same post-processing.
``stab``
    KAK stabilizer selection: only the identity stabilizer is used.
``qsd``
    Three-qubit block resynthesis (:class:`ThreeQubitBlockPass`).
``phase``
    Phase folding on the CX level (and the cancellation pass it triggers).
``approx``
    Bounded approximate synthesis (as ``approximation_degree=None``).
``orient``
    Global CX orientation by Lagrangian dual decomposition
    (:func:`orient_cx_gates` returns its input).
``orient_dual``
    Only the dual (subgradient) iterations of the CX orientation: the
    initial candidates (input orientation and all CX flipped) are still
    evaluated (``iterations=0``).
``sched``
    Depth scheduling on the commutation trace monoid
    (:func:`schedule_for_depth` returns its input).
"""

from __future__ import annotations

import os

__all__ = ["ABLATION_ENV", "COMPONENTS", "ablated", "active_components"]

#: Environment variable holding the comma-separated ablated components.
ABLATION_ENV = "QUSPECT_ABLATE"

COMPONENTS = frozenset(
    {
        "vf2",
        "qap",
        "mirror",
        "comm",
        "lns",
        "kendall",
        "rollout",
        "select",
        "stab",
        "qsd",
        "phase",
        "approx",
        "orient",
        "orient_dual",
        "sched",
    }
)

_cache: tuple[str, frozenset[str]] = ("", frozenset())


def _active() -> frozenset[str]:
    global _cache
    raw = os.environ.get(ABLATION_ENV, "")
    if raw == _cache[0]:
        return _cache[1]
    names = frozenset(x.strip() for x in raw.split(",") if x.strip())
    unknown = names - COMPONENTS
    if unknown:
        raise ValueError(
            f"{ABLATION_ENV}: unknown component(s) {sorted(unknown)}; "
            f"choose from {sorted(COMPONENTS)}."
        )
    _cache = (raw, names)
    return names


def ablated(name: str) -> bool:
    """Whether component ``name`` is switched off by ``QUSPECT_ABLATE``.

    Example:
        >>> ablated("lns")  # default: nothing is ablated
        False
    """
    return name in _active()


def active_components() -> frozenset[str]:
    """The components currently switched off (empty by default).

    Memoization caches keep one table per value, so results computed with a
    component switched off never reach the default pipeline.

    Example:
        >>> active_components()
        frozenset()
    """
    return _active()
