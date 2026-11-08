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

Base-pipeline switches (for grouped ablations and contribution ladders):

``kak``
    Two-qubit block consolidation and KAK resynthesis before routing, after
    routing and after optimization (no :class:`ConsolidateBlocksPass`), the
    pre-routing ``SWAP · U`` block resynthesis of SABRE and the final
    SWAP-face block rewrite.  Gates are translated one by one (unsupported
    gates with a matrix are still synthesized) and simplified by the
    peephole optimizer.  This also disables everything that needs a
    consolidation pass (three-qubit QSD, approximate synthesis and the
    commutation-aware candidate family).
``sabre2``
    The routing extensions of SABRE v2: block nodes (and with them mirror
    gates, the SWAP-fusion bonus and the SU(4) wide lookahead), Toffolis
    routed as three-qubit units (they are decomposed before routing) and
    the pre-routing ``SWAP · U`` block resynthesis.  The LightSABRE core
    (lookahead, decay, release valve, layout trials) stays.
``gauge``
    Single-qubit gauge minimization after translation
    (:func:`minimize_single_qubit_gates` is skipped); the peephole
    single-qubit fusion of the optimizer stays.

Level-3 components added in round 21:

``pp_small``
    Post-routing CNOT-phase region resynthesis on small regions: the 4 / 5
    wire partitions (exact A*) of both the basic (all candidates) and the
    full resynthesis.
``pp_large``
    Post-routing CNOT-phase resynthesis on large regions: the 16-wire
    partitions (beam search beyond 6 wires) of the full resynthesis.
``pp_twolevel``
    Two-level candidate scoring: only the candidate ranked best by the
    basic resynthesis gets the full resynthesis (instead of the best three).
``pp_raw``
    The second full resynthesis pass from the unmerged routed circuit.
``pp_pre``
    The second start: CNOT-phase resynthesis of the logical circuit before
    routing, compiled as an alternative.
``num3q``
    Numerical instantiation of whole three-qubit circuits.
``numblk``
    Numerical instantiation of three-qubit blocks after routing (including
    the output permutation of terminal blocks).
``plain``
    Plain gate-level SABRE candidate family (``plain/...``).
``repass``
    Recompiling the result with the identity layout.
``beam``
    Global beam-search routing candidate (``beam/...``).

The ``pp_*`` switches only remove the post-routing resynthesis or the
pre-routing start they name; ``pp_pre`` uses the full partitions
regardless of ``pp_small`` / ``pp_large``.
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
        "kak",
        "sabre2",
        "gauge",
        "pp_small",
        "pp_large",
        "pp_twolevel",
        "pp_raw",
        "pp_pre",
        "num3q",
        "numblk",
        "plain",
        "repass",
        "beam",
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
