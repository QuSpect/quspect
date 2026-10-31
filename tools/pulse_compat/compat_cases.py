"""Deterministic behaviour probes for the envelopes C++ module.

Every case's outcome is pickled so that two implementations (the envelopes-qc
0.1.5 reference wheel and ``quspect._pulse``) can be compared by ``compare.py``::

    python compat_cases.py OUT.pkl [--module envelopes.envelopes_cpp] [--path DIR]
"""

import argparse
import copy
import importlib
import math
import os
import pickle
import random
import struct
import sys
import warnings

import numpy as np


def encode(value):
    """Turn a result into plain, comparable Python data."""
    if isinstance(value, np.ndarray):
        return ("ndarray", str(value.dtype), value.shape, value.tobytes())
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [encode(v) for v in value])
    if isinstance(value, (bool, int, str, bytes, type(None))):
        return value
    if isinstance(value, float):
        return ("float", value.hex() if math.isfinite(value) else repr(value))
    if isinstance(value, complex):
        return ("complex", encode(value.real), encode(value.imag))
    if isinstance(value, (np.floating, np.complexfloating, np.integer, np.bool_)):
        return ("np", type(value).__name__, encode(value.item()))
    # Envelope / WaveCache objects: describe by type and repr.
    return ("obj", type(value).__name__, repr(value))


class Cases:
    def __init__(self, evc):
        self.evc = evc
        self.cases = {}

    def add(self, name, fn):
        assert name not in self.cases, name
        self.cases[name] = fn

    def run(self):
        results = {}
        trace = "--trace" in sys.argv
        for name, fn in self.cases.items():
            if trace:
                print(name, file=sys.stderr, flush=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    out = ("ok", encode(fn()))
                except Exception as e:  # noqa: BLE001 - we record every failure mode
                    out = ("err", type(e).__name__, str(e))
            out = out + (tuple(sorted({w.category.__name__ for w in caught})),)
            results[name] = out
        return results


def describe(evc, env):
    """Summarize an envelope: type, repr, bounds, complexity and serialization."""
    return (
        type(env).__name__,
        repr(env),
        env.start(),
        env.end(),
        env.is_complex(),
        evc.serialization(env),
    )


def decode_all(evc, env, res=0.5, windows=()):
    wc = evc.WaveCache(res)
    out = [evc.decode_envelope(env, wc)]
    for s, e in windows:
        try:
            out.append(evc.decode_envelope(env, wc, s, e))
        except Exception as ex:  # noqa: BLE001
            out.append(("err", type(ex).__name__, str(ex)))
    # Decode again with a warm cache: must give identical data.
    out.append(evc.decode_envelope(env, wc))
    return out


# ---------------------------------------------------------------- envelope factories


def leaf_factories(evc):
    """name -> (constructor, random-argument generator)."""

    def g_time(r):
        return round(r.uniform(-20, 40), r.choice([0, 1, 2, 3, 5]))

    def g_width(r):
        return round(r.uniform(0.5, 15), r.choice([0, 1, 2, 4]))

    def g_amp(r):
        return round(r.uniform(-1.5, 1.5), r.choice([1, 3, 6]))

    return {
        "Gaussian": lambda r: evc.Gaussian(g_time(r), g_width(r), g_amp(r)),
        "GaussianDRAG": lambda r: evc.GaussianDRAG(
            g_time(r),
            g_width(r),
            g_amp(r),
            r.uniform(-2, 2),
            r.uniform(-0.3, 0.3),
            r.uniform(-4, 4),
        ),
        "CosineDRAG": lambda r: evc.CosineDRAG(
            g_time(r),
            g_width(r),
            g_amp(r),
            r.uniform(-2, 2),
            r.uniform(-0.3, 0.3),
            r.uniform(-4, 4),
        ),
        "Triangle": lambda r: evc.Triangle(
            g_time(r), g_width(r), g_amp(r), r.random() < 0.5
        ),
        "Rect": lambda r: evc.Rect(g_time(r), g_width(r), g_amp(r)),
        "Flattop": lambda r: evc.Flattop(
            g_time(r), g_width(r) * 3, g_width(r) / 3, g_width(r) / 3, g_amp(r)
        ),
        "RippleRect": lambda r: evc.RippleRect(
            g_time(r),
            g_width(r) * 3,
            g_amp(r),
            g_width(r) / 4,
            r.uniform(-0.2, 0.2),
            r.uniform(-0.2, 0.2),
            r.uniform(-0.2, 0.2),
            r.uniform(-0.2, 0.2),
        ),
        "PowerSineCosine": lambda r: evc.PowerSineCosine(
            g_time(r),
            g_width(r) * 2,
            g_amp(r),
            r.randint(0, 4),
            r.randint(0, 4),
            r.uniform(0.2, 2.5),
            r.uniform(-1, 1),
        ),
        "GaussianXPower": lambda r: evc.GaussianXPower(
            g_time(r), g_width(r), g_amp(r), r.randint(0, 4)
        ),
        "CustomReal": lambda r: evc.Custom(
            g_time(r),
            r.choice([0.5, 0.3, 1.0, 0.25, 0.7]),
            np.array([r.uniform(-1, 1) for _ in range(r.randint(1, 40))]),
        ),
        "CustomComplex": lambda r: evc.Custom(
            g_time(r),
            r.choice([0.5, 0.3, 1.0, 0.25, 0.7]),
            np.array(
                [
                    complex(r.uniform(-1, 1), r.uniform(-1, 1))
                    for _ in range(r.randint(1, 40))
                ]
            ),
        ),
    }


def crashes_reference(evc, env):
    """True if env contains a RealEnvProd wrapping a complex envelope.

    The reference wheel segfaults in mix() and copy() on such trees.  The structure
    is found by parsing the serialization format (u64 count, tagged nodes in
    preorder, then u32 child ids).
    """
    data = evc.serialization(env)
    (count,) = struct.unpack_from("<Q", data, 0)
    pos, nodes = 8, []
    fixed = {
        0: 20,
        1: 44,
        2: 44,
        3: 21,
        4: 20,
        5: 36,
        6: 60,
        7: 44,
        8: 24,
        10: 33,
        11: 0,
        12: 8,
    }
    for _ in range(count):
        tag = data[pos]
        pos += 1
        info = {
            "tag": tag,
            "complex": tag in (1, 2, 10, 11),
            "nchild": {11: 2, 12: 1, 13: 1}.get(tag, 0),
        }
        if tag in fixed:
            pos += fixed[tag]
        elif tag == 9:
            info["complex"] = bool(data[pos + 16])
            (n,) = struct.unpack_from("<Q", data, pos + 17)
            pos += 25 + n * (8 if info["complex"] else 4)
        elif tag == 13:
            info["prod_complex"] = bool(data[pos])
            pos += 9 if data[pos] else 5
        elif tag == 14:
            (info["nchild"],) = struct.unpack_from("<Q", data, pos)
            pos += 8
        nodes.append(info)
    for info in nodes:
        info["children"] = list(struct.unpack_from(f"<{info['nchild']}I", data, pos))
        pos += 4 * info["nchild"]

    def is_complex(i):
        n = nodes[i]
        if n["tag"] == 13:
            return n["prod_complex"] or is_complex(n["children"][0])
        if n["tag"] in (12, 14):
            return any(is_complex(c) for c in n["children"])
        return n["complex"]

    return any(
        n["tag"] == 13 and not n["prod_complex"] and is_complex(n["children"][0])
        for n in nodes
    )


def random_expr(evc, r, leaves, depth=0):
    """Random envelope expression -> (env, mix_crashes_reference).

    The reference segfaults in mix() on a RealEnvProd wrapping a complex envelope
    (e.g. mix(GaussianDRAG * 2, ...)); such inputs are never mixed here.
    """
    kind = r.random()
    if depth >= 3 or kind < 0.35:
        return leaves[r.choice(sorted(leaves))](r), False
    sub, risky = random_expr(evc, r, leaves, depth + 1)
    if kind < 0.5:
        return sub >> round(r.uniform(-30, 30), r.choice([0, 1, 2, 4, 6])), risky
    if kind < 0.62:
        return sub * round(r.uniform(-2, 2), 4), risky or sub.is_complex()
    if kind < 0.7:
        return sub * complex(
            round(r.uniform(-2, 2), 3), round(r.uniform(-2, 2), 3)
        ), risky
    if kind < 0.82:
        other, risky2 = random_expr(evc, r, leaves, depth + 1)
        return sub + other, risky or risky2
    if kind < 0.92:
        df, phase, dyn = r.uniform(-0.3, 0.3), r.uniform(-3, 3), r.random() < 0.5
        if crashes_reference(evc, sub):
            return sub, True
        return evc.mix(sub, df, phase, dyn), False
    n = r.randint(1, 5)
    dt = np.array([round(r.uniform(-40, 40), r.choice([0, 1, 3])) for _ in range(n)])
    if r.random() < 0.4:
        return evc.align(sub, dt), risky
    if r.random() < 0.5:
        amp = np.array([r.uniform(-1, 1) for _ in range(n)], dtype=np.float32)
        return evc.align(sub, dt, amp), risky or sub.is_complex()
    amp = np.array(
        [complex(r.uniform(-1, 1), r.uniform(-1, 1)) for _ in range(n)],
        dtype=np.complex64,
    )
    return evc.align(sub, dt, amp), risky


# ---------------------------------------------------------------- case catalogue


def build(evc):
    cases = Cases(evc)
    arr = np.array
    leaves = leaf_factories(evc)

    # Every leaf type: random instances, several resolutions, real/complex paths.
    for name, make in sorted(leaves.items()):
        r = random.Random(f"leaf-{name}")
        for i in range(25):
            env = make(r)
            res = r.choice([0.5, 0.5, 0.25, 1.0, 0.1, 0.3, 2.0])
            shift = round(r.uniform(-3, 3), r.choice([0, 1, 2, 5]))
            s = env.start() + shift - r.uniform(-5, 10)
            e = env.end() + shift + r.uniform(-5, 10)
            cases.add(f"leaf/{name}/{i}/describe", lambda env=env: describe(evc, env))
            cases.add(
                f"leaf/{name}/{i}/decode",
                lambda env=env, res=res, s=s, e=e: decode_all(evc, env, res, [(s, e)]),
            )
            cases.add(
                f"leaf/{name}/{i}/shifted",
                lambda env=env, res=res, sh=shift: decode_all(evc, env >> sh, res),
            )
            cases.add(
                f"leaf/{name}/{i}/complex_path",
                lambda env=env, res=res, sh=shift: decode_all(
                    evc, (env >> sh) * (1 + 0j), res
                ),
            )
            cases.add(
                f"leaf/{name}/{i}/mixed",
                lambda env=env, res=res, sh=shift: decode_all(
                    evc, evc.mix(env >> sh, 0.07, 0.4, sh > 0), res
                ),
            )

    # Random composite expressions.
    for i in range(int(os.environ.get("COMPAT_EXPRS", "300"))):
        r = random.Random(f"expr-{i}")
        env, _ = random_expr(evc, r, leaves)
        risky = crashes_reference(evc, env)
        res = r.choice([0.5, 0.5, 0.25, 1.0, 0.2])
        windows = [
            (env.start() - r.uniform(0, 20), env.end() + r.uniform(0, 20)),
            (env.start() + r.uniform(0, 5), env.end() - r.uniform(0, 5)),
        ]
        cases.add(f"expr/{i}/describe", lambda env=env: describe(evc, env))
        cases.add(
            f"expr/{i}/decode",
            lambda env=env, res=res, w=windows: decode_all(evc, env, res, w),
        )
        cases.add(
            f"expr/{i}/roundtrip",
            lambda env=env: describe(evc, evc.deserialization(evc.serialization(env))),
        )
        cases.add(
            f"expr/{i}/pickle",
            lambda env=env: describe(evc, pickle.loads(pickle.dumps(env))),
        )
        if not risky:  # the reference also segfaults in copy() on these inputs
            cases.add(
                f"expr/{i}/copy",
                lambda env=env: (
                    describe(evc, env.copy()),
                    describe(evc, copy.copy(env)),
                ),
            )
        cases.add(
            f"expr/{i}/decode_roundtrip",
            lambda env=env, res=res: decode_all(
                evc, evc.deserialization(evc.serialization(env)), res
            ),
        )

    # A realistic pulse schedule.
    def schedule():
        xy = evc.GaussianDRAG(t0=15, w=10, amp=1.0, coef=0.5, df=0.1, phase=0.0)
        seq = evc.align(
            xy, np.arange(200) * 50.0, np.linspace(0.1, 1, 200).astype(np.float32)
        )
        seq += evc.mix(
            evc.align(evc.Flattop(0, 30, 2, 2, 0.3), np.arange(20) * 500.0),
            0.05,
            0.3,
            True,
        )
        return decode_all(evc, seq, 0.5, [(-50.0, 10250.0), (100.3, 5000.7)])

    cases.add("schedule", schedule)

    # Structural rewriting rules.
    g = evc.Gaussian(10, 4, 0.7)
    d = evc.GaussianDRAG(15, 10, 1.0, 0.5, 0.1, 0.2)
    rr = evc.Rect(0, 5, 1.0)
    m = evc.mix(g, 0.05, 0.1, True)
    ops = {
        "g+d": lambda: g + d,
        "g+0": lambda: g + 0,
        "0+g": lambda: 0 + g,
        "g+0.0": lambda: g + 0.0,
        "g+1.5": lambda: g + 1.5,
        "1.5+g": lambda: 1.5 + g,
        "g-d": lambda: g - d,
        "-g": lambda: -g,
        "g/2": lambda: g / 2,
        "g*2": lambda: g * 2,
        "2*g": lambda: 2 * g,
        "g*1j": lambda: g * 1j,
        "g*(2+0j)": lambda: g * (2 + 0j),
        "f32*g": lambda: np.float32(2) * g,
        "g*c64": lambda: g * np.complex64(1j),
        "g*c128": lambda: g * np.complex128(1j),
        "g*True": lambda: g * True,
        "g*'a'": lambda: g * "a",
        "g>>3": lambda: g >> 3,
        "g>>f32": lambda: g >> np.float32(0.1),
        "(g>>3)>>2": lambda: (g >> 3) >> 2,
        "(g*2)*3": lambda: (g * 2) * 3,
        "(g*2)>>3": lambda: (g * 2) >> 3,
        "(g>>3)*2": lambda: (g >> 3) * 2,
        "(g+d)*2": lambda: (g + d) * 2,
        "(g+d)>>2": lambda: (g + d) >> 2,
        "(g*2)*1j": lambda: (g * 2) * 1j,
        "(g*1j)*2": lambda: (g * 1j) * 2,
        "(g*1j)*1j": lambda: (g * 1j) * (0.5 + 2j),
        "sum": lambda: sum([g, d, g]),
        "m>>3": lambda: m >> 3,
        "m*2": lambda: m * 2,
        "(g+d)+(g+d)": lambda: (g + d) + (g + d),
        "g+(g+d)": lambda: g + (g + d),
        "(g>>1)>>(-1)": lambda: (g >> 1) >> (-1),
        "mix(g*1j)": lambda: evc.mix(g * 1j, 0.1, 0, True),
        "mix(g>>2)": lambda: evc.mix(g >> 2, 0.1, 0, True),
        "mix(mix)": lambda: evc.mix(m, 0.1, 0.0, False),
        "mix(align)": lambda: evc.mix(evc.align(g * 2, arr([0.0, 1.0])), 0.1, 0, True),
        "mix int dyn": lambda: evc.mix(g, 0.1, 0, 1),
        "mix np.bool": lambda: evc.mix(g, 0.1, 0, np.bool_(False)),
        "align(g*2)": lambda: evc.align(g * 2, arr([0.0, 1.0])),
        "align(m)": lambda: evc.align(m, arr([0.0, 1.0])),
        "align(g+d,amp)": lambda: evc.align(g + d, arr([0.0, 1.0]), arr([1.0, 2.0])),
        "align c128 amp": lambda: evc.align(g, arr([0.0, 1.0]), arr([1.0, 2j])),
        "align c64 amp": lambda: evc.align(
            g * 2, arr([0.0, 1.0]), arr([1.0, 2j], dtype=np.complex64)
        ),
        "align int amp": lambda: evc.align(g, arr([0.0, 1.0]), arr([1, 2])),
        "align list": lambda: evc.align(g, [0.0, 10.0]),
        "align list amp": lambda: evc.align(g, [0.0, 10.0], [1.0, 3.0]),
        "align 2d": lambda: evc.align(g, np.zeros((2, 2))),
        "align strided": lambda: evc.align(g, np.arange(10.0)[::3]),
        "align mismatch": lambda: evc.align(g, arr([0.0, 1.0]), arr([1.0, 2.0, 3.0])),
        "align int dt": lambda: evc.align(g, arr([0, 7])),
        "align scalar": lambda: evc.align(g, 3.0),
    }
    for name, fn in ops.items():
        cases.add(
            f"ops/{name}",
            lambda fn=fn: (lambda e: describe(evc, e) if hasattr(e, "start") else e)(
                fn()
            ),
        )
    for name in [
        "g*c64",
        "align c128 amp",
        "align c64 amp",
        "(g*1j)*1j",
        "f32*g",
        "align(g+d,amp)",
        "mix(mix)",
    ]:
        cases.add(f"ops-decode/{name}", lambda fn=ops[name]: decode_all(evc, fn()))

    def iadd_cases():
        s1 = g + d
        s2 = s1 + rr
        s1 += g
        s3 = s1 + 0
        s1 += rr
        p = s1 * 2
        s1 += rr
        x = evc.Gaussian(1, 2, 3)
        y = x
        x += d
        return [repr(v) for v in (s1, s2, s3, p, x, y)] + [
            s3 is s1,
            y is x,
            (g + 0) is g,
        ]

    cases.add("ops/iadd", iadd_cases)
    # Empty sums crash the reference on start()/decode; only inspect what it supports.
    cases.add(
        "ops/align empty",
        lambda: (
            lambda e: (type(e).__name__, repr(e), e.is_complex(), evc.serialization(e))
        )(evc.align(g, arr([]))),
    )

    # split
    sp = {
        "basic": (
            lambda: evc.split(
                evc.align(g, arr([0.0, 100.0])), arr([-50.0, 60.0]), arr([50.0, 200.0])
            )
        ),
        "one block": (lambda: evc.split(g, arr([-50.0]), arr([50.0]))),
        "one block sum": (lambda: evc.split(g + d, arr([-50.0]), arr([100.0]))),
        "one block bad": (lambda: evc.split(g, arr([0.0]), arr([5.0]))),
        "no block": (
            lambda: evc.split(
                evc.align(g, arr([0.0, 100.0])), arr([-50.0, 200.0]), arr([50.0, 300.0])
            )
        ),
        "partial": (
            lambda: evc.split(
                evc.align(g, arr([0.0, 100.0])), arr([-50.0, 60.0]), arr([50.0, 105.0])
            )
        ),
        "nonsum multi": (lambda: evc.split(g, arr([-50.0, 60.0]), arr([50.0, 105.0]))),
        "nonsum none": (lambda: evc.split(g, arr([50.0, 60.0]), arr([55.0, 105.0]))),
        "complex empty": (
            lambda: evc.split(
                evc.align(d, arr([0.0])), arr([-50.0, 60.0]), arr([50.0, 105.0])
            )
        ),
        "complex mixed": (
            lambda: evc.split(g + (d >> 100), arr([-50.0, 60.0]), arr([50.0, 205.0]))
        ),
        "len mismatch": (lambda: evc.split(g, arr([-50.0, 60.0]), arr([50.0]))),
        "zero blocks": (lambda: evc.split(g, arr([]), arr([]))),
        "prod": (lambda: evc.split(g * 2, arr([-50.0, 60.0]), arr([50.0, 105.0]))),
        "nested sum": (
            lambda: evc.split((g + d) * 2, arr([-50.0, 60.0]), arr([50.0, 105.0]))
        ),
        "shared edge": (lambda: evc.split(g, arr([-2.0, 22.0]), arr([22.0, 30.0]))),
        "two blocks": (lambda: evc.split(g, arr([-5.0, -10.0]), arr([30.0, 40.0]))),
        "lists": (lambda: evc.split(g, [-50.0, 60.0], [50.0, 105.0])),
    }
    for name, fn in sp.items():

        def split_case(fn=fn):
            blocks = fn()
            return [describe(evc, b) for b in blocks] + [
                decode_all(evc, b) for b in blocks
            ]

        cases.add(f"split/{name}", split_case)

    # decode windows and edge cases
    def new_cache(r: float = 0.5) -> object:
        return evc.WaveCache(r)

    dec = {
        "start>end": lambda: evc.decode_envelope(g, new_cache(), 10.0, 5.0),
        "start>=t_end": lambda: evc.decode_envelope(g, new_cache(), 30.0, 40.0),
        "start=t_last": lambda: evc.decode_envelope(g, new_cache(), 22.5, 40.0),
        "end<t_start": lambda: evc.decode_envelope(g, new_cache(), -40.0, -30.0),
        "end=t_start": lambda: evc.decode_envelope(g, new_cache(), -40.0, -1.5),
        "start=end": lambda: evc.decode_envelope(g, new_cache(), 5.0, 5.0),
        "no wc": lambda: evc.decode_envelope(g),
        "None se": lambda: evc.decode_envelope(g, new_cache(), None, None),
        "int se": lambda: evc.decode_envelope(g, new_cache(), -5, 30),
        "Gaussian w=0": lambda: evc.decode_envelope(evc.Gaussian(0, 0, 1), new_cache()),
        "Rect tlen=0": lambda: evc.decode_envelope(evc.Rect(0, 0, 1), new_cache()),
        "Rect neg": lambda: evc.decode_envelope(evc.Rect(0, -3, 1), new_cache()),
        "Triangle tlen=0": lambda: evc.decode_envelope(
            evc.Triangle(0, 0, 1, True), new_cache()
        ),
        "huge t": lambda: evc.decode_envelope(
            evc.Gaussian(1e6 + 0.123, 4, 0.7), new_cache()
        ),
        "neg t": lambda: evc.decode_envelope(
            evc.Gaussian(-1e4 - 0.371, 4, 0.7), new_cache(), -1e4 - 30, -1e4 + 30
        ),
        "res 1/3": lambda: evc.decode_envelope(g >> 0.1, new_cache(1 / 3)),
        "res odd": lambda: evc.decode_envelope(d >> 0.123, new_cache(0.37), -3, 40),
        "writable": lambda: evc.decode_envelope(g, new_cache())[1].flags.writeable,
    }
    for name, fn in dec.items():
        cases.add(f"decode/{name}", fn)

    def cache_repr():
        wc = new_cache()
        envs = [
            g,
            g >> 0.25,
            g >> 10,
            d,
            rr,
            evc.mix(g, 0.05, 0.1, True),
            evc.mix(rr, 0.05, 0.1, False),
            evc.Custom(0.1, 0.3, arr([1.0, 2.0, 3.0])),
            evc.Triangle(0, 3, 1, True),
            evc.PowerSineCosine(0, 3, 1, 2, 1, 1.0, 0.0),
        ]
        for e in envs:
            evc.decode_envelope(e, wc)
        # Hash-map iteration order is unspecified; compare the set of lines.
        return sorted(repr(wc).splitlines())

    cases.add("cache/repr_lines", cache_repr)
    cases.add("cache/repr_empty", lambda: repr(new_cache(0.25)))
    for name, res in {
        "zero": 0,
        "neg": -1,
        "tiny": 1e-7,
        "atol": 1e-5,
        "nan": float("nan"),
        "int": 2,
    }.items():
        cases.add(
            f"cache/ctor/{name}",
            lambda res=res: (repr(new_cache(res)), new_cache(res).resolution()),
        )
    cases.add("cache/pickle", lambda: pickle.dumps(new_cache()))
    cases.add("cache/eq", lambda: new_cache() == new_cache())

    # Custom inputs
    cu = {
        "bad dt": lambda: evc.Custom(0, 0, np.ones(3)),
        "neg dt": lambda: evc.Custom(0, -1, np.ones(3)),
        "2d": lambda: evc.Custom(0, 1, np.ones((2, 2))),
        "2d bad dt": lambda: evc.Custom(0, 0, np.ones((2, 2))),
        "empty": lambda: evc.Custom(0, 1, np.ones(0)),
        "empty bad dt": lambda: evc.Custom(0, 0, np.ones(0)),
        "int": lambda: evc.Custom(0, 1, np.arange(3)),
        "uint8": lambda: evc.Custom(0, 1, np.arange(3, dtype=np.uint8)),
        "bool": lambda: evc.Custom(0, 1, np.array([True, False, True])),
        "f16": lambda: evc.Custom(0, 1, np.array([1, 2, 3], dtype=np.float16)),
        "f32": lambda: evc.Custom(0, 1, np.array([1, 2, 3], dtype=np.float32)),
        "c64": lambda: evc.Custom(0, 1, np.array([1, 2j, 3], dtype=np.complex64)),
        "list": lambda: evc.Custom(0, 1, [1, 2, 3]),
        "object": lambda: evc.Custom(0, 1, np.array([1, "a"], dtype=object)),
        "str": lambda: evc.Custom(0, 1, np.array(["a", "b"])),
        "strided": lambda: evc.Custom(0, 1, np.arange(10.0)[::3]),
        "0d": lambda: evc.Custom(0, 1, np.array(1.0)),
        "nan": lambda: evc.Custom(0, 1, np.array([np.nan, 1.0])),
    }
    for name, fn in cu.items():

        def custom_case(fn=fn):
            env = fn()
            return (
                describe(evc, env),
                decode_all(evc, env, 0.3),
                decode_all(evc, env * 1j, 0.5),
            )

        cases.add(f"custom/{name}", custom_case)

    # Constructor argument handling
    ctor = {
        "kwargs": lambda: evc.Gaussian(t0=1, w=2, amp=3),
        "missing": lambda: evc.Gaussian(1, 2),
        "extra": lambda: evc.Gaussian(1, 2, 3, 4),
        "str": lambda: evc.Gaussian("1", 2, 3),
        "np scalars": lambda: evc.Gaussian(
            np.float32(1.1), np.int64(2), np.float64(0.3)
        ),
        "Triangle int fall": lambda: evc.Triangle(0, 1, 1, 1),
        "Triangle np bool": lambda: evc.Triangle(0, 1, 1, np.bool_(True)),
        "PSC float m": lambda: evc.PowerSineCosine(0, 1, 1, 2.0, 1, 1.0, 0.0),
        "XPower neg m": lambda: evc.GaussianXPower(0, 1, 1, -1),
        "amp f32 round": lambda: evc.Gaussian(0, 1, 0.1),
        "AbstractEnvelope()": lambda: evc.AbstractEnvelope(),
        "Envelope()": lambda: evc.Envelope(),
        "EnvSum()": lambda: evc.EnvSum(),
        "MixExp()": lambda: evc.MixExp(),
    }
    for name, fn in ctor.items():
        cases.add(f"ctor/{name}", lambda fn=fn: (lambda e: describe(evc, e))(fn()))

    # Serialization robustness
    good = evc.serialization(evc.mix((g >> 1) * 2 + d, 0.05, 0.0, True) + rr)
    ser = {
        "junk tag": lambda: evc.deserialization(
            b"\x01\x00\x00\x00\x00\x00\x00\x00\x63"
        ),
        "lone MixExp": lambda: evc.deserialization(
            struct.pack("<QBddddB", 1, 10, -2.0, 22.0, 0.1, 0.2, 1)
        ),
        "lone EnvShift": lambda: evc.deserialization(
            struct.pack("<QBdBddfI", 2, 12, 1.5, 0, 10.0, 4.0, 0.7, 1)
        ),
        "prod complex": lambda: evc.deserialization(
            struct.pack("<QBBffBddfI", 2, 13, 1, 0.5, -2.0, 0, 10.0, 4.0, 0.7, 1)
        ),
        "nested": lambda: evc.deserialization(good),
        "shared": lambda: evc.serialization(g + g),
        "shared deep": lambda: evc.serialization(
            evc.align(g * 2, arr([0.0, 1.0])) + evc.align(g * 2, arr([0.0, 1.0]))
        ),
        "str arg": lambda: evc.deserialization("abc"),
        "setstate wrong type": lambda: evc.Gaussian.__setstate__(
            evc.Gaussian(0, 1, 1), evc.serialization(rr)
        ),
        "pickle WaveCache": lambda: pickle.dumps(new_cache()),
        "pickle each": lambda: [
            repr(pickle.loads(pickle.dumps(x)))
            for x in (
                g,
                d,
                rr,
                m,
                g * 2,
                g * 1j,
                g >> 1,
                g + d,
                evc.mix(g, 0.1, 0, True),
            )
        ],
        "pickle types": lambda: [
            type(pickle.loads(pickle.dumps(x))).__name__
            for x in (g, d, rr, m, g * 2, g * 1j, g >> 1, g + d)
        ],
    }
    for name, fn in ser.items():
        cases.add(
            f"serial/{name}",
            lambda fn=fn: (lambda e: describe(evc, e) if hasattr(e, "start") else e)(
                fn()
            ),
        )

    # Module surface
    cases.add(
        "api/names", lambda: sorted(n for n in dir(evc) if not n.startswith("__"))
    )
    cases.add(
        "api/mro",
        lambda: {
            n: [c.__name__ for c in getattr(evc, n).__mro__]
            for n in dir(evc)
            if isinstance(getattr(evc, n), type)
        },
    )
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--module", default="envelopes.envelopes_cpp")
    ap.add_argument("--path", help="directory prepended to sys.path")
    ap.add_argument(
        "--trace", action="store_true", help="print each case name to stderr"
    )
    args = ap.parse_args()
    if args.path:
        sys.path.insert(0, args.path)
    evc = importlib.import_module(args.module)

    results = build(evc).run()
    with open(args.out, "wb") as f:
        pickle.dump({"module": evc.__file__, "results": results}, f)
    print(f"{len(results)} cases -> {args.out} ({evc.__file__})")


if __name__ == "__main__":
    main()
