"""Unified test grids (task §4.3) — exact-decimal generation, locked ordering and IDs.

Grids reproduce the RMBPO author configs (configs/robust_mbpo/<task>-friction-torso.py):
  hopper:      torso mass 2.50..4.75 step .25 (10) x friction mult 0.60..1.40 step .10 (9)  -> 90
  halfcheetah: torso mass 3.00..9.50 step .50 (14) x friction mult 0.70..3.70 step .30 (11) -> 154
  walker2d:    torso mass 1..8 step 1 (8) x friction mult {0.25,0.50,...,2.00} (8)          -> 64
               (the common 64-cell subset of the author's 80-cell walker grid)

Walker2d author grid uses friction offsets [-1.0,-0.75,-0.5,-0.25,0,0.2,0.25,0.5,0.75,1.0];
the common subset keeps the 8 offsets [-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.0] which map to
multipliers 0.25..2.00 given nominal sliding friction 1.0. All values are produced from
integers and divided by a power of ten so no float drift can shift a cell boundary.

Ordering: mass ascending (outer), friction ascending (inner). condition_id = int index;
grid points are also given a stable string id "m<mass>_f<mult>".
"""
from __future__ import annotations

from fractions import Fraction

GRID_SPECS = {
    # task: (mass numerators, mass denominator, friction numerators, friction denominator)
    "hopper": ([250 + 25 * i for i in range(10)], 100,
               [60 + 10 * j for j in range(9)], 100),
    "halfcheetah": ([300 + 50 * i for i in range(14)], 100,
                    [70 + 30 * j for j in range(11)], 100),
    "walker2d": ([100 * (i + 1) for i in range(8)], 100,
                 [25 * (j + 1) for j in range(8)], 100),
}

MASS_RANGE_TRAIN = (Fraction(9, 10), Fraction(11, 10))     # m/m0 ~ U[0.90, 1.10]
FRICTION_RANGE_TRAIN = (Fraction(9, 10), Fraction(11, 10))  # mult ~ U[0.90, 1.10]

_cached = {}


def nominal_torso_mass(task: str) -> float:
    """True nominal torso mass read from the pristine raw environment (task §4.1)."""
    if "nom" not in _cached:
        import gymnasium as gym
        raw = {"hopper": "Hopper-v4", "halfcheetah": "HalfCheetah-v4", "walker2d": "Walker2d-v4"}
        masses = {}
        for t, eid in raw.items():
            e = gym.make(eid)
            masses[t] = float(e.unwrapped.model.body_mass[1])
            e.close()
        _cached["nom"] = masses
    return _cached["nom"][task]


def grid(task: str, with_nominal: bool = False):
    """Return list of dicts (locked order): {condition_id, mass_kg, friction_multiplier,
    is_ood, is_nominal, condition_str}. Integer id for grid cells 0..N-1; the nominal
    condition (appended when with_nominal) gets id -1."""
    if task not in _cached:
        mass_num, mass_den, fric_num, fric_den = GRID_SPECS[task]
        m0 = Fraction(nominal_torso_mass(task)).limit_denominator(1000)
        m_lo, m_hi = MASS_RANGE_TRAIN
        f_lo, f_hi = FRICTION_RANGE_TRAIN
        out = []
        k = 0
        for mn in mass_num:
            mass = Fraction(mn, mass_den)
            for fn in fric_num:
                fric = Fraction(fn, fric_den)
                mass_mult = mass / m0
                ood = not (m_lo <= mass_mult <= m_hi and f_lo <= fric <= f_hi)
                out.append({
                    "condition_id": k,
                    "mass_kg": float(mass),
                    "friction_multiplier": float(fric),
                    "is_ood": bool(ood),
                    "is_nominal": False,
                    "condition_str": f"m{mn:04d}x{mass_den:03d}_f{fn:04d}x{fric_den:03d}",
                })
                k += 1
        _cached[task] = out
    out = list(_cached[task])
    if with_nominal:
        m0 = nominal_torso_mass(task)
        out.append({
            "condition_id": -1,
            "mass_kg": float(m0),
            "friction_multiplier": 1.0,
            "is_ood": False,
            "is_nominal": True,
            "condition_str": "nominal",
        })
    return out


def n_conditions(task: str) -> int:
    return len(grid(task))


def summary() -> dict:
    s = {}
    for t in GRID_SPECS:
        g = grid(t)
        s[t] = {"n": len(g), "n_ood": sum(1 for c in g if c["is_ood"]),
                "nominal_mass": nominal_torso_mass(t)}
    return s
