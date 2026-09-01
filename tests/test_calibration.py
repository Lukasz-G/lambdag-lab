# Forensic-cost semantics: the silent system costs exactly 1 bit; the PAV floor
# never exceeds the total; the floor is invariant under monotone maps.
import numpy as np

from lambdag import cllr, cllr_min


def test_silence_costs_one_bit():
    z = np.zeros(50)
    assert cllr(z, z) == 1.0


def test_perfect_system_costs_nothing():
    assert cllr(np.full(50, 12.0), np.full(50, -12.0)) < 1e-3


def test_floor_leq_total_and_monotone_invariant():
    rng = np.random.default_rng(0)
    same = rng.normal(1.0, 1.0, 200)
    diff = rng.normal(-1.0, 1.5, 200)
    assert cllr_min(same, diff) <= cllr(same, diff) + 1e-12
    a = cllr_min(same, diff)
    b = cllr_min(2.5 * same + 3.0, 2.5 * diff + 3.0)
    assert abs(a - b) < 1e-9


def test_wrong_way_confidence_is_punished():
    # confidently wrong LRs must cost far more than silence
    same = np.full(20, -6.0)
    diff = np.full(20, 6.0)
    assert cllr(same, diff) > 5.0
