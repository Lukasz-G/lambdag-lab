# THE regression oracle: HPY with concentration 0, minimal tables and the KN
# discount reproduces interpolated Kneser-Ney EXACTLY (~1e-12). If a refactor
# breaks this equality, the refactor is wrong. See docs/design.md.
import random

import numpy as np
import pytest

from lambdag import LambdaG

VOCAB = [f"w{i}" for i in range(14)] + [",", "."]


def _sentences(rng, n, lo=4, hi=12):
    return [[rng.choice(VOCAB) for _ in range(rng.randint(lo, hi))]
            for _ in range(n)]


@pytest.fixture(scope="module")
def data():
    rng = random.Random(7)
    return (_sentences(rng, 200),   # reference
            _sentences(rng, 30),    # known
            _sentences(rng, 20))    # questioned


def _score(engine, engine_params, data):
    ref, known, quest = data
    lg = LambdaG(N=5, r=5, engine=engine, random_state=0,
                 **({"engine_params": engine_params} if engine_params else {}))
    lg.set_reference(ref)
    return lg.score(quest, known, with_details=True)


def test_kn_equals_hpy_theta0_minimal(data):
    kn = _score("kn", None, data)
    hpy = _score("hpy", {"concentration": 0.0, "table_estimator": "minimal",
                         "discount": 0.75}, data)
    # end-to-end
    assert kn.lambda_G == pytest.approx(hpy.lambda_G, abs=1e-9)
    # token-by-token
    for a, b in zip(kn.token_lambda, hpy.token_lambda):
        np.testing.assert_allclose(a, b, atol=1e-9)


def test_score_deterministic(data):
    a = _score("kn", None, data)
    b = _score("kn", None, data)
    assert a.lambda_G == b.lambda_G
