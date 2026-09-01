# The padding convention: each sentence carries one <EOS> slot at the end of
# its per-token evidence vector (n <BOS> are context-only and get no slot).
import random

from lambdag import LambdaG

VOCAB = [f"w{i}" for i in range(10)]


def test_token_lambda_has_one_eos_slot_per_sentence():
    rng = random.Random(3)
    mk = lambda n: [[rng.choice(VOCAB) for _ in range(rng.randint(3, 9))]
                    for _ in range(n)]
    lg = LambdaG(N=4, r=3, engine="kn", random_state=0)
    lg.set_reference(mk(80))
    quest = mk(6)
    res = lg.score(quest, mk(15), with_details=True)
    assert len(res.token_lambda) == len(quest)
    for sent, tok in zip(quest, res.token_lambda):
        assert len(tok) == len(sent) + 1          # tokens + <EOS>
    # summed sentence evidence must reproduce the total
    total = sum(float(t.sum()) for t in res.token_lambda)
    assert abs(total - res.lambda_G) < 1e-8
    # N(Q)/V1(Q) are masked-token statistics without padding
    assert res.n_query_tokens == sum(len(s) for s in quest)
