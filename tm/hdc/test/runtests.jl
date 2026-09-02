using HDC, Random, Test

@testset "packed primitives" begin
    sp = Space(1000); rng = MersenneTwister(1)
    a = randhv(sp, rng); b = randhv(sp, rng)
    @test a[end] & ~sp.tail == 0                         # padding stays 0
    @test b[end] & ~sp.tail == 0
    @test abs(hamming(a, b) - sp.D / 2) < sp.D * 0.12    # random pair ≈ D/2 apart
    @test sim(sp, a, a) == 1.0

    c = newhv(sp); bind!(c, a, b)                         # bind is involutive
    d = newhv(sp); bind!(d, c, b)
    @test d == a

    acc = Vector{Int32}(undef, sp.W * 64); out = newhv(sp)
    bundle!(sp, out, [a, a, a], acc, rng)                # majority of copies = the copy
    @test out == a

    x = zeros(UInt64, sp.W); y = zeros(UInt64, sp.W); z = zeros(UInt64, sp.W)
    setbit!(x, 1); setbit!(y, 1); setbit!(z, 2)          # bit1 in 2/3 -> set; bit2 in 1/3 -> unset
    bundle!(sp, out, [x, y, z], acc, rng)
    @test getbit(out, 1) && !getbit(out, 2)

    p = random_setbit(x, rng); @test p == 1              # x has exactly one set bit
    @test random_setbit(zeros(UInt64, sp.W), rng) == 0
end

@testset "graded level codes" begin
    sp = Space(4000); rng = MersenneTwister(2)
    P = level_codes(sp, 8, rng)
    s(i, j) = sim(sp, @view(P[:, i]), @view(P[:, j]))
    @test s(1, 2) > s(1, 5) > s(1, 8)                    # similarity decays with distance
    @test s(1, 8) < 0.25                                 # endpoints ≈ orthogonal
end

@testset "window encoder" begin
    vocab = ["w$i" for i in 1:60]; w2i = Dict(w => i for (i, w) in enumerate(vocab))
    cb = Codebook(2000, vocab, w2i, fill(10, 60); sub = 3, rng = MersenneTwister(3))
    enc = Encoder(cb; stride = 1, pos = POS_GRADED, maxwin = 20, rng = MersenneTwister(4))

    win = [1, 2, 3, 4, 5, 6, 7, 8]
    v  = encode_window(enc, win, MersenneTwister(9))
    v2 = encode_window(enc, win, MersenneTwister(9))
    @test v[end] & ~cb.sp.tail == 0                      # clean padding
    @test v == v2                                        # deterministic for fixed rng

    near = copy(win); near[end] = 20                     # one token changed
    vn = encode_window(enc, near, MersenneTwister(9))
    far = [rand(MersenneTwister(7), 1:60) for _ in 1:8]
    vf = encode_window(enc, far, MersenneTwister(9))
    @test sim(cb.sp, v, vn) > sim(cb.sp, v, vf)          # similar windows -> similar vectors

    # short window (< sub) still encodes without error
    @test encode_window(enc, [1, 2], MersenneTwister(9)) isa Vector{UInt64}
end

@testset "self-referential learner (smoke) + collapse contrast" begin
    rng = MersenneTwister(10)
    topics = [["a$i" for i in 1:6], ["b$i" for i in 1:6], ["c$i" for i in 1:6]]
    labels = Dict{String,Int}()
    for (t, ws) in enumerate(topics), w in ws; labels[w] = t; end
    sents = [rand(rng, topics[rand(rng, 1:3)], 6) for _ in 1:3000]
    vocab, w2i, counts, ids = build_vocab(sents; min_count = 2)

    D = 2000
    cb = Codebook(D, vocab, w2i, counts; sub = 3, rng = MersenneTwister(11))
    E0 = copy(cb.E)
    enc = Encoder(cb; stride = 1, pos = POS_GRADED, maxwin = 12, rng = MersenneTwister(12))
    d_init = diagnostics(cb; labels = labels, rng = MersenneTwister(1))

    train!(enc, ids, AttractRepel(4, 5, 5); window = 3, epochs = 5, rng = MersenneTwister(13))
    d_ar = diagnostics(cb; labels = labels, rng = MersenneTwister(1))
    @test cb.E != E0                                     # embeddings actually moved
    @test all(isfinite, (d_ar.mean_sim, d_ar.sim_std, d_ar.bit_entropy))
    @test !isnan(d_ar.purity)

    # pure attraction should collapse MORE (higher mean pairwise similarity) than attract+repel
    cb2 = Codebook(D, vocab, w2i, counts; sub = 3, rng = MersenneTwister(11))
    enc2 = Encoder(cb2; stride = 1, pos = POS_GRADED, maxwin = 12, rng = MersenneTwister(12))
    train!(enc2, ids, Surprise(4); window = 3, epochs = 5, rng = MersenneTwister(13))
    d_su = diagnostics(cb2; labels = labels, rng = MersenneTwister(1))
    @test d_su.mean_sim > d_init.mean_sim                # attraction raises similarity
    @test d_su.mean_sim > d_ar.mean_sim                  # repulsion holds words apart

    # capacity-conserving rule preserves Hamming weight exactly
    cb3 = Codebook(D, vocab, w2i, counts; sub = 3, rng = MersenneTwister(11))
    w_before = [hamweight(@view cb3.E[:, v]) for v in 1:length(vocab)]
    enc3 = Encoder(cb3; stride = 1, pos = POS_GRADED, maxwin = 12, rng = MersenneTwister(12))
    train!(enc3, ids, SparseCapacity(3); window = 3, epochs = 2, rng = MersenneTwister(13))
    w_after = [hamweight(@view cb3.E[:, v]) for v in 1:length(vocab)]
    @test w_before == w_after
end

@testset "character-structured word forms" begin
    words = ["laufen", "laufend", "lauf", "haus", "hausmeister", "katze", "hund", "gelaufen"]
    ce = CharEncoder(words, 4000; sub = 3, pos = POS_NONE, rng = MersenneTwister(20))
    sp = ce.enc.cb.sp; rng = MersenneTwister(21)
    f(w) = word_form(ce, w, rng)
    @test sim(sp, f("laufen"), f("laufend")) > sim(sp, f("laufen"), f("haus"))   # shared stem
    @test sim(sp, f("haus"), f("hausmeister")) > sim(sp, f("haus"), f("katze"))  # compound
    @test sim(sp, f("gelaufen"), f("laufen")) > sim(sp, f("gelaufen"), f("katze")) # OOV via chars

    vocab, w2i, counts, ids = build_vocab([words]; min_count = 1)
    cb = Codebook(4000, vocab, w2i, counts; sub = 3, rng = MersenneTwister(22))
    ce2 = CharEncoder(vocab, 4000; sub = 3, pos = POS_NONE, rng = MersenneTwister(22))
    init_from_forms!(cb, ce2; rng = MersenneTwister(23))
    @test hamweight(@view cb.E[:, w2i["haus"]]) > 0                              # atoms initialised
    # words sharing a stem are neighbours already at init (before any context learning)
    nb = first.(nearest(cb, "laufen"; k = 3))
    @test "laufend" in nb || "lauf" in nb
end

@testset "form anchor persists character structure through training" begin
    rng = MersenneTwister(30)
    words = ["laufen", "laufend", "lauf", "gelaufen", "haus", "hausmeister", "hausen",
             "katze", "hund", "gehen", "geht"]
    sents = [rand(rng, words, 6) for _ in 1:1500]
    vocab, w2i, counts, ids = build_vocab(sents; min_count = 1)
    D = 3000
    cb = Codebook(D, vocab, w2i, counts; sub = 3, formhold = D ÷ 2, rng = MersenneTwister(31))
    ce = CharEncoder(vocab, D; sub = 3, pos = POS_NONE, rng = MersenneTwister(31))
    init_from_forms!(cb, ce; rng = MersenneTwister(32))
    Eb = copy(cb.E)
    enc = Encoder(cb; stride = 1, pos = POS_GRADED, maxwin = 12, rng = MersenneTwister(33))
    train!(enc, ids, AttractRepel(4, 5, 5); window = 3, epochs = 4, rng = MersenneTwister(34))
    # frozen region (bits where writable==0) is byte-for-byte unchanged
    @test all((cb.E[i, v] & ~cb.writable[i]) == (Eb[i, v] & ~cb.writable[i])
              for i in 1:cb.sp.W, v in eachindex(vocab))
    @test cb.E != Eb                                        # writable bits did change
    nb = first.(nearest(cb, "laufen"; k = 4))               # char structure survives training
    @test any(w in nb for w in ("laufend", "lauf", "gelaufen"))
end
