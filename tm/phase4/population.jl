# Population-relative pair features: give the HDC-TM the property that makes λ_G
# forensically meaningful -- scoring a pair against a REFERENCE POPULATION rather than
# scoring bare similarity.
#
# WHY NOT THE OBVIOUS THING (worked out algebraically before implementing; see the
# project notes). The tempting move is to z-score the pair feature z_b = Ũ_b·K̃_b against
# the reference documents R_j. It collapses: z_b is bilinear and Ũ_b is constant across
# j, so
#       mean_j(Ũ_b·R̃_j,b) = Ũ_b·R̄_b        std_j(Ũ_b·R̃_j,b) = |Ũ_b|·s_b
# and the z-score reduces to sign(Ũ_b)·(K̃_b−R̄_b)/s_b -- the questioned document's
# MAGNITUDE cancels, which reproduces exactly the sign-only binarisation that was
# measured to fail in phase 3. Plain mean-subtraction is close to a no-op too, because
# `cen()` already subtracts the corpus mean (the Burrows-Delta step), so K̃_b − R̄_b ≈ K̃_b
# whenever R̄ is a global average.
#
# What actually distinguishes λ_G is LOCALITY and competing explanations: its r reference
# models are drawn PER CASE and SIZE-MATCHED to the known document. Hence the two
# constructions below.
#
#   1. case_reference_mean  -- R̄ from the r reference sketches NEAREST to K (a
#      case-relevant population, i.e. the candidate's own style family), so subtracting
#      it discounts what the candidate shares with writers like them, not with the corpus
#      at large. This is the part that attacks the relevant-population problem.
#
#   2. population_bits!     -- a RANK block, which survives the bilinear cancellation
#      because ranks are invariant to the positive scaling that killed the z-score:
#        (a) per-dimension percentile of K̃_b within {R̃_j,b}, coarsely thermometer-coded
#            over a strided subset of dimensions (the full D would swamp the input);
#        (b) an impostor-style count: over how many reference documents does K agree with
#            Q at least as strongly as the reference does -- the Koppel & Winter statistic,
#            per whole document rather than per dimension.
#
# Both are cheap: (1) is O(r·D) per case, (2) is O(r·D) per case.

using LinearAlgebra: dot

"""
    case_reference_mean(K, R; r=30) -> Vector{Float64}

Mean of the `r` reference sketches most similar to the known sketch `K`.
`R` is a vector of centred reference sketches. Falls back to the global mean of `R`
when fewer than `r` references exist.
"""
function case_reference_mean(K::Vector{Float64}, R::Vector{Vector{Float64}}; r::Int = 30)
    isempty(R) && return zeros(Float64, length(K))
    if length(R) <= r
        return sum(R) ./ length(R)
    end
    sims = [dot(K, x) for x in R]
    idx = partialsortperm(sims, 1:r, rev = true)      # the candidate's own style family
    m = zeros(Float64, length(K))
    @inbounds for i in idx
        m .+= R[i]
    end
    m ./ r
end

"""
    population_bits!(bits, off, U, K, R, stride, nq) -> Int

Append the rank-based population block and return the number of bits written.

`U`, `K` are the centred questioned/known sketches, `R` the reference sketches.
`stride` subsamples dimensions (D/stride percentile features, 2 bits each);
`nq` is the number of impostor-count thermometer bits.
"""
function population_bits!(bits::Vector{Bool}, off::Int, U::Vector{Float64},
                          K::Vector{Float64}, R::Vector{Vector{Float64}},
                          stride::Int, nq::Int)
    D = length(K); nR = length(R)
    p = off
    if nR == 0
        @inbounds for i in (off+1):(off + 2 * length(1:stride:D) + nq)
            bits[i] = false
        end
        return 2 * length(1:stride:D) + nq
    end

    # (a) per-dimension percentile of K within the reference population
    @inbounds for b in 1:stride:D
        c = 0
        for j in 1:nR
            c += (R[j][b] < K[b]) ? 1 : 0
        end
        q = c / nR
        bits[p+1] = q > 0.5
        bits[p+2] = q > 0.85                      # "unusually high for this population"
        p += 2
    end

    # (b) impostor count: how often does K agree with U at least as much as a reference
    # does? A rank statistic over whole documents, so positive rescaling of U cannot
    # distort it -- which is precisely why it survives where the z-score collapsed.
    zK = dot(U, K)
    beat = 0
    @inbounds for j in 1:nR
        beat += (zK >= dot(U, R[j])) ? 1 : 0
    end
    frac = beat / nR
    @inbounds for t in 1:nq
        bits[p+t] = frac >= t / (nq + 1)
    end
    p += nq
    return p - off
end
