# Julia-side loader for the category-annotated POSNoise companion files
# (posnoise_lists/aligned/, built by data_prep/build_aligned_lists.py).
#
# Two consumers on this side of the pipeline:
#
#   * Shared symbol atoms in the hyperdimensional encoder: tokens that carry
#     the same functional class across languages share one class atom, so
#     multilingual profiles agree wherever the grammar agrees; unlisted tokens
#     keep their language-local random atoms.  Use `token_atoms` with the
#     encoder's existing random-atom generator as `fresh`.
#
#   * Readable literals in per-author profile classifiers: name a feature by
#     the class string of its tokens ("AUX_ADV_PRON") instead of raw indices.
#     Use `class_of` when building the feature vocabulary.
#
# Classes: AUX LVERB ADV DET ADP PRON CCONJ SCONJ PART INTJ NUM MWE (OTH/UNK
# are treated as classless and fall back).

module AlignedAtoms

export ALIGNED_CLASSES, load_aligned, class_of, token_atoms

const ALIGNED_CLASSES = ["AUX", "LVERB", "ADV", "DET", "ADP", "PRON",
                         "CCONJ", "SCONJ", "PART", "INTJ", "NUM", "MWE"]
const NOCLASS = Set(["OTH", "UNK"])

"entry => class for one language code; empty Dict if the file is missing."
function load_aligned(code::AbstractString;
                      dir::AbstractString=joinpath(@__DIR__, "..",
                                                   "posnoise_lists", "aligned"))
    files = sort(filter(f -> startswith(f, "POSNoise_Aligned_" *
                                        uppercasefirst(lowercase(code))),
                        isdir(dir) ? readdir(dir) : String[]))
    isempty(files) && return Dict{String,String}()
    table = Dict{String,String}()
    for line in eachline(joinpath(dir, files[end]))
        (isempty(line) || startswith(line, "#")) && continue
        f = split(line, '\t')
        length(f) >= 3 && (table[f[1]] = f[3])
    end
    table
end

"Functional class of a kept token, or `nothing` when unlisted or classless."
function class_of(token::AbstractString, table::Dict{String,String})
    cls = get(table, lowercase(token), nothing)
    (cls === nothing || cls in NOCLASS) ? nothing : cls
end

"""
    token_atoms(tokens, table, class_atoms, fresh)

Atom assignment with cross-language sharing: a token whose class is known gets
that class's shared atom from `class_atoms::Dict{String,T}`; any other token
gets `fresh(token)` (the encoder's usual language-local random atom). Returns
Dict token => atom.
"""
function token_atoms(tokens, table::Dict{String,String},
                     class_atoms::Dict{String,T}, fresh) where {T}
    out = Dict{String,T}()
    for t in tokens
        cls = class_of(t, table)
        out[t] = cls === nothing ? fresh(t) : class_atoms[cls]
    end
    out
end

end # module
