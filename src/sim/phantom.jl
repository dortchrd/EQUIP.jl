# =================================================================================================
# Digital phantoms: parameter maps with spatial structure, from a label image
#
#   labels (integers) ──phantom(labels, tissues)──▶ maps (a named tuple, like `truth`)
#
# Later: smooth variation within a tissue, partial volume at edges, a B1 field, brain-like
# label images from segmentations (BrainWeb, KomaMRI's brain phantom).
# =================================================================================================

"""
    phantom(labels, tissues) -> named tuple of maps

Parameter maps from an integer label image: `tissues` gives each label's parameters, e.g.
`[1 => truth(m, :GM), 2 => truth(m, :WM)]`. Every tissue gives the same parameter names: numbers,
which become maps, or ties (`R1m = :R1f`), which must be the same in every tissue and are kept.
Voxels with any other label (e.g. 0, the background) hold NaN. The result is ready for
[`simulate`](@ref), and is the truth for [`compare`](@ref).

```julia
m = T2Decay(TE = 0.01:0.01:0.32)
labels = labelshapes((64, 64))
truthmaps = phantom(labels, [1 => truth(m, :GM), 2 => truth(m, :WM), 3 => truth(m, :blood)])
data = simulate(m, truthmaps; snr = 50)
res = fit(m, data; mask = labels .> 0)
compare(res, truthmaps; labels)
```
"""
function phantom(labels::AbstractArray{<:Integer}, tissues)
    list = collect(tissues)                                   # label => named tuple pairs
    isempty(list) && throw(ArgumentError("give at least one tissue, e.g. [1 => truth(m, :WM)]"))
    names = keys(last(first(list)))
    all(p -> keys(last(p)) == names, list) ||
        throw(ArgumentError("every tissue must give the same parameters, in the same order"))
    # A tie is kept as it is (the same for every voxel); every number becomes a map.
    tied(n) = _istie(last(first(list))[n])
    for n in names
        if tied(n)
            all(p -> last(p)[n] == last(first(list))[n], list) ||
                throw(ArgumentError("`$n` is a tie; it must be the same tie in every tissue"))
        else
            all(p -> last(p)[n] isa Real, list) ||
                throw(ArgumentError("phantom values must be numbers or ties; `$n` is not"))
        end
    end
    maps = NamedTuple{names}(ntuple(i -> tied(names[i]) ? last(first(list))[names[i]] :
                                         fill(NaN, size(labels)), length(names)))
    for (label, t) in list
        voxels = findall(==(label), labels)
        for n in names
            tied(n) || (maps[n][voxels] .= t[n])
        end
    end
    return maps
end

"""
    labelshapes(dims) -> label image

A synthetic label image of size `dims` (2-D or 3-D) that needs no data: an outer ellipse
(label 1, e.g. gray matter), an inner one (label 2, e.g. white matter), a small disc in it
(label 3, e.g. a lesion), and background (label 0).
"""
function labelshapes(dims::Dims)
    N = length(dims)
    N in (2, 3) || throw(ArgumentError("labelshapes makes 2-D or 3-D images"))
    outer = ntuple(i -> i == 1 ? 0.9 : 0.75, N)               # semi-axes, as fractions of the half-size
    inner = ntuple(i -> i == 1 ? 0.6 : 0.45, N)
    lesion = ntuple(i -> i == 1 ? 0.3 : 0.0, N)               # centre of the small disc
    labels = zeros(Int, dims)
    for I in CartesianIndices(labels)
        x = ntuple(i -> (I[i] - (dims[i] + 1) / 2) / (dims[i] / 2), N)    # -1 to 1 along each axis
        inside(axes) = sum((x ./ axes) .^ 2) <= 1
        labels[I] = sum(abs2, x .- lesion) <= 0.12^2 ? 3 : inside(inner) ? 2 : inside(outer) ? 1 : 0
    end
    return labels
end
