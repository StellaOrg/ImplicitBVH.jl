"""
    $(TYPEDEF)

Canonical Morton encoding algorithm using bit interleaving.

The Morton code type is specified by the `exemplar` parameter; supported types are
`UInt16`, `UInt32` and `UInt64`.

If `compute_extrema=false` and `mins` / `maxs` are defined, they will not be computed from the
distribution of bounding volumes; useful if you have a fixed simulation box, for example. You
**must** ensure that no bounding volume centers will touch or be outside these bounds, otherwise
logically incorrect results will be silently produced.

# Examples
Use 32-bit Morton encoding:
```julia
using ImplicitBVH
options = BVHOptions(morton=DefaultMortonAlgorithm(UInt32))
```
"""
struct DefaultMortonAlgorithm{M <: Union{UInt16, UInt32, UInt64}, T} <: MortonAlgorithm
    exemplar::M
    compute_extrema::Bool
    mins::NTuple{3, T}
    maxs::NTuple{3, T}
end

Base.eltype(::DefaultMortonAlgorithm{M}) where M = M

function DefaultMortonAlgorithm(
    exemplar::Union{M, Type{M}},;
    compute_extrema::Bool=true,
    mins::NTuple{3, T}=(NaN32, NaN32, NaN32),
    maxs::NTuple{3, T}=(NaN32, NaN32, NaN32),
) where {T, M <: Union{UInt16, UInt32, UInt64}}
    if exemplar isa Type
        exemplar = zero(M)
    end
    DefaultMortonAlgorithm(exemplar, compute_extrema, mins, maxs)
end


function morton_encode!(
    bounding_volumes::AbstractVector{<:BoundingVolume},
    alg::DefaultMortonAlgorithm,
    options::BVHOptions,
)
    # Trivial case
    length(bounding_volumes) == 0 && return bounding_volumes

    # Compute exclusive bounds [xmin, ymin, zmin], [xmax, ymax, zmax].
    if alg.compute_extrema
        mins, maxs = bounding_volumes_extrema(bounding_volumes, options)
    else
        mins = options.mins
        maxs = options.maxs
    end

    _morton_encode!(bounding_volumes, alg, mins, maxs, options)
end


function _morton_encode!(bounding_volumes, alg::DefaultMortonAlgorithm, mins, maxs, options)

    # Parallelise on CPU / GPU
    AK.foreachindex(
        bounding_volumes,
        block_size=options.block_size,
        max_tasks=options.num_threads,
        min_elems=options.min_mortons_per_thread,
    ) do i
        @inbounds bv_center = center(bounding_volumes[i].volume)
        @inbounds morton = morton_encode_single(bv_center, mins, maxs, alg)
        @inbounds bounding_volumes[i] = BoundingVolume(
            bounding_volumes[i].volume,
            bounding_volumes[i].index,
            morton,                         # Update morton code
        )
    end

    bounding_volumes
end


"""
    morton_encode_single(centre, mins, maxs, U::MortonUnsignedType=UInt32)

Return Morton code for a single 3D position `centre` scaled uniformly between `mins` and `maxs`.
Works transparently for SVector, Vector, etc. with eltype UInt16, UInt32 or UInt64.
"""
@inline function morton_encode_single(centre, mins, maxs, alg::DefaultMortonAlgorithm)
    U = eltype(alg)
    scaling = morton_scaling(U)

    # Scaling number between (0, 1)
    scaled1 = (centre[1] - mins[1]) / (maxs[1] - mins[1])
    scaled2 = (centre[2] - mins[2]) / (maxs[2] - mins[2])
    scaled3 = (centre[3] - mins[3]) / (maxs[3] - mins[3])

    # Scaling to (0, morton_scaling)
    index1 = unsafe_trunc(U, scaled1 * scaling)
    index2 = unsafe_trunc(U, scaled2 * scaling)
    index3 = unsafe_trunc(U, scaled3 * scaling)

    # Shift into position - XYZXYZXYZ
    m = (morton_split3(index1) << 2) | (morton_split3(index2) << 1) | morton_split3(index3)
    m
end


"""
    morton_split3(v::UInt16)
    morton_split3(v::UInt32)
    morton_split3(v::UInt64)

Shift a number's individual bits such that they have two zeros between them.
"""
@inline function morton_split3(v::UInt16)
    # Extract first 5 bits
    s = v & 0x001f

    s = (s | s << 8) & 0x100f
    s = (s | s << 4) & 0x10c3
    s = (s | s << 2) & 0x1249

    s
end


@inline function morton_split3(v::UInt32)
    # Extract first 10 bits
    s = v & 0x0000_03ff

    # Following StackOverflow discussion: https://stackoverflow.com/questions/18529057/
    # produce-interleaving-bit-patterns-morton-keys-for-32-bit-64-bit-and-128bit

    s = (s | s << 16) & 0x30000ff
    s = (s | s << 8) & 0x0300f00f
    s = (s | s << 4) & 0x30c30c3
    s = (s | s << 2) & 0x9249249

    s
end


@inline function morton_split3(v::UInt64)
    # Extract first 21 bits
    s = v & 0x0_001f_ffff

    s = (s | s << 32) & 0x1f00000000ffff
    s = (s | s << 16) & 0x1f0000ff0000ff
    s = (s | s << 8) & 0x100f00f00f00f00f
    s = (s | s << 4) & 0x10c30c30c30c30c3
    s = (s | s << 2) & 0x1249249249249249

    s
end


"""
    morton_scaling(::Type{UInt16}) = 2^5
    morton_scaling(::Type{UInt32}) = 2^10
    morton_scaling(::Type{UInt64}) = 2^21

Exclusive maximum number possible to use for 3D Morton encoding for each type.
"""
morton_scaling(::Type{UInt16}) = 2^5
morton_scaling(::Type{UInt32}) = 2^10
morton_scaling(::Type{UInt64}) = 2^21


"""
    relative_precision(::Type{Float16}) = 1e-2
    relative_precision(::Type{Float32}) = 1e-5
    relative_precision(::Type{Float64}) = 1e-14

Relative precision value for floating-point types.
"""
relative_precision(::Type{Float16}) = Float16(1e-2)
relative_precision(::Type{Float32}) = Float32(1e-5)
relative_precision(::Type{Float64}) = Float64(1e-14)
