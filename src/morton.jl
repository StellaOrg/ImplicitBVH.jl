"""
    $(TYPEDEF)

Supported unsigned integer types for Morton encoding: $(MortonUnsigned).
"""
const MortonUnsigned = Union{UInt16, UInt32, UInt64}


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


"""
    morton_encode_single(centre, mins, maxs, U::MortonUnsignedType=UInt32)

Return Morton code for a single 3D position `centre` scaled uniformly between `mins` and `maxs`.
Works transparently for SVector, Vector, etc. with eltype UInt16, UInt32 or UInt64.
"""
@inline function morton_encode_single(centre, mins, maxs, ::Type{U}=UInt32) where {U <: MortonUnsigned}
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
    m = (morton_split3(index1) << 2) + (morton_split3(index2) << 1) + morton_split3(index3)
    m
end


function _compute_extrema(bounding_volumes::AbstractVector, options)

    function min_centers(a, b)
        # a and b are NTuple{3, Float}
        (
            a[1] < b[1] ? a[1] : b[1],
            a[2] < b[2] ? a[2] : b[2],
            a[3] < b[3] ? a[3] : b[3],
        )
    end

    function max_centers(a, b)
        # a and b are NTuple{3, Float}
        (
            a[1] > b[1] ? a[1] : b[1],
            a[2] > b[2] ? a[2] : b[2],
            a[3] > b[3] ? a[3] : b[3],
        )
    end

    # Get numeric type of the inner bounding volume
    T = eltype(eltype(bounding_volumes))

    xyzmin = AK.mapreduce(
        center,             # Take the centre of each bounding volume
        min_centers,        # Reduce to the 3D minimum
        bounding_volumes,
        init=(floatmax(T), floatmax(T), floatmax(T)),
        neutral=(floatmax(T), floatmax(T), floatmax(T)),
        max_tasks=options.num_threads,
        min_elems=options.min_mortons_per_thread,
        block_size=options.block_size,
    )

    xyzmax = AK.mapreduce(
        center,
        max_centers,
        bounding_volumes,
        init=(floatmin(T), floatmin(T), floatmin(T)),
        neutral=(floatmin(T), floatmin(T), floatmin(T)),
        max_tasks=options.num_threads,
        min_elems=options.min_mortons_per_thread,
        block_size=options.block_size,
    )

    xyzmin, xyzmax
end


"""
    bounding_volumes_extrema(bounding_volumes)

Compute exclusive lower and upper bounds in iterable of bounding volumes, e.g. Vector{BBox}.
"""
function bounding_volumes_extrema(bounding_volumes::AbstractVector, options=BVHOptions())

    # Compute exact extrema
    (xmin, ymin, zmin), (xmax, ymax, zmax) = _compute_extrema(bounding_volumes, options)

    # Expand extrema by float precision to ensure morton codes are exclusively bounded by them
    T = typeof(xmin)

    xmin = xmin - relative_precision(T) * abs(xmin) - floatmin(T)
    ymin = ymin - relative_precision(T) * abs(ymin) - floatmin(T)
    zmin = zmin - relative_precision(T) * abs(zmin) - floatmin(T)

    xmax = xmax + relative_precision(T) * abs(xmax) + floatmin(T)
    ymax = ymax + relative_precision(T) * abs(ymax) + floatmin(T)
    zmax = zmax + relative_precision(T) * abs(zmax) + floatmin(T)

    (xmin, ymin, zmin), (xmax, ymax, zmax)
end


"""
    morton_encode!(
        mortons::AbstractVector{U},
        bounding_volumes,
        options=BVHOptions(),
    ) where {U <: MortonUnsigned}

    morton_encode!(
        mortons::AbstractVector{U},
        bounding_volumes::AbstractVector,
        mins,
        maxs,
        options=BVHOptions(),
    ) where {U <: MortonUnsigned}

Encode each bounding volume into vector of corresponding Morton codes such that they uniformly
cover the maximum Morton range given an unsigned integer type `U <: ` [`MortonUnsigned`](@ref).

!!! warning
    The dimension-wise exclusive `mins` and `maxs` *must* be correct; if any bounding volume center
    is equal to, or beyond `mins` / `maxs`, the results will be silently incorrect.
"""
function morton_encode!(
    mortons::AbstractVector{U},
    bounding_volumes::AbstractVector,
    mins,
    maxs,
    options=BVHOptions(),
) where {U <: MortonUnsigned}

    # Bounds checking
    @argcheck firstindex(mortons) == firstindex(bounding_volumes) == 1
    @argcheck length(mortons) == length(bounding_volumes)
    @argcheck length(mins) == length(maxs) == 3

    # Trivial case
    length(bounding_volumes) == 0 && return nothing

    # Parallelise on CPU / GPU
    AK.foreachindex(
        mortons,
        block_size=options.block_size,
        max_tasks=options.num_threads,
        min_elems=options.min_mortons_per_thread,
    ) do i
        @inbounds bv_center = center(bounding_volumes[i])
        @inbounds mortons[i] = morton_encode_single(bv_center, mins, maxs, U)
    end

    nothing
end


function morton_encode!(
    mortons::AbstractVector{U},
    bounding_volumes,
    options=BVHOptions(),
) where {U <: MortonUnsigned}

    # Compute exclusive bounds [xmin, ymin, zmin], [xmax, ymax, zmax].
    if options.compute_extrema
        mins, maxs = bounding_volumes_extrema(bounding_volumes, options)
    else
        mins = options.mins
        maxs = options.maxs
    end
    morton_encode!(mortons, bounding_volumes, mins, maxs, options)
    nothing
end


"""
    morton_encode(
        bounding_volumes,
        ::Type{U}=UInt,
        options=BVHOptions(),
    ) where {U <: MortonUnsigned}

Encode the centers of some `bounding_volumes` as Morton codes of type `U <: `
[`MortonUnsigned`](@ref). See [`morton_encode!`](@ref) for full details. 
"""
function morton_encode(
    bounding_volumes,
    ::Type{U}=UInt32,
    options=BVHOptions(),
) where {U <: MortonUnsigned}

    # Pre-allocate vector of morton codes
    mortons = similar(bounding_volumes, U)
    morton_encode!(mortons, bounding_volumes, options)
    mortons
end
