"""
    $(TYPEDEF)

Acceptable unsigned integer types for Morton encoding: $(MortonUnsigned).
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
@inline function morton_encode_single(centre, mins, maxs, ::Type{U}=UInt) where {U <: MortonUnsigned}
    scaling = morton_scaling(U)
    m = zero(U)

    @inbounds for i in 1:3
        scaled = (centre[i] - mins[i]) / (maxs[i] - mins[i])    # Scaling number between (0, 1)
        index = U(floor(scaled * scaling))                      # Scaling to (0, morton_scaling)
        m += morton_split3(index) << (3 - i)                    # Shift into position - XYZXYZXYZ
    end

    m
end


function morton_encode_range!(
    mortons::AbstractVector{U},
    bounding_volumes,
    mins, maxs,
    irange,
) where {U <: MortonUnsigned}

    @inbounds for i in irange[1]:irange[2]
        bv_center = center(bounding_volumes[i])
        mortons[i] = morton_encode_single(bv_center, mins, maxs, U)
    end

    nothing
end


"""
    bounding_volumes_extrema(bounding_volumes)

Compute exclusive lower and upper bounds in iterable of bounding volumes, e.g. Vector{BBox}.
"""
function bounding_volumes_extrema(bounding_volumes)

    xmin, ymin, zmin = center(bounding_volumes[1])
    xmax, ymax, zmax = xmin, ymin, zmin

    @inbounds for i in 2:length(bounding_volumes)

        xc, yc, zc = center(bounding_volumes[i])

        xc < xmin && (xmin = xc)
        yc < ymin && (ymin = yc)
        zc < zmin && (zmin = zc)

        xc > xmax && (xmax = xc)
        yc > ymax && (ymax = yc)
        zc > zmax && (zmax = zc)
    end

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
    morton_encode!(mortons::AbstractVector{U}, bounding_volumes) where {U <: MortonUnsigned}
    morton_encode!(mortons::AbstractVector{U}, bounding_volumes, mins, maxs)

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
) where {U <: MortonUnsigned}

    # Bounds checking
    @assert firstindex(mortons) == firstindex(bounding_volumes) == 1
    @assert length(mortons) == length(bounding_volumes)
    @assert length(mins) == length(maxs) == 3

    # Trivial case
    length(bounding_volumes) == 0 && return nothing

    # Encode bounding volumes' centres across multiple threads using contiguous ranges
    tp = TaskPartitioner(length(bounding_volumes), Threads.nthreads(), 1000)
    if tp.num_tasks == 1
        morton_encode_range!(
            mortons, bounding_volumes,
            mins, maxs,
            (firstindex(bounding_volumes), lastindex(bounding_volumes)),
        )
    else
        tasks = Vector{Task}(undef, tp.num_tasks)
        @inbounds for i in 1:tp.num_tasks
            tasks[i] = Threads.@spawn morton_encode_range!(
                mortons, bounding_volumes,
                mins, maxs,
                tp[i],
            )
        end
        @inbounds for i in 1:tp.num_tasks
            wait(tasks[i])
        end
    end

    nothing
end


function morton_encode!(mortons::AbstractVector{U}, bounding_volumes) where {U <: MortonUnsigned}
    # Compute exclusive bounds [xmin, ymin, zmin], [xmax, ymax, zmax].
    # TODO: see uint_encode from Base.Sort, scaling might be better
    mins, maxs = bounding_volumes_extrema(bounding_volumes)
    morton_encode!(mortons, bounding_volumes, mins, maxs)
    nothing
end


"""
    morton_encode(bounding_volumes, ::Type{U}=UInt) where {U <: MortonUnsigned}

Encode the centers of some `bounding_volumes` as Morton codes of type `U <: `
[`MortonUnsigned`](@ref). See [`morton_encode!`](@ref) for full details. 
"""
function morton_encode(bounding_volumes, ::Type{U}=UInt) where {U <: MortonUnsigned}
    # Pre-allocate vector of morton codes
    mortons = similar(bounding_volumes, U)
    morton_encode!(mortons, bounding_volumes)
    mortons
end
