"""
    $(TYPEDEF)

Acceptable unsigned integer types for Morton encoding.
"""
const MortonUnsigned = Union{UInt16, UInt32, UInt64}

"""
    $(TYPEDEF)

Type values of acceptable unsigned integer types for Morton encoding.
"""
const MortonUnsignedType = Union{Type{UInt16}, Type{UInt32}, Type{UInt64}}


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
    morton_encode_single(centre, mins, maxs, U::MortonUnsignedType=UInt32)

Return Morton code for a single 3D position `centre` scaled uniformly between `mins` and `maxs`.
Works transparently for SVector, Vector, etc. with eltype UInt16, UInt32 or UInt64.
"""
@inline function morton_encode_single(centre, mins, maxs, U::MortonUnsignedType=UInt32)
    scaling = morton_scaling(U)
    m = zero(U)

    for i in 1:3
        scaled = (centre[i] - mins[i]) / (maxs[i] - mins[i])    # Scaling number between (0, 1)
        index = U(floor(scaled * scaling))                      # Scaling to (0, morton_scaling)
        m += morton_split3(index) << (3 - i)                    # Shift into position - XYZXYZXYZ
    end

    m
end


"""
    bounding_volumes_extrema(bounding_volumes)

Compute exclusive lower and upper bounds in iterable of bounding volumes, e.g. Vector{BBox}.
"""
function bounding_volumes_extrema(bounding_volumes)
    mins = center(bounding_volumes[1]) |> MVector{3}
    maxs = center(bounding_volumes[1]) |> MVector{3}

    for i in 2:length(bounding_volumes)
        for j in 1:3
            if center(bounding_volumes[i])[j] < mins[j]
                mins[j] = center(bounding_volumes[i])[j]
            end

            if center(bounding_volumes[i])[j] > maxs[j]
                maxs[j] = center(bounding_volumes[i])[j]
            end
        end
    end

    # Expand extrema by float precision to ensure morton codes < 2^21
    T = eltype(mins)
    return (
        SVector{3}(mins - T(1e-5) * abs.(mins) .- floatmin(T)),
        SVector{3}(maxs + T(1e-5) * abs.(maxs) .+ floatmin(T)),
    )
end


"""
    morton_encode(bounding_volumes, U::MortonUnsignedType=UInt32)
    morton_encode!(mortons, bounding_volumes, U)
    morton_encode!(mortons, bounding_volumes, mins, maxs, U)

Encode each bounding volume into Vector{U} of corresponding Morton codes such that they uniformly
cover the maximum Morton range given an unsigned integer type U.

If not provided, allocate Vector{U} for Morton codes. Likewise, compute dimension-wise exclusive
minima and maxima if not provided.
"""
function morton_encode!(mortons, bounding_volumes, mins, maxs, U::MortonUnsignedType=UInt32)
    # Bounds checking and trivial case
    @boundscheck length(mortons) >= length(bounding_volumes) || throw(BoundsError(mortons))
    length(bounding_volumes) == 0 && return

    # Allow collections that don't start at 1
    mindices = eachindex(mortons)
    bindices = eachindex(bounding_volumes)

    # Encode each bounding volume provided they have a `center` function.
    Threads.@threads for i in 1:length(bounding_volumes)
        bv_center = center(bounding_volumes[bindices[i]])
        mortons[mindices[i]] = morton_encode_single(bv_center, mins, maxs, U)
    end
end


function morton_encode!(mortons, bounding_volumes, U::MortonUnsignedType=UInt32)
    # Compute exclusive bounds [xmin, ymin, zmin], [xmax, ymax, zmax].
    mins, maxs = bounding_volumes_extrema(bounding_volumes)
    morton_encode!(mortons, bounding_volumes, mins, maxs, U)
end


function morton_encode(bounding_volumes, U::MortonUnsignedType=UInt32)
    # Pre-allocate vector of morton codes
    mortons = Vector{U}(undef, length(bounding_volumes))
    morton_encode!(mortons, bounding_volumes, U)
    mortons
end
