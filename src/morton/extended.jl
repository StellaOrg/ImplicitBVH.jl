"""
    $(TYPEDEF)

Extended Morton encoding algorithm for improved spatial locality during BVH construction.
Implements the extensions proposed by Vinkler et al. [1]:

- Adaptive axis ordering driven by the scene extents.
- Variable bit counts per dimension determined by repeated longest-axis splits.
- Optional size bits that capture primitive extent to reduce BVH overlap.

The Morton code type is specified by the `exemplar` parameter; supported types are
`UInt16`, `UInt32` and `UInt64`.

[1] Vinkler M, Bittner J, Havran V. Extended Morton codes for high performance bounding volume
    hierarchy construction. Proceedings of High Performance Graphics (HPG) 2017.

# Examples
Use 32-bit Extended Morton encoding with default scheduling:
```julia
using ImplicitBVH
options = BVHOptions(morton=ExtendedMortonAlgorithm(UInt32(0)))
```

Use 64-bit encoding but disable size bits (e.g. for compact diagonals):
```julia
options = BVHOptions(morton=ExtendedMortonAlgorithm(UInt64(0); size_budget=0))
```
"""
struct ExtendedMortonAlgorithm{M <: Union{UInt16, UInt32, UInt64}, T} <: MortonAlgorithm
    exemplar::M
    compute_extrema::Bool
    mins::NTuple{3, T}
    maxs::NTuple{3, T}
    size_interval::Int
    size_budget::Int
    use_sqrt_size::Bool
end

Base.eltype(::ExtendedMortonAlgorithm{M}) where M = M

function ExtendedMortonAlgorithm(
    exemplar;
    compute_extrema::Bool=true,
    mins::NTuple{3, T}=(NaN32, NaN32, NaN32),
    maxs::NTuple{3, T}=(NaN32, NaN32, NaN32),
    size_interval::Union{Nothing, Int}=nothing,
    size_budget::Union{Nothing, Int}=nothing,
    sqrt_size::Union{Nothing, Bool}=nothing,
) where T
    M = typeof(exemplar)
    interval = something(size_interval, _default_size_interval(M))
    budget = something(size_budget, _default_size_budget(M))
    sqrt_flag = something(sqrt_size, interval >= 7)
    ExtendedMortonAlgorithm(exemplar, compute_extrema, mins, maxs, interval, budget, sqrt_flag)
end


function morton_encode!(bounding_volumes, alg::ExtendedMortonAlgorithm, options=BVHOptions())

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


struct ExtendedMortonParams{N, R}
    mins::NTuple{3, R}
    scales::NTuple{3, R}
    schedule::NTuple{N, UInt8}
    bits_per_axis::NTuple{4, UInt8}
    max_values::NTuple{4, UInt64}
    size_scale::R
    sqrt_size::Bool
end


@inline _default_size_interval(::Type{UInt16}) = 0
@inline _default_size_interval(::Type{UInt32}) = 7
@inline _default_size_interval(::Type{UInt64}) = 7

@inline _default_size_budget(::Type{UInt16}) = 0
@inline _default_size_budget(::Type{UInt32}) = 4
@inline _default_size_budget(::Type{UInt64}) = 6

@inline function _clamp_size_budget(n_bits::Int, interval::Int, budget::Int)
    if interval <= 0 || budget <= 0
        return 0
    end
    max_possible = n_bits ÷ interval
    max_possible <= 0 ? 0 : min(budget, max_possible)
end


function ExtendedMortonAlgorithm(
    exemplar,
    compute_extrema::Bool,
    mins::NTuple{3, T},
    maxs::NTuple{3, T},
    size_interval::Int=_default_size_interval(typeof(exemplar)),
    size_budget::Int=_default_size_budget(typeof(exemplar)),
    use_sqrt_size::Bool=size_interval >= 7,
) where T
    size_interval < 0 && throw(ArgumentError("size_interval must be non-negative"))
    size_budget < 0 && throw(ArgumentError("size_budget must be non-negative"))
    adjusted_budget = size_interval == 0 ? 0 : size_budget
    adjusted_sqrt = adjusted_budget > 0 ? use_sqrt_size : false
    ExtendedMortonAlgorithm{typeof(exemplar), T}(
        exemplar,
        compute_extrema,
        mins,
        maxs,
        size_interval,
        adjusted_budget,
        adjusted_sqrt,
    )
end


@inline function _volume_diagonal(b::BBox{T}) where T
    dx = b.up[1] - b.lo[1]
    dy = b.up[2] - b.lo[2]
    dz = b.up[3] - b.lo[3]
    sqrt(dx * dx + dy * dy + dz * dz)
end

@inline _volume_diagonal(s::BSphere{T}) where T = T(2) * s.r


@inline function _max_value_for_bits(bits::UInt8)
    b = Int(bits)
    if b <= 0
        return UInt64(0)
    elseif b >= 64
        return typemax(UInt64)
    else
        return (UInt64(1) << b) - UInt64(1)
    end
end


@inline function _scalar_axis_scale(range::R, max_value::UInt64, bits::UInt8, epsval::R) where R <: AbstractFloat
    if bits == 0 || max_value == 0 || !isfinite(range) || !(range > epsval)
        return zero(R)
    end
    R(max_value) / range
end

@inline function _axis_scales(ranges::NTuple{3, R}, bits::NTuple{4, UInt8}, max_values::NTuple{4, UInt64}) where R <: AbstractFloat
    epsval = eps(one(R))
    (
        _scalar_axis_scale(ranges[1], max_values[1], bits[1], epsval),
        _scalar_axis_scale(ranges[2], max_values[2], bits[2], epsval),
        _scalar_axis_scale(ranges[3], max_values[3], bits[3], epsval),
    )
end


@inline function _compute_size_scale(ranges::NTuple{3, R}, bits::UInt8, max_value::UInt64, sqrt_mode::Bool) where R <: AbstractFloat
    if bits == 0 || max_value == 0
        return zero(R)
    end
    (isfinite(ranges[1]) && isfinite(ranges[2]) && isfinite(ranges[3])) || return zero(R)
    diag_sq = ranges[1] * ranges[1] + ranges[2] * ranges[2] + ranges[3] * ranges[3]
    if !(diag_sq > zero(R)) || !isfinite(diag_sq)
        return zero(R)
    end
    diag = sqrt(diag_sq)
    denom = sqrt_mode ? sqrt(diag) : diag
    if !(denom > eps(one(R))) || !isfinite(denom)
        return zero(R)
    end
    R(max_value) / denom
end


@inline function _select_axis(lengths::NTuple{3, R}, fallback::Int) where R <: AbstractFloat
    l1, l2, l3 = lengths
    axis = 1
    max_val = l1
    if l2 > max_val
        axis = 2
        max_val = l2
    end
    if l3 > max_val
        axis = 3
        max_val = l3
    end
    if !isfinite(max_val) || max_val <= zero(R)
        return fallback
    end
    axis
end


@inline function _halve_axis(lengths::NTuple{3, R}, axis::Int, half::R) where R <: AbstractFloat
    if axis == 1
        new_val = lengths[1] * half
        (!isfinite(new_val) || new_val < zero(R)) && (new_val = zero(R))
        return (new_val, lengths[2], lengths[3])
    elseif axis == 2
        new_val = lengths[2] * half
        (!isfinite(new_val) || new_val < zero(R)) && (new_val = zero(R))
        return (lengths[1], new_val, lengths[3])
    else
        new_val = lengths[3] * half
        (!isfinite(new_val) || new_val < zero(R)) && (new_val = zero(R))
        return (lengths[1], lengths[2], new_val)
    end
end


function _build_extended_schedule(extents::NTuple{3, R}, n_bits::Int, interval::Int, budget::Int) where R <: AbstractFloat
    schedule = Vector{UInt8}(undef, n_bits)
    lengths = (abs(extents[1]), abs(extents[2]), abs(extents[3]))
    size_bits_used = 0
    half = R(0.5)
    for idx in 1:n_bits
        fallback = ((idx - 1) % 3) + 1
        if interval > 0 && size_bits_used < budget && idx % interval == 0
            schedule[idx] = UInt8(4)
            size_bits_used += 1
        else
            axis = _select_axis(lengths, fallback)
            schedule[idx] = UInt8(axis)
            lengths = _halve_axis(lengths, axis, half)
        end
    end
    Tuple(schedule)
end


function _count_axis_bits(schedule::NTuple{N, UInt8}) where N
    c1 = UInt8(0)
    c2 = UInt8(0)
    c3 = UInt8(0)
    c4 = UInt8(0)
    @inbounds for idx in 1:N
        axis = schedule[idx]
        if axis == UInt8(1)
            c1 += UInt8(1)
        elseif axis == UInt8(2)
            c2 += UInt8(1)
        elseif axis == UInt8(3)
            c3 += UInt8(1)
        elseif axis == UInt8(4)
            c4 += UInt8(1)
        end
    end
    (c1, c2, c3, c4)
end


function _make_extended_params(mins::NTuple{3, R}, ranges::NTuple{3, R}, schedule::NTuple{N, UInt8}, use_sqrt_size::Bool) where {N, R <: AbstractFloat}
    bits_per_axis = _count_axis_bits(schedule)
    max_values = (
        _max_value_for_bits(bits_per_axis[1]),
        _max_value_for_bits(bits_per_axis[2]),
        _max_value_for_bits(bits_per_axis[3]),
        _max_value_for_bits(bits_per_axis[4]),
    )
    scales = _axis_scales(ranges, bits_per_axis, max_values)
    sqrt_mode = use_sqrt_size && max_values[4] != 0
    size_scale = _compute_size_scale(ranges, bits_per_axis[4], max_values[4], sqrt_mode)
    ExtendedMortonParams(mins, scales, schedule, bits_per_axis, max_values, size_scale, sqrt_mode)
end


@inline function _quantize_axis(value::R, min_value::R, scale::R, max_value::UInt64) where R <: AbstractFloat
    if max_value == 0 || scale <= zero(R) || !isfinite(scale)
        return UInt64(0)
    end
    encoded = (value - min_value) * scale
    if !(encoded >= zero(R)) || isnan(encoded)
        return UInt64(0)
    end
    max_r = R(max_value)
    if encoded >= max_r
        return max_value
    elseif !isfinite(encoded)
        return UInt64(0)
    end
    unsafe_trunc(UInt64, encoded)
end


@inline function _quantize_size(diagonal::R, scale::R, max_value::UInt64, sqrt_mode::Bool) where R <: AbstractFloat
    if max_value == 0 || scale <= zero(R) || !isfinite(scale)
        return UInt64(0)
    end
    size_measure = sqrt_mode ? sqrt(max(diagonal, zero(R))) : max(diagonal, zero(R))
    (!isfinite(size_measure)) && (size_measure = zero(R))
    encoded = size_measure * scale
    if !(encoded >= zero(R)) || isnan(encoded)
        return UInt64(0)
    end
    max_r = R(max_value)
    if encoded >= max_r
        return max_value
    elseif !isfinite(encoded)
        return UInt64(0)
    end
    unsafe_trunc(UInt64, encoded)
end


@inline function _assemble_code(quantised::NTuple{4, UInt64}, schedule::NTuple{N, UInt8}, bits::NTuple{4, UInt8}, ::Type{U}) where {N, U <: Unsigned}
    c1 = Int(bits[1])
    c2 = Int(bits[2])
    c3 = Int(bits[3])
    c4 = Int(bits[4])
    code = zero(U)
    @inbounds for idx in 1:N
        axis = schedule[idx]
        bitpos = N - idx
        if axis == UInt8(1)
            c1 -= 1
            bit = (quantised[1] >> c1) & UInt64(0x1)
        elseif axis == UInt8(2)
            c2 -= 1
            bit = (quantised[2] >> c2) & UInt64(0x1)
        elseif axis == UInt8(3)
            c3 -= 1
            bit = (quantised[3] >> c3) & UInt64(0x1)
        else
            c4 -= 1
            bit = (quantised[4] >> c4) & UInt64(0x1)
        end
        code |= U(bit) << bitpos
    end
    code
end


@inline function _extended_morton_encode_single(centre, diagonal, params::ExtendedMortonParams{N, R}, ::Type{U}) where {N, R <: AbstractFloat, U <: Unsigned}
    c1 = _quantize_axis(R(centre[1]), params.mins[1], params.scales[1], params.max_values[1])
    c2 = _quantize_axis(R(centre[2]), params.mins[2], params.scales[2], params.max_values[2])
    c3 = _quantize_axis(R(centre[3]), params.mins[3], params.scales[3], params.max_values[3])
    diag_r = R(diagonal)
    c4 = _quantize_size(diag_r, params.size_scale, params.max_values[4], params.sqrt_size)
    _assemble_code((c1, c2, c3, c4), params.schedule, params.bits_per_axis, U)
end


function _morton_encode!(bounding_volumes, alg::ExtendedMortonAlgorithm, mins, maxs, options)
    U = eltype(alg)
    n_bits = sizeof(U) * 8
    size_budget = _clamp_size_budget(n_bits, alg.size_interval, alg.size_budget)
    use_sqrt = size_budget > 0 && alg.use_sqrt_size

    R = promote_type(typeof(mins[1]), typeof(maxs[1]), Float64)
    mins_r = ntuple(i -> R(mins[i]), 3)
    ranges_r = ntuple(i -> R(abs(maxs[i] - mins[i])), 3)

    schedule = _build_extended_schedule(ranges_r, n_bits, alg.size_interval, size_budget)
    params = _make_extended_params(mins_r, ranges_r, schedule, use_sqrt)
    size_enabled = params.max_values[4] != 0

    AK.foreachindex(
        bounding_volumes,
        block_size=options.block_size,
        max_tasks=options.num_threads,
        min_elems=options.min_mortons_per_thread,
    ) do i
        @inbounds bv = bounding_volumes[i]
        centre = center(bv.volume)
        diag_measure = size_enabled ? _volume_diagonal(bv.volume) : zero(typeof(centre[1]))
        morton = _extended_morton_encode_single(centre, diag_measure, params, U)
        @inbounds bounding_volumes[i] = BoundingVolume(bv.volume, bv.index, morton)
    end

    bounding_volumes
end
