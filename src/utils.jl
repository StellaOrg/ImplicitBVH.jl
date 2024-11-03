"""
    $(TYPEDEF)

Options for building and traversing bounding volume hierarchies, including parallel strategy
settings.

An exemplar of an index (e.g. `Int32(0)`) is used to deduce the types of indices used in the BVH
building ([`ImplicitTree`](@ref), order) and traversal ([`IndexPair`](@ref)).

The CPU scheduler can be `:threads` (for base Julia threads) or `:polyester` (for Polyester.jl
threads).

If `compute_extrema=false` and `mins` / `maxs` are defined, they will not be computed from the
distribution of bounding volumes; useful if you have a fixed simulation box, for example. You
**must** ensure that no bounding volume centers will touch or be outside these bounds, otherwise
logically incorrect results will be silently produced.

# Methods
    BVHOptions(;

        # Example index from which to deduce type
        index_exemplar::I               = Int32(0),

        # CPU threading
        scheduler::Symbol               = :threads,
        num_threads::Int                = Threads.nthreads(),
        min_mortons_per_thread::Int     = 1000,
        min_boundings_per_thread::Int   = 1000,
        min_traversals_per_thread::Int  = 1000,

        # GPU scheduling
        block_size::Int                 = 256,

        # Minima / maxima
        compute_extrema::Bool           = true,
        mins::NTuple{3, T}              = (NaN32, NaN32, NaN32),
        maxs::NTuple{3, T}              = (NaN32, NaN32, NaN32),
    ) where {I <: Integer, T}

# Fields
    $(TYPEDFIELDS)

"""
struct BVHOptions{I <: Integer, T}

    # Example index from which to deduce type
    index_exemplar::I

    # CPU threading
    scheduler::Symbol
    num_threads::Int
    min_mortons_per_thread::Int
    min_boundings_per_thread::Int
    min_traversals_per_thread::Int

    # GPU scheduling
    block_size::Int

    # Minima / maxima
    compute_extrema::Bool
    mins::NTuple{3, T}
    maxs::NTuple{3, T}
end


function BVHOptions(;

    # Example index from which to deduce type
    index_exemplar::I               = Int32(0),

    # CPU threading
    scheduler::Symbol               = :threads,
    num_threads::Int                = Threads.nthreads(),
    min_mortons_per_thread::Int     = 1000,
    min_boundings_per_thread::Int   = 1000,
    min_traversals_per_thread::Int  = 1000,

    # GPU scheduling
    block_size::Int                 = 256,

    # Minima / maxima
    compute_extrema::Bool           = true,
    mins::NTuple{3, T}              = (NaN32, NaN32, NaN32),
    maxs::NTuple{3, T}              = (NaN32, NaN32, NaN32),
) where {I <: Integer, T}

    # Correctness checks
    @argcheck num_threads > 0
    @argcheck min_mortons_per_thread > 0
    @argcheck min_boundings_per_thread > 0
    @argcheck min_traversals_per_thread > 0
    @argcheck block_size > 0

    # If we want to avoid computing extrema, make sure `mins` and `maxs` were defined
    if !compute_extrema
        @argcheck all(isfinite, mins)
        @argcheck all(isfinite, maxs)
    end

    BVHOptions(
        index_exemplar,
        scheduler,
        num_threads,
        min_mortons_per_thread,
        min_boundings_per_thread,
        min_traversals_per_thread,
        block_size,
        compute_extrema,
        mins,
        maxs,
    )
end



# Get pair of children indices of nodes in a perfect binary tree
_leftleft(implicit1::I, implicit2::I) where I = (implicit1 * I(2), implicit2 * I(2))
_leftright(implicit1::I, implicit2::I) where I = (implicit1 * I(2), implicit2 * I(2) + I(1))
_rightleft(implicit1::I, implicit2::I) where I = (implicit1 * I(2) + I(1), implicit2 * I(2))
_rightright(implicit1::I, implicit2::I) where I = (implicit1 * I(2) + I(1), implicit2 * I(2) + I(1))

_leftnoop(implicit1::I, implicit2::I) where I = (implicit1 * I(2), implicit2)
_rightnoop(implicit1::I, implicit2::I) where I = (implicit1 * I(2) + I(1), implicit2)
_noopleft(implicit1::I, implicit2::I) where I = (implicit1, implicit2 * I(2))
_noopright(implicit1::I, implicit2::I) where I = (implicit1, implicit2 * I(2) + I(1))




# Fast ilog2 adapted from https://github.com/jlapeyre/ILog2.jl - thank you!
# Included here directly to minimise dependencies and possible errors surface area
const IntBits = Union{Int8, Int16, Int32, Int64, Int128,
                      UInt8, UInt16, UInt32, UInt64, UInt128}

@generated function msbindex(::Type{T}) where {T <: Integer}
    sizeof(T) * 8 - 1
end

ilog2(x, ::typeof(RoundUp)) = ispow2(x) ? ilog2(x) : ilog2(x) + 1
ilog2(x, ::typeof(RoundDown)) = ilog2(x)

@inline function ilog2(n::T) where {T <: IntBits}
    @boundscheck n > zero(T) || throw(DomainError(n))
    unsafe_ilog2(n)
end

unsafe_ilog2(x, ::typeof(RoundUp)) = ispow2(x) ? unsafe_ilog2(x) : unsafe_ilog2(x) + 1
unsafe_ilog2(x, ::typeof(RoundDown)) = unsafe_ilog2(x)

@inline function unsafe_ilog2(n::T) where {T <: IntBits}
    msbindex(T) - leading_zeros(n)
end




# Specialised maths functions
pow2(n::I) where I <: Integer = one(I) << n


function dot3(x, y)
    x[1] * y[1] + x[2] * y[2] + x[3] * y[3]
end


function dist3sq(x, y)
    (x[1] - y[1]) * (x[1] - y[1]) +
    (x[2] - y[2]) * (x[2] - y[2]) +
    (x[3] - y[3]) * (x[3] - y[3])
end


dist3(x, y) = sqrt(dist3sq(x, y))

minimum2(a, b) = a < b ? a : b
minimum3(a, b, c) = a < b ? minimum2(a, c) : minimum2(b, c)

maximum2(a, b) = a > b ? a : b
maximum3(a, b, c) = a > b ? maximum2(a, c) : maximum2(b, c)
