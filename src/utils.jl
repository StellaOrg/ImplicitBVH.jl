"""
    $(TYPEDEF)

Options for building and traversing bounding volume hierarchies, including parallel strategy
settings.

An exemplar of an index (e.g. `Int32(0)`) is used to deduce the types of indices used in the BVH
building ([`ImplicitTree`](@ref), order) and traversal ([`IndexPair`](@ref)).

# Methods
    BVHOptions(;

        # Example index from which to deduce type
        index::Union{I, Type{I}}            = Int32(0),

        # Morton encoding algorithm
        morton::M                           = DefaultMortonAlgorithm(UInt32(0)),

        # CPU threading
        num_threads::Int                    = Threads.nthreads(),
        min_mortons_per_thread::Int         = 100,
        min_sorts_per_thread::Int           = 100,
        min_boundings_per_thread::Int       = 100,
        min_traversals_per_thread::Int      = 100,

        # GPU scheduling
        block_size::Int                     = 256,
    ) where {I <: Integer, M}

# Fields
    $(TYPEDFIELDS)

"""
struct BVHOptions{I <: Integer, M}

    # Example index from which to deduce type
    index_exemplar::I

    # Morton encoding algorithm
    morton::M

    # CPU threading
    num_threads::Int
    min_mortons_per_thread::Int
    min_sorts_per_thread::Int
    min_boundings_per_thread::Int
    min_traversals_per_thread::Int

    # GPU scheduling
    block_size::Int
end


function BVHOptions(;

    # Example index from which to deduce type
    index::Union{I, Type{I}}            = Int32(0),

    # Morton encoding algorithm
    morton::M                           = DefaultMortonAlgorithm(UInt32(0)),

    # CPU threading
    num_threads::Int                    = Threads.nthreads(),
    min_mortons_per_thread::Int         = 100,
    min_sorts_per_thread::Int           = 100,
    min_boundings_per_thread::Int       = 100,
    min_traversals_per_thread::Int      = 100,

    # GPU scheduling
    block_size::Int                     = 256,
) where {I <: Integer, M}

    # Correctness checks
    @argcheck num_threads > 0
    @argcheck min_mortons_per_thread > 0
    @argcheck min_sorts_per_thread > 0
    @argcheck min_boundings_per_thread > 0
    @argcheck min_traversals_per_thread > 0
    @argcheck block_size > 0

    index_exemplar = index isa Type ? zero(I) : index

    BVHOptions(
        index_exemplar,
        morton,
        num_threads,
        min_mortons_per_thread,
        min_sorts_per_thread,
        min_boundings_per_thread,
        min_traversals_per_thread,
        block_size,
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




# Simple MVector (like from StaticArrays but without the boundschecks which fail GPU compilation
# when run with --check-bounds=yes like in tests). Very unsafe, not made for general use.
mutable struct SimpleMVector{N, T}
    data::NTuple{N, T}
end

@inline function SimpleMVector{N, T}(::UndefInitializer) where {N, T}
    SimpleMVector{N, T}(getfield(Ref{NTuple{N, T}}(), 1))
end

@inline function Base.getindex(v::SimpleMVector{N, T}, i::Integer) where {N, T}
    GC.@preserve v unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(v)), i)
end

@inline function Base.setindex!(v::SimpleMVector{N, T}, val, i::Integer) where {N, T}
    GC.@preserve v unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(v)), convert(T, val), i)
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





#     _k2ij_inclusive(n, k) -> (i, j)
# 
# Unrank the 0-based inclusive upper-triangle index `k` into the pair `(i, j)` with
# `0 ≤ i ≤ j < n` in lexicographic block order:
# 
#     (0,0),(0,1),...,(0,n-1),
#     (1,1),(1,2),...,(1,n-1),
#     ...
#     (n-1,n-1)
# 
# Precondition: `0 ≤ k < n*(n+1) ÷ 2`.
# 
# Uses a Float32 approximation to get an initial block index `i` and then corrects
# it with a few integer steps to guarantee the correct block. All integer literals
# are lifted into type `I`.
@inline function _k2ij_inclusive(n::I, k::I) where I <: Integer

    # S_before(i) = number of pairs before block i = i*n - i*(i-1)/2
    S_before(t) = t * n - (t * (t - I(1))) ÷ I(2)

    # Float32 initial guess: solve i^2 - (2n+1)*i + 2k = 0, take floor of smaller root
    t = I(2) * n + I(1)                      # 2n+1
    discr_f = Float32(t)^2 - Float32(I(8)) * Float32(k)  # (2n+1)^2 - 8k
    i = unsafe_trunc(I, (Float32(t) - sqrt(discr_f)) / Float32(I(2)))
    if i < zero(I)
        i = zero(I)
    elseif i >= n
        i = n - I(1)
    end

    # Correct so that S_before(i) ≤ k < S_before(i+1)
    while i > zero(I) && S_before(i) > k
        i -= I(1)
    end
    while i + I(1) < n && S_before(i + I(1)) <= k
        i += I(1)
    end

    prev = S_before(i)
    offset = k - prev
    j = i + offset  # block i spans j = i..(n-1)

    return (i, j)
end


#     _k2ij_exclusive(n, k) -> (i, j)
# 
# Unrank the 0-based exclusive upper-triangle index `k` into the pair `(i, j)` with
# `0 ≤ i < j < n` in lexicographic block order:
# 
#     (0,1),(0,2),...,(0,n-1),
#     (1,2),(1,3),...,(n-2,n-1)
# 
# Precondition: `0 ≤ k < n*(n-1) ÷ 2`.
# 
# Uses a Float32 closed-form approximation for the block index `i` and then
# performs a small integer correction to guarantee correctness. Returned pair is
# 0-based.
@inline function _k2ij_exclusive(n::I, k::I) where I <: Integer

    # S_before(i) = number of pairs before block i = i*(2n - i - 1)/2
    S_before(t) = (t * (I(2) * n - t - I(1))) ÷ I(2)

    # Float32 initial guess for i: solve i*(2n - i -1)/2 ≤ k < (i+1)*(2n - (i+1) -1)/2
    # Quadratic: i^2 - (2n-1)*i + 2k = 0, take floor of smaller root
    t = I(2) * n - I(1)              # 2n -1
    discr_f = Float32(t)^2 - Float32(I(8)) * Float32(k)  # (2n-1)^2 - 8k
    i = unsafe_trunc(I, (Float32(t) - sqrt(discr_f)) / Float32(I(2)))
    if i < zero(I)
        i = zero(I)
    elseif i >= n - I(1)
        i = n - I(2)  # since i < n-1
    end

    # Correct so that S_before(i) ≤ k < S_before(i+1)
    while i > zero(I) && S_before(i) > k
        i -= I(1)
    end
    while i + I(1) < n - I(1) && S_before(i + I(1)) <= k
        i += I(1)
    end

    prev = S_before(i)
    offset = k - prev
    j = i + I(1) + offset  # block i spans j = i+1 .. n-1

    return (i, j)
end
