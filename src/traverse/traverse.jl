"""
    const IndexPair{I} = Tuple{I, I}

Alias for a tuple of two indices representing e.g. a contacting pair.
"""
const IndexPair{I} = Tuple{I, I}


"""
    $(TYPEDEF)

Collected BVH traversal `contacts` vector, some stats, plus the two buffers `cache1` and `cache2`
which can be reused for future traversals to minimise memory allocations.

# Fields
- `start_level1::Int`: the level at which the single/pair-tree traversal started for the first BVH.
- `start_level2::Int`: the level at which the pair-tree traversal started for the second BVH.
- `num_checks::Int`: the total number of contact checks done.
- `num_contacts::Int`: the number of contacts found.
- `contacts::view(cache1, 1:num_contacts)`: the contacting pairs found, as a view into `cache1`.
- `cache1::C1{IndexPair} <: AbstractVector`: first BVH traversal buffer.
- `cache2::C2{IndexPair} <: AbstractVector`: second BVH traversal buffer.
"""
struct BVHTraversal{C1 <: AbstractVector, C2 <: AbstractVector}
    # Stats
    start_level1::Int
    start_level2::Int
    num_checks::Int

    # Data
    num_contacts::Int
    cache1::C1
    cache2::C2
end


# Constructor in the case of single-tree traversal (e.g. traverse(bvh)), when we only have a
# single start_level
function BVHTraversal(
    start_level::Int,
    num_checks::Int,
    num_contacts::Int,
    cache1::AbstractVector,
    cache2::AbstractVector,
)
    BVHTraversal(start_level, start_level, num_checks, num_contacts, cache1, cache2)
end


# Custom pretty-printing
function Base.show(io::IO, t::BVHTraversal{C1, C2}) where {C1, C2}
    print(
        io,
        """
        BVHTraversal
          start_level1: $(typeof(t.start_level1)) $(t.start_level1)
          start_level2: $(typeof(t.start_level2)) $(t.start_level2)
          num_checks:   $(typeof(t.num_checks)) $(t.num_checks)
          num_contacts: $(typeof(t.num_contacts)) $(t.num_contacts)
          contacts:     $(Base.typename(typeof(t.contacts)).wrapper){IndexPair}($(size(t.contacts)))
          cache1:       $C1($(size(t.cache1)))
          cache2:       $C2($(size(t.cache2)))
        """
    )
end


function Base.getproperty(bt::BVHTraversal, sym::Symbol)
   if sym === :contacts
       return @view bt.cache1[1:bt.num_contacts]
   else
       return getfield(bt, sym)
   end
end

Base.propertynames(::BVHTraversal) = (:start_level1, :start_level2, :num_checks, :contacts,
                                      :num_contacts, :cache1, :cache2)


"""
    default_start_level(bvh::BVH)::Int
    default_start_level(num_leaves::Integer)::Int

Compute the default start level when traversing a single BVH tree.
"""
function default_start_level(bvh::BVH)::Int
    maximum2(bvh.tree.levels ÷ 2, bvh.built_level)
end


function default_start_level(num_leaves::Integer)::Int
    # Compute the default start level from the number of leaves (geometries) only
    @boundscheck if num_leaves < 1
        throw(DomainError(num_leaves, "must have at least one geometry!"))
    end

    levels = @inbounds ilog2(num_leaves, RoundUp) + 1   # number of binary tree levels
    maximum2(levels ÷ 2, 1)
end




# Single BVH and BVH-BVH traversal
include("traverse_single.jl")
include("traverse_pair.jl")
