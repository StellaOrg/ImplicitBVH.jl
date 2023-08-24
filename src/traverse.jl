"""
    $(TYPEDEF)

Alias for a tuple of two indices representing e.g. a contacting pair.
"""
const IndexPair = Tuple{Int, Int}


"""
    $(TYPEDEF)

Collected BVH traversal `contacts` vector, some stats, plus the two buffers `cache1` and `cache2`
which can be reused for future traversals to minimise memory allocations.

# Fields
- `start_level::Int`: the level at which the traversal started.
- `num_checks::Int`: the total number of contact checks done.
- `num_contacts::Int`: the number of contacts found.
- `contacts::view(cache1, 1:num_contacts)`: the contacting pairs found, as a view into `cache1`.
- `cache1::C1{IndexPair} <: AbstractVector`: first BVH traversal buffer.
- `cache2::C2{IndexPair} <: AbstractVector`: second BVH traversal buffer.
"""
struct BVHTraversal{C1 <: AbstractVector, C2 <: AbstractVector}
    # Stats
    start_level::Int
    num_checks::Int

    # Data
    num_contacts::Int
    cache1::C1
    cache2::C2
end


# Custom pretty-printing
function Base.show(io::IO, t::BVHTraversal{C1, C2}) where {C1, C2}
    print(
        io,
        """
        BVHTraversal
          start_level:  $(typeof(t.start_level)) $(t.start_level)
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

Base.propertynames(::BVHTraversal) = (:start_level, :num_checks, :contacts,
                                      :num_contacts, :cache1, :cache2)


# Single BVH and BVH-BVH traversal
include("traverse_single.jl")
include("traverse_pair.jl")
