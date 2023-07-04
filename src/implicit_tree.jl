"""
    $(TYPEDEF)

Implicit binary tree for `num_leaves` elements, where nodes are labelled according to a
breadth-first search.

# Methods
    ImplicitTree(num_leaves::Integer)
    ImplicitTree{T}(num_leaves::Integer)

# Fields
    $(TYPEDFIELDS)

# Examples

```julia
julia> using ImplicitBVH

# Given 5 geometric elements (e.g. bounding boxes) we construct the following implicit tree
# having the 5 real leaves at implicit indices 8-12 plus 3 virtual leaves.
#         Nodes & Leaves                Tree Level
#               1                       1
#       2               3               2
#   4       5       6        7v         3
# 8   9   10 11   12 13v  14v  15v      4
julia> tree = ImplicitTree(5)
ImplicitTree{Int64}
  levels: Int64 4
  real_leaves: Int64 5
  real_nodes: Int64 11
  virtual_leaves: Int64 3
  virtual_nodes: Int64 4

# We can keep all tree nodes in a contiguous vector with no extra padding for the virtual
# nodes by computing the real memory index of real nodes; e.g. real memory index of node 8
# skips node 7 which is virtual:
julia> memory_index(tree, 8)
7

# We can get the range of indices of real nodes on a given level
julia> level_indices(tree, 3)
(4, 6)

# And we can check if a node at a given implicit index is virtual
julia> isvirtual(tree, 6)
false

julia> isvirtual(tree, 7)
true
```
"""
struct ImplicitTree{T <: Integer}
    "Number of levels in the tree."
    levels::T

    "Number of real leaves - i.e. the elements from which the tree was constructed."
    real_leaves::T

    "Total number of real nodes in tree."
    real_nodes::T

    "Number of virtual leaves needed at the bottom level to have a perfect binary tree."
    virtual_leaves::T

    "Total number of virtual nodes in tree needed for a complete binary tree."
    virtual_nodes::T
end


# Custom print
function Base.print(io::IO, t::ImplicitTree{T}) where {T}
    print(io, "ImplicitTree{$T}(levels: $(t.levels), real_leaves: $(t.real_leaves))")
end



function ImplicitTree{T}(num_leaves::Integer) where {T <: Integer}
    @boundscheck if num_leaves < 1
        throw(DomainError(num_leaves, "must have at least one geometry!"))
    end

    lr = num_leaves                             # number of real leaves
    levels = @inbounds ilog2(lr, RoundUp) + 1   # number of binary tree levels

    lv = 2^(levels - 1) - lr                    # number of virtual leaves
    nv = 2lv - count_ones(lv)                   # number of virtual nodes
    nr = 2lr - 1 + count_ones(lv)               # number of real nodes

    ImplicitTree{T}(levels, lr, nr, lv, nv)
end


# Convenience constructors
ImplicitTree(num_leaves::Integer) = ImplicitTree{typeof(num_leaves)}(num_leaves)


"""
    memory_index(tree::ImplicitTree, implicit_index::Integer)

Return actual memory index for a node at implicit index i in a perfect BFS-labelled tree.
"""
@inline function memory_index(tree::ImplicitTree, implicit_index::Integer)

    # This will be elided when @inbounds
    @boundscheck begin
        if !(1 <= implicit_index <= 2^tree.levels - 1)
            throw(BoundsError(tree, implicit_index))
        end
    end

    # Level at which the implicit index is
    implicit_level = @inbounds ilog2(implicit_index, RoundDown) + 1

    # Number of virtual nodes at level before
    virtual_nodes_level = tree.virtual_leaves >> (tree.levels - (implicit_level - 1))

    # Total number of virtual nodes up to the level before
    virtual_nodes_before = 2 * virtual_nodes_level - count_ones(virtual_nodes_level)

    # Skipping the number of virtual_nodes we had before the implicit_index
    implicit_index - virtual_nodes_before
end


"""
    level_indices(tree::ImplicitTree, level::Integer)

Return range Tuple{Int64, Int64} of memory indices of elements at `level`.
"""
@inline function level_indices(tree::ImplicitTree, level::Integer)
    
    # This will be elided when @inbounds
    @boundscheck begin
        if !(1 <= level <= tree.levels)
            throw(BoundsError(tree, level))
        end
    end

    # Index of first element at this level
    start = @inbounds memory_index(tree, 2^(level - 1))
    nreal = 1 << (level - 1) - tree.virtual_leaves >> (tree.levels - level)
    stop = start + nreal - 1

    start, stop
end


"""
    isvirtual(tree::ImplicitTree, implicit_index::Integer)

Check if given `implicit_index` corresponds to a virtual node.
"""
@inline function isvirtual(tree::ImplicitTree, implicit_index::Integer)
    # This will be elided when @inbounds
    @boundscheck begin
        if !(1 <= implicit_index <= 2^tree.levels - 1)
            throw(BoundsError(tree, implicit_index))
        end
    end

    # Level at which the implicit index is
    level = @inbounds ilog2(implicit_index, RoundDown) + 1
    level_first = 1 << (level - 1)
    nreal = level_first - tree.virtual_leaves >> (tree.levels - level)

    # If index is beyond last real node, it's virtual
    implicit_index - level_first + 1 > nreal    
end
