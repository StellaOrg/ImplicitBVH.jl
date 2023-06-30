"""
    $(TYPEDEF)

Collected statistics about a BVH construction and contact traversal.

# Fields
    $(TYPEDFIELDS)

"""
@with_kw mutable struct BVHStats
    start_level::Union{Nothing, Int} = nothing
    num_checks::Union{Nothing, Int} = nothing
    num_contacts::Union{Nothing, Int} = nothing
end


"""
    $(TYPEDEF)

Implicit bounding volume hierarchy constructed from an iterable of some geometric primitives'
(e.g. triangles in a mesh) bounding volumes forming the [`ImplicitTree`](@ref) leaves. The leaves
and merged nodes above them can have different types - e.g. BSphere{Float64} for leaves merged into
larger BBox{Float64}.

The initial geometric primitives are sorted according to their Morton-encoded coordinates; the
unsigned integer type used for the Morton encoding can be chosen between UInt16, UInt32 and UInt64.

Finally, the tree can be incompletely-built up to a given `built_level` and later start contact
detection downwards from this level, e.g.:

```
Implicit tree from 11 bounding volumes - the virtual nodes are not stored in memory
Level                                               Nodes
  1                                                   1
  2                              2                                          3
  3                  4                     5                     6                       7v
  4            8           9         10         11         12         13          14v          15v
  5         16   17     18   19    20  21     22  23     24  25     26  27v    28v   29v    30v   31v
            -------------------------Real-----------------------------  -----------Virtual-----------
```

# Methods
    BVH(
        bounding_volumes::AbstractVector{L},
        node_type::Type{N}=L,
        morton_type::MortonUnsignedType=UInt64,
        built_level::Integer=1,
    ) where {N, L}

# Fields
    $(TYPEDFIELDS)

# Examples

Simple usage with bounding spheres and default 64-bit types:

```jldoctest
using IBVH
using StaticArrays

# Generate some simple bounding spheres
bounding_spheres = [
    BSphere(SA[0., 0., 0.], 0.5),
    BSphere(SA[0., 0., 1.], 0.6),
    BSphere(SA[0., 0., 2.], 0.5),
    BSphere(SA[0., 0., 3.], 0.4),
    BSphere(SA[0., 0., 4.], 0.6),
]

# Build BVH
bvh = BVH(bounding_spheres)

# Traverse BVH for contact detection
traversal = traverse(bvh)
@show traversal.contacts;
;

# output
traversal.contacts = [(4, 5), (1, 2), (2, 3)]
```

Using `Float32` bounding spheres for leaves, `Float32` bounding boxes for nodes above, and `UInt32`
Morton codes:

```jldoctest
using IBVH
using StaticArrays

# Generate some simple bounding spheres
bounding_spheres = [
    BSphere{Float32}(SA[0., 0., 0.], 0.5),
    BSphere{Float32}(SA[0., 0., 1.], 0.6),
    BSphere{Float32}(SA[0., 0., 2.], 0.5),
    BSphere{Float32}(SA[0., 0., 3.], 0.4),
    BSphere{Float32}(SA[0., 0., 4.], 0.6),
]

# Build BVH
bvh = BVH(bounding_spheres, BBox{Float32}, UInt32)

# Traverse BVH for contact detection
traversal = traverse(bvh)
@show traversal.contacts;
;

# output
traversal.contacts = [(4, 5), (1, 2), (2, 3)]
```

Build BVH up to level 2 and start traversing down from level 3, reusing the previous traversal
cache:

```julia
bvh = BVH(bounding_spheres, BBox{Float32}, UInt32, 2)
traversal = traverse(bvh, 3, traversal)
```
"""
@with_kw struct BVH{VN <: AbstractVector, VL <: AbstractVector, VO <: AbstractVector}
    tree::ImplicitTree{Int}

    nodes::VN
    leaves::VL
    order::VO

    built_level::Int
    stats::BVHStats=BVHStats()
end


function BVH(
    bounding_volumes::AbstractVector{L},
    node_type::Type{N}=L,
    morton_type::Type{U}=UInt,
    built_level::Integer=1,
) where {L, N, U <: MortonUnsigned}

    # Pre-compute shape of the implicit tree
    numbv = length(bounding_volumes)
    tree = ImplicitTree{Int}(numbv)

    # Ensure correctness
    @assert 1 <= built_level <= tree.levels "built_level must be above leaf-level"
    @assert firstindex(bounding_volumes) == 1 "vector types used must be 1-indexed"

    # Ensure efficiency
    isconcretetype(N) || @warn "node_type given as unsized type (e.g. BBox instead of \
                                BBox{Float64}) leading to non-inline vector storage"

    isconcretetype(L) || @warn "bounding_volumes given as unsized type (e.g. BBox instead of \
                                BBox{Float64}) leading to non-inline vector storage"

    # Compute morton codes for the bounding volumes
    mortons = similar(bounding_volumes, morton_type)
    morton_encode!(mortons, bounding_volumes)

    # Compute indices that sort codes along the Z-curve - closer objects have closer Morton codes
    order = sortperm(mortons)

    # Pre-allocate vector of bounding volumes for the real nodes above the bottom level
    bvh_nodes = similar(bounding_volumes, N, tree.real_nodes - tree.real_leaves)

    # Aggregate bounding volumes up to root
    if tree.real_nodes >= 2
        aggregate_oibvh!(bvh_nodes, bounding_volumes, tree, order, built_level)
    end

    BVH(tree, bvh_nodes, bounding_volumes, order, built_level, BVHStats())
end


# Build OIBVH nodes above the leaf-level from the bottom up, inplace
function aggregate_oibvh!(bvh_nodes, bvh_leaves, tree, order, built_level=1)

    # Special case: aggregate level above leaves - might have different node types
    aggregate_last_level!(bvh_nodes, bvh_leaves, tree, order)

    level = tree.levels - 2
    while level >= built_level
        aggregate_level!(bvh_nodes, level, tree)
        level -= 1
    end

    nothing
end


@inline function aggregate_last_level_range!(
    bvh_nodes,
    bvh_leaves,
    order,
    start_pos,
    num_nodes_next,
    irange,
)
    # The bvh_nodes are not sorted! Instead, we have the indices permutation in `order`
    @inbounds for i in irange[1]:irange[2]

        lchild_implicit = 2i - 1
        rchild_implicit = 2i

        rchild_virtual = rchild_implicit >= num_nodes_next

        lchild_index = order[lchild_implicit]
        if !rchild_virtual
            rchild_index = order[rchild_implicit]
        end

        # If using different node type than leaf type (e.g. BSphere leaves and BBox nodes) do
        # conversion; this conditional is optimised away at compile-time
        if eltype(bvh_nodes) === eltype(bvh_leaves)
            # If right child is virtual, set the parent BV to the left child one; otherwise merge
            if rchild_virtual
                bvh_nodes[start_pos - 1 + i] = bvh_leaves[lchild_index]
            else
                bvh_nodes[start_pos - 1 + i] = bvh_leaves[lchild_index] + bvh_leaves[rchild_index]
            end
        else
            if rchild_virtual
                bvh_nodes[start_pos - 1 + i] = eltype(bvh_nodes)(bvh_leaves[lchild_index])
            else
                bvh_nodes[start_pos - 1 + i] = eltype(bvh_nodes)(bvh_leaves[lchild_index],
                                                                 bvh_leaves[rchild_index])
            end
        end
    end

    nothing
end


@inline function aggregate_last_level!(bvh_nodes, bvh_leaves, tree, order)
    # Memory index of first node on this level (i.e. first above leaf-level)
    level = tree.levels - 1
    start_pos = memory_index(tree, 1 << (level - 1))

    # Number of real nodes on this level
    num_nodes = 1 << (level - 1) - tree.virtual_leaves >> (tree.levels - level)

    # Merge all pairs of children below this level
    num_nodes_next = 1 << level - tree.virtual_leaves >> (tree.levels - (level + 1))

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = TaskPartitioner(num_nodes, Threads.nthreads(), 100)
    if tp.num_tasks == 1
        @inbounds aggregate_last_level_range!(
            bvh_nodes, bvh_leaves, order,
            start_pos, num_nodes_next, (1, num_nodes)
        )
    else
        tasks = Vector{Task}(undef, tp.num_tasks)
        for i in 1:tp.num_tasks
            @inbounds tasks[i] = Threads.@spawn aggregate_last_level_range!(
                bvh_nodes, bvh_leaves, order,
                start_pos, num_nodes_next, tp[i],
            )
        end
        for i in 1:tp.num_tasks
            wait(tasks[i])
        end
    end

    nothing
end


@inline function aggregate_level_range!(
    bvh_nodes,
    start_pos,
    start_pos_next,
    num_nodes_next,
    irange,
)
    @inbounds for i in irange[1]:irange[2]
        lchild_index = start_pos_next + 2i - 2
        rchild_index = start_pos_next + 2i - 1

        if rchild_index >= start_pos_next + num_nodes_next
            # If right child is virtual, set the parent BV to the child one
            bvh_nodes[start_pos - 1 + i] = bvh_nodes[lchild_index]
        else
            # Merge children bounding volumes
            bvh_nodes[start_pos - 1 + i] = bvh_nodes[lchild_index] + bvh_nodes[rchild_index]
        end
    end

    nothing
end


@inline function aggregate_level!(bvh_nodes, level, tree)
    # Memory index of first node on this level
    start_pos = memory_index(tree, 1 << (level - 1))

    # Number of real nodes on this level
    num_nodes = 1 << (level - 1) - tree.virtual_leaves >> (tree.levels - level)

    # Merge all pairs of children below this level
    start_pos_next = memory_index(tree, 1 << level)
    num_nodes_next = 1 << level - tree.virtual_leaves >> (tree.levels - (level + 1))

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = TaskPartitioner(num_nodes, Threads.nthreads(), 100)
    if tp.num_tasks == 1
        @inbounds aggregate_level_range!(
            bvh_nodes, start_pos,
            start_pos_next, num_nodes_next, (1, num_nodes),
        )
    else
        tasks = Vector{Task}(undef, tp.num_tasks)
        for i in 1:tp.num_tasks
            @inbounds tasks[i] = Threads.@spawn aggregate_level_range!(
                bvh_nodes, start_pos,
                start_pos_next, num_nodes_next, tp[i],
            )
        end
        for i in 1:tp.num_tasks
            wait(tasks[i])
        end
    end

    nothing
end
