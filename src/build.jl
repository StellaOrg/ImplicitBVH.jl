"""
    $(TYPEDEF)

Implicit bounding volume hierarchy constructed from an iterable of some geometric primitives'
(e.g. triangles in a mesh) bounding volumes forming the [`ImplicitTree`](@ref) leaves. The leaves
and merged nodes above them can have different types - e.g. `BSphere{Float64}` for leaves
merged into larger `BBox{Float64}`.

The initial geometric primitives are sorted according to their Morton-encoded coordinates; the
unsigned integer type used for the Morton encoding can be chosen between `$(MortonUnsigned)`.

Finally, the tree can be incompletely-built up to a given `built_level` and later start contact
detection downwards from this level, e.g.:

```
Implicit tree from 5 bounding volumes - i.e. the real leaves

Tree Level          Nodes & Leaves               Build Up    Traverse Down
    1                     1                         Ʌ              |
    2             2               3                 |              |
    3         4       5       6        7v           |              |
    4       8   9   10 11   12 13v  14v  15v        |              V
            -------Real------- ---Virtual---
```

# Methods
    function BVH(
        bounding_volumes::AbstractVector{L},
        node_type::Type{N}=L,
        morton_type::Type{U}=UInt,
        built_level::Integer=1;
        num_threads=Threads.nthreads(),
    ) where {L, N, U <: MortonUnsigned}

# Fields
- `tree::`[`ImplicitTree`](@ref)`{Int}`
- `nodes::VN <: AbstractVector`
- `leaves::VL <: AbstractVector`
- `order::VO <: AbstractVector`
- `built_level::Int`


# Examples

Simple usage with bounding spheres and default 64-bit types:

```jldoctest
using ImplicitBVH
using ImplicitBVH: BSphere

# Generate some simple bounding spheres
bounding_spheres = [
    BSphere([0., 0., 0.], 0.5),
    BSphere([0., 0., 1.], 0.6),
    BSphere([0., 0., 2.], 0.5),
    BSphere([0., 0., 3.], 0.4),
    BSphere([0., 0., 4.], 0.6),
]

# Build BVH
bvh = BVH(bounding_spheres)

# Traverse BVH for contact detection
traversal = traverse(bvh)
@show traversal.contacts;
;

# output
traversal.contacts = [(1, 2), (2, 3), (4, 5)]
```

Using `Float32` bounding spheres for leaves, `Float32` bounding boxes for nodes above, and `UInt32`
Morton codes:

```jldoctest
using ImplicitBVH
using ImplicitBVH: BBox, BSphere

# Generate some simple bounding spheres
bounding_spheres = [
    BSphere{Float32}([0., 0., 0.], 0.5),
    BSphere{Float32}([0., 0., 1.], 0.6),
    BSphere{Float32}([0., 0., 2.], 0.5),
    BSphere{Float32}([0., 0., 3.], 0.4),
    BSphere{Float32}([0., 0., 4.], 0.6),
]

# Build BVH
bvh = BVH(bounding_spheres, BBox{Float32}, UInt32)

# Traverse BVH for contact detection
traversal = traverse(bvh)
@show traversal.contacts;
;

# output
traversal.contacts = [(1, 2), (2, 3), (4, 5)]
```

Build BVH up to level 2 and start traversing down from level 3, reusing the previous traversal
cache:

```julia
bvh = BVH(bounding_spheres, BBox{Float32}, UInt32, 2)
traversal = traverse(bvh, 3, traversal)
```
"""
struct BVH{VN <: AbstractVector, VL <: AbstractVector, VO <: AbstractVector}
    built_level::Int
    tree::ImplicitTree{Int}
    nodes::VN
    leaves::VL
    order::VO
end


# Custom pretty-printing
function Base.show(io::IO, b::BVH{VN, VL, VO}) where {VN, VL, VO}
    print(
        io,
        """
        BVH
          built_level: $(typeof(b.built_level)) $(b.built_level)
          tree:        $(b.tree)
          nodes:       $(VN)($(size(b.nodes)))
          leaves:      $(VL)($(size(b.leaves)))
          order:       $(VO)($(size(b.order)))
        """
    )
end



function BVH(
    bounding_volumes::AbstractVector{L},
    node_type::Type{N}=L,
    morton_type::Type{U}=UInt,
    built_level=1;
    num_threads=Threads.nthreads(),
) where {L, N, U <: MortonUnsigned}

    # Ensure correctness
    @assert firstindex(bounding_volumes) == 1 "BVH vector types used must be 1-indexed"

    # Ensure efficiency
    isconcretetype(N) || @warn "node_type given as unsized type (e.g. BBox instead of \
                                BBox{Float64}) leading to non-inline vector storage"

    isconcretetype(L) || @warn "bounding_volumes given as unsized type (e.g. BBox instead of \
                                BBox{Float64}) leading to non-inline vector storage"

    # Pre-compute shape of the implicit tree
    numbv = length(bounding_volumes)
    tree = ImplicitTree{Int}(numbv)

    # Compute level up to which tree should be built
    if built_level isa Integer
        @assert 1 <= built_level <= tree.levels
        built_ilevel = Int(built_level)
    elseif built_level isa AbstractFloat
        @assert 0 <= built_level <= 1
        built_ilevel = round(Int, tree.levels + (1 - tree.levels) * built_level)
    else
        throw(TypeError(:BVH, "built_level (the level to build BVH up to)",
                        Union{Integer, AbstractFloat}, typeof(built_level)))
    end

    # Compute morton codes for the bounding volumes
    mortons = similar(bounding_volumes, morton_type)
    @inbounds morton_encode!(mortons, bounding_volumes, num_threads=num_threads)

    # Compute indices that sort codes along the Z-curve - closer objects have closer Morton codes
    # TODO: check parallel SyncSort or ThreadsX.QuickSort
    order = sortperm(mortons)

    # Pre-allocate vector of bounding volumes for the real nodes above the bottom level
    bvh_nodes = similar(bounding_volumes, N, tree.real_nodes - tree.real_leaves)

    # Aggregate bounding volumes up to root
    if tree.real_nodes >= 2
        aggregate_oibvh!(bvh_nodes, bounding_volumes, tree, order, built_ilevel, num_threads)
    end

    BVH(built_ilevel, tree, bvh_nodes, bounding_volumes, order)
end


# Build ImplicitBVH nodes above the leaf-level from the bottom up, inplace
function aggregate_oibvh!(bvh_nodes, bvh_leaves, tree, order, built_level, num_threads)

    # Special case: aggregate level above leaves - might have different node types
    aggregate_last_level!(bvh_nodes, bvh_leaves, tree, order, num_threads)

    level = tree.levels - 2
    while level >= built_level
        aggregate_level!(bvh_nodes, level, tree, num_threads)
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

        rchild_virtual = rchild_implicit > num_nodes_next

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


@inline function aggregate_last_level!(bvh_nodes, bvh_leaves, tree, order, num_threads)
    # Memory index of first node on this level (i.e. first above leaf-level)
    level = tree.levels - 1
    start_pos = memory_index(tree, pow2(level - 1))

    # Number of real nodes on this level
    num_nodes = pow2(level - 1) - tree.virtual_leaves >> (tree.levels - level)

    # Merge all pairs of children below this level
    num_nodes_next = pow2(level) - tree.virtual_leaves >> (tree.levels - (level + 1))

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = TaskPartitioner(num_nodes, num_threads, 100)
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

        if rchild_index > start_pos_next + num_nodes_next - 1
            # If right child is virtual, set the parent BV to the child one
            bvh_nodes[start_pos - 1 + i] = bvh_nodes[lchild_index]
        else
            # Merge children bounding volumes
            bvh_nodes[start_pos - 1 + i] = bvh_nodes[lchild_index] + bvh_nodes[rchild_index]
        end
    end

    nothing
end


@inline function aggregate_level!(bvh_nodes, level, tree, num_threads)
    # Memory index of first node on this level
    start_pos = memory_index(tree, pow2(level - 1))

    # Number of real nodes on this level
    num_nodes = pow2(level - 1) - tree.virtual_leaves >> (tree.levels - level)

    # Merge all pairs of children below this level
    start_pos_next = memory_index(tree, pow2(level))
    num_nodes_next = pow2(level) - tree.virtual_leaves >> (tree.levels - (level + 1))

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = TaskPartitioner(num_nodes, num_threads, 100)
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
