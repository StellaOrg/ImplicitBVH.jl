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
    # Normal constructor which builds BVH
    BVH(
        bounding_volumes::AbstractVector{L},
        node_type::Type{N}=L,
        morton_type::Type{U}=UInt32,
        built_level=1;
        options=BVHOptions(),
    ) where {L, N, U <: MortonUnsigned}

    # Copy constructor reusing previous memory; previous
    # bounding volumes are moved to new_positions.
    BVH(
        prev::BVH,
        new_positions::AbstractMatrix,
        built_level=1;
        options=BVHOptions(),
    )

# Fields
- `tree::`[`ImplicitTree`](@ref)`{I <: Integer}`
- `nodes::VN <: AbstractVector`
- `leaves::VL <: AbstractVector`
- `mortons::VM <: AbstractVector`
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
traversal.contacts = Tuple{Int32, Int32}[(1, 2), (2, 3), (4, 5)]
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
traversal.contacts = Tuple{Int32, Int32}[(1, 2), (2, 3), (4, 5)]
```

Build BVH up to level 2 and start traversing down from level 3, reusing the previous traversal
cache:

```julia
bvh = BVH(bounding_spheres, BBox{Float32}, UInt32, 2)
traversal = traverse(bvh, 3, traversal)
```

Update previous BVH bounding volumes' positions and rebuild BVH *reusing previous memory*:

```julia
new_positions = rand(3, 5)
bvh_rebuilt = BVH(bvh, new_positions)
```
"""
struct BVH{
    I <: Integer,
    VN <: AbstractVector,
    VL <: AbstractVector,
    VM <: AbstractVector,
    VO <: AbstractVector,
}
    built_level::I
    tree::ImplicitTree{I}
    nodes::VN
    leaves::VL
    mortons::VM
    order::VO
end


# Custom pretty-printing
function Base.show(io::IO, b::BVH{I, VN, VL, VM, VO}) where {I, VN, VL, VM, VO}
    print(
        io,
        """
        BVH
          built_level: $(typeof(b.built_level)) $(b.built_level)
          tree:        $(b.tree)
          nodes:       $(VN)($(size(b.nodes)))
          leaves:      $(VL)($(size(b.leaves)))
          mortons:     $(VM)($(size(b.mortons)))
          order:       $(VO)($(size(b.order)))
        """
    )
end


# Normal constructor which builds BVH
function BVH(
    bounding_volumes::AbstractVector{L},
    node_type::Type{N}=L,
    morton_type::Type{U}=UInt32,
    built_level=1;
    options=BVHOptions(),
) where {L, N, U <: MortonUnsigned}

    # Ensure correctness
    @argcheck firstindex(bounding_volumes) == 1 "BVH vector types used must be 1-indexed"

    # Ensure efficiency
    isconcretetype(N) || @warn "node_type given as unsized type (e.g. BBox instead of \
                                BBox{Float64}) leading to non-inline vector storage"

    isconcretetype(L) || @warn "bounding_volumes given as unsized type (e.g. BBox instead of \
                                BBox{Float64}) leading to non-inline vector storage"

    # Get index type from exemplar
    I = get_index_type(options)

    # Pre-compute shape of the implicit tree
    numbv = length(bounding_volumes)
    tree = ImplicitTree{I}(numbv)

    # Compute level up to which tree should be built
    built_ilevel = compute_build_level(tree, built_level)

    # Compute morton codes for the bounding volumes
    mortons = similar(bounding_volumes, morton_type)
    @inbounds morton_encode!(mortons, bounding_volumes, options)

    # Compute indices that sort codes along the Z-curve - closer objects have closer Morton codes
    order = similar(mortons, I)
    AK.sortperm!(order, mortons, block_size=options.block_size)

    # Pre-allocate vector of bounding volumes for the real nodes above the bottom level
    bvh_nodes = similar(bounding_volumes, N, Int(tree.real_nodes - tree.real_leaves))

    # Aggregate bounding volumes up to built_ilevel
    if tree.real_nodes >= 2
        aggregate_oibvh!(bvh_nodes, bounding_volumes, tree, order, built_ilevel, options)
    end

    BVH(I(built_ilevel), tree, bvh_nodes, bounding_volumes, mortons, order)
end


# Copy constructor reusing previous memory; previous bounding volumes are moved to new_positions.
function BVH(
    prev::BVH,
    new_positions::AbstractMatrix,
    built_level=1;
    options=BVHOptions(),
)

    # Ensure correctness
    @argcheck size(new_positions, 1) == 3 "new_positions need 3D coordinates on first axis"
    @argcheck size(new_positions, 2) == length(prev.leaves) "Different number of new_positions"
    @argcheck firstindex(new_positions, 1) == 1 "BVH vector types must be 1-indexed"
    @argcheck firstindex(new_positions, 2) == 1 "BVH vector types must be 1-indexed"

    # Get index type from exemplar
    index_type = get_index_type(options)

    # Extract previous BVH fields
    tree = prev.tree
    bvh_nodes = prev.nodes
    bounding_volumes = prev.leaves
    mortons = prev.mortons
    order = prev.order

    # Compute level up to which tree should be built
    built_ilevel = index_type(compute_build_level(tree, built_level))

    # Move the centres of previous bounding volumes to the coordinates of new_positions; ensure we
    # use the same type as in prev
    @inbounds @inline function _move_bv!(bounding_volumes, new_positions, i)
        bv = bounding_volumes[i]
        c = center(bv)
        T = eltype(c)

        # Special-case BSphere which can be translated by just overwriting centre
        if bv isa BSphere
            new_x = (T(new_positions[1, i]), T(new_positions[2, i]), T(new_positions[3, i]))
            bounding_volumes[i] = BSphere{T}(new_x, bv.r)
        else
            dx = (T(new_positions[1, i]) - c[1],
                  T(new_positions[2, i]) - c[2],
                  T(new_positions[3, i]) - c[3])
            bounding_volumes[i] = translate(bv, dx)
        end
    end

    AK.foreachindex(
        bounding_volumes,
        block_size=options.block_size,
        scheduler=options.scheduler,
        max_tasks=options.num_threads,
        min_elems=options.min_boundings_per_thread,
    ) do i
        _move_bv!(bounding_volumes, new_positions, i)
    end

    # Recompute Morton codes
    @inbounds morton_encode!(mortons, bounding_volumes, options)

    # Compute indices that sort codes along the Z-curve - closer objects have closer Morton codes
    if mortons isa AbstractGPUVector
        AK.merge_sortperm_lowmem!(order, mortons, block_size=options.block_size)
    else
        sortperm!(order, mortons)
    end

    # Aggregate bounding volumes up to built_ilevel
    if tree.real_nodes >= 2
        aggregate_oibvh!(bvh_nodes, bounding_volumes, tree, order, built_ilevel, options)
    end

    BVH(built_ilevel, tree, bvh_nodes, bounding_volumes, mortons, order)
end


# Compute level up to which tree should be built
function compute_build_level(tree, built_level)

    index_type = get_index_type(tree)

    if built_level isa Integer
        @argcheck 1 <= built_level <= tree.levels
        built_ilevel = index_type(built_level)
    elseif built_level isa AbstractFloat
        @argcheck 0 <= built_level <= 1
        built_ilevel = round(index_type, tree.levels + (1 - tree.levels) * built_level)
    else
        throw(TypeError(:BVH, "built_level (the level to build BVH up to)",
                        Union{Integer, AbstractFloat}, typeof(built_level)))
    end

    built_ilevel
end

# Build ImplicitBVH nodes above the leaf-level from the bottom up, inplace
function aggregate_oibvh!(bvh_nodes, bvh_leaves, tree, order, built_level, options)

    # Special case: aggregate level above leaves - might have different node types
    aggregate_last_level!(bvh_nodes, bvh_leaves, tree, order, options)

    level = tree.levels - 2
    while level >= built_level
        aggregate_level!(bvh_nodes, level, tree, options)
        level -= 1
    end

    nothing
end


@inline function aggregate_last_level!(bvh_nodes, bvh_leaves, tree, order, options)

    # Make sure types in the core kernel (which will be executed on CPU and GPU) are coherent;
    # important, as GPUs are 32-bit machines
    I = get_index_type(options)

    # Memory index of first node on this level (i.e. first above leaf-level)
    level = tree.levels - 0x1
    start_pos::I = memory_index(tree, pow2(level - 0x1))

    # Number of real nodes on this level
    num_nodes::I = pow2(level - 0x1) - tree.virtual_leaves >> (tree.levels - level)

    # Merge all pairs of children below this level
    num_nodes_next::I = pow2(level) - tree.virtual_leaves >> (tree.levels - (level + 0x1))

    # Multithreaded CPU / GPU implementation
    aggregate_last_level_kernel!(
        0x1:num_nodes, get_backend(bvh_nodes), options,
        bvh_nodes, bvh_leaves, order,
        start_pos, num_nodes_next,
    )

    nothing
end


function aggregate_last_level_kernel!(
    irange, backend, options,
    bvh_nodes, bvh_leaves, order,
    start_pos, num_nodes_next,
)
    AK.foreachindex(
        irange, backend,
        block_size=options.block_size,
        scheduler=options.scheduler,
        max_tasks=options.num_threads,
        min_elems=options.min_boundings_per_thread,
    ) do i
        _aggregate_last_level_at!(
            bvh_nodes, bvh_leaves, order,
            start_pos, num_nodes_next, i,
        )
    end
end


@inline @inbounds function _aggregate_last_level_at!(
    bvh_nodes,
    bvh_leaves,
    order,
    start_pos,
    num_nodes_next,
    i,
)
    lchild_implicit = 0x2 * i - 0x1
    rchild_implicit = 0x2 * i

    rchild_virtual = rchild_implicit > num_nodes_next

    # The bvh_nodes are not sorted! Instead, we have the indices permutation in `order`
    lchild_index = order[lchild_implicit]
    if !rchild_virtual
        rchild_index = order[rchild_implicit]
    end

    # If using different node type than leaf type (e.g. BSphere leaves and BBox nodes) do
    # conversion; this conditional is optimised away at compile-time
    if eltype(bvh_nodes) === eltype(bvh_leaves)
        # If right child is virtual, set the parent BV to the left child one; otherwise merge
        if rchild_virtual
            bvh_nodes[start_pos - 0x1 + i] = bvh_leaves[lchild_index]
        else
            bvh_nodes[start_pos - 0x1 + i] = bvh_leaves[lchild_index] + bvh_leaves[rchild_index]
        end
    else
        if rchild_virtual
            bvh_nodes[start_pos - 0x1 + i] = eltype(bvh_nodes)(bvh_leaves[lchild_index])
        else
            bvh_nodes[start_pos - 0x1 + i] = eltype(bvh_nodes)(bvh_leaves[lchild_index],
                                                               bvh_leaves[rchild_index])
        end
    end

    nothing
end


@inline function aggregate_level!(bvh_nodes, level, tree, options)

    # Make sure types in the core kernel (which will be executed on CPU and GPU) are coherent;
    # important, as GPUs are 32-bit machines
    I = get_index_type(options)

    # Memory index of first node on this level
    start_pos::I = memory_index(tree, pow2(level - 0x1))

    # Number of real nodes on this level
    num_nodes::I = pow2(level - 0x1) - tree.virtual_leaves >> (tree.levels - level)

    # Merge all pairs of children below this level
    start_pos_next::I = memory_index(tree, pow2(level))
    num_nodes_next::I = pow2(level) - tree.virtual_leaves >> (tree.levels - (level + 0x1))

    # Multithreaded CPU / GPU implementation
    aggregate_level_kernel!(
        0x1:num_nodes, get_backend(bvh_nodes), options,
        bvh_nodes,
        start_pos, start_pos_next, num_nodes_next,
    )

    nothing
end


function aggregate_level_kernel!(
    irange, backend, options,
    bvh_nodes,
    start_pos, start_pos_next, num_nodes_next,
)
    AK.foreachindex(
        irange, backend,
        block_size=options.block_size,
        scheduler=options.scheduler,
        max_tasks=options.num_threads,
        min_elems=options.min_boundings_per_thread,
    ) do i
        _aggregate_level_at!(bvh_nodes, start_pos, start_pos_next, num_nodes_next, i)
    end    
end


@inline @inbounds function _aggregate_level_at!(
    bvh_nodes,
    start_pos,
    start_pos_next,
    num_nodes_next,
    i,
)
    lchild_index = start_pos_next + 0x2 * i - 0x2
    rchild_index = start_pos_next + 0x2 * i - 0x1

    if rchild_index > start_pos_next + num_nodes_next - 0x1
        # If right child is virtual, set the parent BV to the child one
        bvh_nodes[start_pos - 0x1 + i] = bvh_nodes[lchild_index]
    else
        # Merge children bounding volumes
        bvh_nodes[start_pos - 0x1 + i] = bvh_nodes[lchild_index] + bvh_nodes[rchild_index]
    end

    nothing
end
