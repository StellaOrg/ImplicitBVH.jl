"""
    $(TYPEDEF)

Implicit bounding volume hierarchy constructed from an iterable of some geometric primitives'
(e.g. triangles in a mesh) bounding volumes forming the [`ImplicitTree`](@ref) leaves. The leaves
and merged nodes above them can have different types - e.g. `BSphere{Float64}` for leaves
merged into larger `BBox{Float64}`.

The initial geometric primitives are sorted according to their Morton-encoded coordinates; the
encoding algorithm is specified within the [`BVHOptions`](@ref).

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
        node_type::Type{N}=BBox{Float32};
        built_level::Union{Integer, AbstractFloat}=1,
        cache::Union{Nothing, BVH}=nothing,
        options=BVHOptions(),
    ) where {L, N}

# Fields
- `built_level::Int` - level up to which the BVH tree has been built
- `tree::`[`ImplicitTree`](@ref)`{I <: Integer}`
- `skips::VS <: AbstractVector` - vector of skips (number of indices to jump in memory, per level)
- `nodes::VN <: AbstractVector` - vector of bounding volumes for the internal nodes above the leaves
- `leaves::VL <: AbstractVector` - vector of sorted [`BoundingVolume`](@ref) for the leaves

# Examples
Simple usage with bounding spheres as leaves (defaults include `BBox{Float32}` nodes, `UInt32`
Morton codes, and `Int32` indices):

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

Using `Float32` bounding spheres for leaves, `Float32` bounding boxes for nodes above, `UInt64`
Morton codes and `Int64` indices:

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
options = BVHOptions(index=Int64, morton=DefaultMortonAlgorithm(UInt64))
bvh = BVH(bounding_spheres, BBox{Float32}, options=options)

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
bvh = BVH(bounding_spheres, BBox{Float32}, built_level=2)
traversal = traverse(bvh, start_level=3, cache=traversal)
```

Reuse previous BVH memory for a new BVH (i.e. the nodes, but not the leaves, which are modified
in-place):

```julia
bvh = BVH(bounding_spheres, BBox{Float32}, cache=bvh)
```

Move previous BVH leaves / bounding volumes to new locations then rebuild BVH in-place:
```julia
for ibv in eachindex(bvh.leaves)
    bvh.leaves[ibv] = BoundingVolume(
        bvh.leaves[ibv].volume,             # Update if needed
        bvh.leaves[ibv].index,              # Your simulation indices, reported in contacts
        bvh.leaves[ibv].morton,             # Will be recomputed when rebuilding
    )
end
bvh = BVH(bvh.leaves, BBox{Float32}, cache=bvh)
```

Manually wrap bounding volumes into [`BoundingVolume`](@ref) structs before building BVH:
```jldoctest
using ImplicitBVH
using ImplicitBVH: BoundingVolume, BSphere, BBox

# Generate some simple bounding spheres with explicit simulation indices (will be reported in
# contacts; allows different indexing strategies) and dummy morton codes (will be computed when
# building the BVH)
bounding_spheres = [
    BoundingVolume(BSphere{Float32}([0., 0., 1.], 0.6), Int32(1), UInt32(0)),
    BoundingVolume(BSphere{Float32}([0., 0., 2.], 0.5), Int32(2), UInt32(0)),
    BoundingVolume(BSphere{Float32}([0., 0., 0.], 0.5), Int32(3), UInt32(0)),
    BoundingVolume(BSphere{Float32}([0., 0., 3.], 0.4), Int32(4), UInt32(0)),
    BoundingVolume(BSphere{Float32}([0., 0., 4.], 0.6), Int32(5), UInt32(0)),
]

# Build BVH
bvh = BVH(bounding_spheres, BBox{Float32})

# The bounding_spheres was modified in-place, without extra allocations
@show bounding_spheres[1];
;

# output
bounding_spheres[1] = ImplicitBVH.BoundingVolume{ImplicitBVH.BSphere{Float32}, Int32, UInt32}(ImplicitBVH.BSphere{Float32}((0.0f0, 0.0f0, 0.0f0), 0.5f0), 3, 0x06186186)
```
"""
struct BVH{
    I <: Integer,
    VS <: AbstractVector,
    VN <: AbstractVector,
    VL <: AbstractVector,
}
    built_level::I
    tree::ImplicitTree{I}
    skips::VS
    nodes::VN
    leaves::VL
end


# For GPU transfer
function Adapt.adapt_structure(to, bvh::BVH)
    BVH(
        Adapt.adapt_structure(to, bvh.built_level),
        Adapt.adapt_structure(to, bvh.tree),
        Adapt.adapt_structure(to, bvh.skips),
        Adapt.adapt_structure(to, bvh.nodes),
        Adapt.adapt_structure(to, bvh.leaves),
    )
end


# Custom pretty-printing
function Base.show(io::IO, b::BVH{I, VS, VN, VL}) where {I, VS, VN, VL}
    print(
        io,
        """
        BVH
          built_level: $(typeof(b.built_level)) $(b.built_level)
          tree:        $(b.tree)
          skips:       $(VS)($(size(b.skips)))
          nodes:       $(VN)($(size(b.nodes)))
          leaves:      $(VL)($(size(b.leaves)))
        """
    )
end


# Normal constructor which builds BVH
function BVH(
    bounding_volumes::AbstractVector{L},
    node_type::Type{N}=BBox{Float32};
    built_level::Union{Integer, AbstractFloat}=1,
    cache::Union{Nothing, BVH}=nothing,
    options=BVHOptions(),
) where {L, N}

    # Ensure correctness
    @argcheck firstindex(bounding_volumes) == 1 "BVH vector types used must be 1-indexed"

    # Ensure efficiency
    isconcretetype(N) || @warn "node_type given as unsized type (e.g. BBox instead of \
                                BBox{Float64}) leading to non-inline vector storage"

    isconcretetype(L) || @warn "bounding_volumes given as unsized type (e.g. BBox instead of \
                                BBox{Float64}) leading to non-inline vector storage"

    # Get index type from exemplar
    I = get_index_type(options)

    # Wrap the bounding volumes into BoundingVolume (with index and morton code) if needed
    if eltype(bounding_volumes) <: BoundingVolume
        check_bounding_volume_types(bounding_volumes, options)
        bounding_volumes_wrapped = bounding_volumes
    else
        bounding_volumes_wrapped = wrap_bounding_volumes(bounding_volumes, options)
    end

    # Pre-compute shape of the implicit tree
    numbv = length(bounding_volumes)
    tree = ImplicitTree{I}(numbv)

    # Pre-compute skips (number of indices to jump in memory, per level) for traversal
    if isnothing(cache)
        skips = similar(bounding_volumes, I, tree.levels)
    else
        @argcheck eltype(cache.skips) === I
        skips = cache.skips
        length(skips) == tree.levels || resize!(skips, tree.levels)
    end
    compute_skips!(skips, tree)

    # Compute level up to which tree should be built
    built_ilevel = compute_build_level(tree, built_level)

    # Compute morton codes for the bounding volumes
    morton_encode!(bounding_volumes_wrapped, options)

    # Sort bounding volumes along the Z-curve - closer objects have closer Morton codes
    AK.sort!(
        bounding_volumes_wrapped, by=bv->bv.morton,
        max_tasks=options.num_threads,
        min_elems=options.min_sorts_per_thread,
        block_size=options.block_size,
    )

    # Pre-allocate vector of bounding volumes for the real nodes above the bottom level
    num_nodes = Int(tree.real_nodes - tree.real_leaves)
    if isnothing(cache)
        bvh_nodes = similar(bounding_volumes, N, num_nodes)
    else
        @argcheck eltype(cache.nodes) === N
        bvh_nodes = cache.nodes
        length(bvh_nodes) == num_nodes || resize!(bvh_nodes, num_nodes)
    end

    # Aggregate bounding volumes up to built_ilevel
    if tree.real_nodes >= 2
        aggregate_oibvh!(bvh_nodes, bounding_volumes_wrapped, tree, built_ilevel, options)
    end

    BVH(I(built_ilevel), tree, skips, bvh_nodes, bounding_volumes_wrapped)
end


# Old constructor interface
function BVH(
    bounding_volumes::AbstractVector{L},
    node_type::Type{N},
    morton_type::Type{U},
    built_level=1,
    cache::Union{Nothing, BVH}=nothing;
    options=BVHOptions(),
) where {L, N, U <: Union{UInt16, UInt32, UInt64}}

    # Update options with given morton type
    options_updated = BVHOptions(
        index_exemplar=options.index_exemplar,
        morton=U(0),
        mins=options.mins,
        maxs=options.maxs,
        compute_extrema=options.compute_extrema,
        block_size=options.block_size,
        num_threads=options.num_threads,
        min_sorts_per_thread=options.min_sorts_per_thread,
        min_boundings_per_thread=options.min_boundings_per_thread,
        min_mortons_per_thread=options.min_mortons_per_thread,
    )

    BVH(
        bounding_volumes,
        node_type;
        built_level=built_level,
        cache=cache,
        options=options_updated,
    )
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


function wrap_bounding_volumes(bounding_volumes, options)
    LeafType = eltype(bounding_volumes)
    IndexType = typeof(options.index_exemplar)
    MortonType = eltype(options.morton)

    bounding_volumes_wrapped = similar(
        bounding_volumes,
        BoundingVolume{LeafType, IndexType, MortonType},
    )
    index_exemplar = IndexType(0)
    morton_dummy = MortonType(0)

    AK.foreachindex(
        bounding_volumes,
        block_size=options.block_size,
        max_tasks=options.num_threads,
    ) do i
        bounding_volumes_wrapped[i] = BoundingVolume(
            bounding_volumes[i],
            typeof(index_exemplar)(i),
            morton_dummy,
        )
    end
    bounding_volumes_wrapped
end


function check_bounding_volume_types(::AbstractVector{<:BoundingVolume{V, I, M}}, options) where {V, I, M}
    @argcheck I === typeof(options.index_exemplar) "BoundingVolume index type $(I) does not match \
                                                    BVHOptions index_exemplar type \
                                                    $(typeof(options.index_exemplar))"
    @argcheck M === eltype(options.morton) "BoundingVolume morton type $(M) does not match \
                                            BVHOptions morton type $(eltype(options.morton))"
    nothing
end


# Build ImplicitBVH nodes above the leaf-level from the bottom up, inplace
function aggregate_oibvh!(bvh_nodes, bvh_leaves, tree, built_level, options)

    # Special case: aggregate level above leaves - might have different node types
    aggregate_last_level!(bvh_nodes, bvh_leaves, tree, options)

    level = tree.levels - 2
    while level >= built_level
        aggregate_level!(bvh_nodes, level, tree, options)
        level -= 1
    end

    nothing
end


@inline function aggregate_last_level!(bvh_nodes, bvh_leaves, tree, options)

    # Make sure types in the core kernel (which will be executed on CPU and GPU) are coherent;
    # important, as GPUs are 32-bit machines
    I = get_index_type(options)

    # Memory index of first node on this level (i.e. first above leaf-level)
    level::I = tree.levels - 0x1
    start_pos::I = memory_index(tree, pow2(level - 0x1))

    # Number of real nodes on this level
    num_nodes::I = pow2(level - 0x1) - tree.virtual_leaves >> 0x1

    # Merge all pairs of children below this level
    num_nodes_next::I = tree.real_leaves

    # Multithreaded CPU / GPU implementation
    aggregate_last_level_kernel!(
        0x1:num_nodes, get_backend(bvh_nodes), options,
        bvh_nodes, bvh_leaves,
        start_pos, num_nodes_next,
    )

    nothing
end


function aggregate_last_level_kernel!(
    irange, backend, options,
    bvh_nodes, bvh_leaves,
    start_pos, num_nodes_next,
)
    AK.foreachindex(
        irange, backend,
        block_size=options.block_size,
        max_tasks=options.num_threads,
        min_elems=options.min_boundings_per_thread,
    ) do i
        _aggregate_last_level_at!(
            bvh_nodes, bvh_leaves,
            start_pos, num_nodes_next, i,
        )
    end
end


@inline function _aggregate_last_level_at!(
    nodes, leaves,
    start_pos, num_nodes_next,
    i,
)
    lchild_index = 0x2 * i - 0x1
    rchild_index = 0x2 * i
    rchild_isvirtual = rchild_index > num_nodes_next

    # If using different node type than leaf type (e.g. BSphere leaves and BBox nodes) do
    # conversion; this conditional is optimised away at compile-time
    if same_leaf_node(leaves, nodes)
        # If right child is virtual, set the parent BV to the left child one; otherwise merge
        if rchild_isvirtual
            @inbounds nodes[start_pos - 0x1 + i] = leaves[lchild_index].volume
        else
            @inbounds nodes[start_pos - 0x1 + i] = (leaves[lchild_index].volume +
                                                    leaves[rchild_index].volume)
        end
    else
        NodeType = eltype(nodes)
        if rchild_isvirtual
            @inbounds nodes[start_pos - 0x1 + i] = NodeType(leaves[lchild_index].volume)
        else
            @inbounds nodes[start_pos - 0x1 + i] = NodeType(leaves[lchild_index].volume,
                                                            leaves[rchild_index].volume)
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
        max_tasks=options.num_threads,
        min_elems=options.min_boundings_per_thread,
    ) do i
        _aggregate_level_at!(bvh_nodes, start_pos, start_pos_next, num_nodes_next, i)
    end    
end


@inline function _aggregate_level_at!(
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
        @inbounds bvh_nodes[start_pos - 0x1 + i] = bvh_nodes[lchild_index]
    else
        # Merge children bounding volumes
        @inbounds bvh_nodes[start_pos - 0x1 + i] = (bvh_nodes[lchild_index] +
                                                    bvh_nodes[rchild_index])
    end

    nothing
end
