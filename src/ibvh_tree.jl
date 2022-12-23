"""
    $(TYPEDEF)

Alias for a tuple of two indices representing e.g. a contacting pair.
"""
const IndexPair = Tuple{Int64, Int64}


"""
    $(TYPEDEF)

A single BVH leaf containing a given bounding `volume::BV`, its corresponding `index::Int64` in
the input vector and the computed `morton::M` value.

# Fields
    $(TYPEDFIELDS)
"""
@with_kw struct BVHLeaf{M, BV}
    morton::M
    volume::BV
    index::Int64
end

bvtype(::AbstractVector{BVHLeaf{M, BV}}) where {M, BV} = BV


"""
    $(TYPEDEF)

Collected statistics about a BVH construction and contact traversal.

# Fields
    $(TYPEDFIELDS)

"""
@with_kw mutable struct BVHStats
    start_level::Union{Nothing, Int64} = nothing
    num_checks::Union{Nothing, Int64} = nothing
    num_contacts::Union{Nothing, Int64} = nothing
end


"""
    $(TYPEDEF)

Collected BVH traversal `contacts` list, plus the two buffers `cache1` and `cache2` which can be
reused for future traversals to minimise memory allocations.
"""
@with_kw struct BVHTraversal{VC <: AbstractVector}
    num_contacts::Int64
    cache1::VC
    cache2::VC
end


function Base.getproperty(bt::BVHTraversal, sym::Symbol)
   if sym === :contacts
       return @view bt.cache1[1:bt.num_contacts]
   else
       return getfield(bt, sym)
   end
end

Base.propertynames(::BVHTraversal) = (:contacts, :num_contacts, :cache1, :cache2)




"""
    $(TYPEDEF)

Implicit bounding volume hierarchy constructed from an iterable of some geometric primitives'
(e.g. triangles in a mesh) bounding volumes forming the [`ImplicitTree`](@ref) leaves. The leaves
and merged nodes above them can have different types - e.g. BSphere{Float64} for leaves merged into
larger BBox{Float64}.

The initial geometric primitives are sorted according to their Morton-encoded coordinates; the
unsigned integer type used for the Morton encoding can be chosen between UInt16, UInt32 and UInt64.

Finally, the tree can be incompletely-built up to a given `built_level` and later start contact
detection downwards from this level.

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
@with_kw struct BVH{VN <: AbstractVector, VL <: AbstractVector}
    tree::ImplicitTree{Int64}
    nodes::VN
    leaves::VL

    built_level::Int64
    stats::BVHStats=BVHStats()
end




function BVH(
    bounding_volumes::AbstractVector{L},
    node_type::Type{N}=L,
    morton_type::MortonUnsignedType=UInt64,
    built_level::Integer=1,
) where {N, L}

    # Pre-compute shape of the implicit tree
    numbv = length(bounding_volumes)
    tree = ImplicitTree{Int64}(numbv)

    # Ensure correctness / efficiency
    @assert built_level <= tree.levels "built_level must be above leaf-level"
    @assert firstindex(bounding_volumes) == 1 "vector types used must be 1-indexed"
    N isa DataType || @warn "node_type given as unsized type (e.g. BBox instead of BBox{Float64}) \
                             leading to non-inline vector storage"

    # Create vector of leaves, storing BV indices, morton codes and the actual BVs; if previous
    # BVHLeaf are used carry on
    if L <: BVHLeaf
        bvh_leaves = bounding_volumes
    else
        bvh_leaves = similar(bounding_volumes, BVHLeaf{morton_type, L}, numbv)
        fill_bvh_leaves!(bvh_leaves, bounding_volumes)
    end

    # Sort Vector{BVHLeaf} by Morton code; the index and bounding volume will be moved with it
    # [TODO]: This is the current bottleneck; perhaps implement this:
    # Theoretically-Efficient and Practical Parallel In-Place Radix Sorting
    # https://people.csail.mit.edu/jshun/RegionsSort.pdf
    # https://github.com/omarobeya/parallel-inplace-radixsort
    sort!(bvh_leaves; by=b->b.morton)

    # Pre-allocate vector of bounding volumes for the real nodes above the bottom level
    bvh_nodes = similar(bvh_leaves, N, tree.real_nodes - tree.real_leaves)

    # Aggregate bounding volumes up to root
    if tree.real_nodes >= 2
        aggregate_oibvh!(bvh_nodes, bvh_leaves, tree, built_level)
    end

    BVH(tree, bvh_nodes, bvh_leaves, built_level, BVHStats())
end




"""
    traverse(
        bvh::BVH,
        start_level=max(bvh.tree.levels ÷ 2, bvh.built_level),
        cache::Union{Nothing, BVHTraversal}=nothing,
    )::BVHTraversal

Traverse `bvh` downwards from `start_level`, returning all contacting bounding volume leaves. The
returned [`BVHTraversal`](@ref) also contains two contact buffers that can be reused on future
traversals.
    
# Examples

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
traversal = traverse(bvh, 2)

# Reuse traversal buffers for future contact detection - possibly with different BVHs
traversal = traverse(bvh, 2, traversal)
@show traversal.contacts;
;

# output
traversal.contacts = [(4, 5), (1, 2), (2, 3)]
```
"""
function traverse(
    bvh,
    start_level=max(bvh.tree.levels ÷ 2, bvh.built_level),
    cache::Union{Nothing, BVHTraversal}=nothing,
)

    @assert bvh.tree.levels >= start_level >= bvh.built_level
    bvh.stats.start_level = start_level

    # No contacts / traversal for a single node
    if bvh.tree.real_nodes <= 1
        return BVHTraversal(0, similar(bvh.nodes, IndexPair, 0), similar(bvh.nodes, IndexPair, 0))
    end

    # Allocate and add all possible BVTT contact pairs to start with
    bvtt1, bvtt2, num_bvtt = initial_bvtt(bvh, start_level, cache)

    bvh.stats.num_checks = num_bvtt

    level = start_level
    while level < bvh.tree.levels
        # We can have maximum 4 new checks per contact-pair; resize destination BVTT accordingly
        length(bvtt2) < 4 * num_bvtt && resize!(bvtt2, 4 * 4 * num_bvtt)

        # Check contacts in bvtt1 and add future checks in bvtt2; only sprout self-checks before
        # second-to-last level as leaf self-checks are pointless
        self_checks = level < bvh.tree.levels - 1
        num_bvtt = traverse_nodes_atomic!(bvh, bvtt1, bvtt2, num_bvtt, self_checks)

        bvh.stats.num_checks += num_bvtt

        # Swap source and destination buffers for next iteration
        bvtt1, bvtt2 = bvtt2, bvtt1
        level += 1
    end

    # Arrived at final leaf level, now populating contact list
    length(bvtt2) < num_bvtt && resize!(bvtt2, num_bvtt)
    num_bvtt = traverse_leaves_atomic!(bvh, bvtt1, bvtt2, num_bvtt)

    bvh.stats.num_contacts = num_bvtt

    # Return contact list and the other buffer as possible cache
    BVHTraversal(num_bvtt, bvtt2, bvtt1)
end






@inline function fill_bvh_leaves!(
    bvh_leaves::AbstractVector{BVHLeaf{M, L}},
    bounding_volumes,
) where {M, L}

    # Compute Morton codes and bounding volume indices
    mins, maxs = bounding_volumes_extrema(bounding_volumes)

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = TaskPartitioner(length(bounding_volumes), Threads.nthreads(), 100)

    if tp.num_tasks == 1
        @inbounds fill_bvh_leaves_range!(
            bvh_leaves, bounding_volumes,
            mins, maxs, (1, length(bounding_volumes)),
        )
    else
        tasks = Vector{Task}(undef, tp.num_tasks)
        for i in 1:tp.num_tasks
            @inbounds tasks[i] = Threads.@spawn fill_bvh_leaves_range!(
                bvh_leaves, bounding_volumes,
                mins, maxs, tp[i],
            )
        end
        for i in 1:tp.num_tasks
            wait(tasks[i])
        end
    end

    nothing
end


@inline function fill_bvh_leaves_range!(
    bvh_leaves::AbstractVector{BVHLeaf{M, L}},
    bounding_volumes,
    mins, maxs, irange,
) where {M, L}

    # Save index and Morton code for each bounding volume provided they have a `center` function
    for i in irange[1]:irange[2]
        @inbounds bvh_leaves[i] = BVHLeaf{M, L}(
            morton_encode_single(center(bounding_volumes[i]), mins, maxs, M),
            bounding_volumes[i],
            i,
        )
    end

    nothing
end


# Build OIBVH nodes above the leaf-level from the bottom-up, inplace
function aggregate_oibvh!(bvh_nodes, bvh_leaves, tree, built_level=1)

    # Special case: aggregate level above leaves - might have different node types
    aggregate_last_level!(bvh_nodes, bvh_leaves, tree)

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
    num_nodes_next,
    start_pos,
    irange,
)
    for i in irange[1]:irange[2]
        lchild_index = 2i - 1
        rchild_index = 2i

        # If using different node type than leaf type (e.g. BSphere leaves and BBox nodes) do
        # conversion; this conditional is optimised away at compile-time
        if eltype(bvh_nodes) == bvtype(bvh_leaves)
            # If right child is virtual, set the parent BV to the left child one; otherwise merge
            if rchild_index >= num_nodes_next
                bvh_nodes[start_pos - 1 + i] = bvh_leaves[lchild_index].volume
            else
                bvh_nodes[start_pos - 1 + i] = (bvh_leaves[lchild_index].volume +
                                                bvh_leaves[rchild_index].volume)
            end
        else
            if rchild_index >= num_nodes_next
                bvh_nodes[start_pos - 1 + i] = eltype(bvh_nodes)(bvh_leaves[lchild_index].volume)
            else
                bvh_nodes[start_pos - 1 + i] = eltype(bvh_nodes)(bvh_leaves[lchild_index].volume,
                                                                 bvh_leaves[rchild_index].volume)
            end
        end
    end

    nothing
end


@inline function aggregate_last_level!(bvh_nodes, bvh_leaves, tree)
    # Memory index of first node on this level
    level = tree.levels - 1
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
        @inbounds aggregate_last_level_range!(
            bvh_nodes, bvh_leaves,
            num_nodes_next, start_pos, (1, num_nodes)
        )
    else
        tasks = Vector{Task}(undef, tp.num_tasks)
        for i in 1:tp.num_tasks
            @inbounds tasks[i] = Threads.@spawn aggregate_last_level_range!(
                bvh_nodes, bvh_leaves,
                num_nodes_next, start_pos, tp[i],
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
    for i in irange[1]:irange[2]
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


@inline function initial_bvtt(bvh, start_level, cache)
    # Generate all possible contact checks for the given start_level to avoid the very little
    # work to do at the top
    level_nodes = 2^(start_level - 1)
    level_checks = level_nodes * (level_nodes + 1) ÷ 2

    # If we're not at leaf-level, allocate enough memory for next BVTT expansion
    initial_number = start_level == bvh.tree.levels ? level_checks : 4 * level_checks

    if (
        isnothing(cache) ||
        length(cache.cache1) < initial_number ||
        length(cache.cache2) < initial_number
    )
        bvtt1 = similar(bvh.nodes, IndexPair, initial_number)
        bvtt2 = similar(bvh.nodes, IndexPair, initial_number)
    else
        bvtt1 = cache.cache1
        bvtt2 = cache.cache2
    end

    # Insert all node-node checks - i.e. no self-checks
    num_bvtt = 0
    num_real = level_nodes - bvh.tree.virtual_leaves >> (bvh.tree.levels - start_level)
    for i in level_nodes:level_nodes + num_real - 2
        for j in i + 1:level_nodes + num_real - 1
            num_bvtt += 1
            bvtt1[num_bvtt] = (i, j)
        end
    end

    # Only insert self-checks if we still have nodes below us; leaf-level self-checks aren't needed
    if start_level != bvh.tree.levels
        for i in level_nodes:level_nodes + num_real - 1
            num_bvtt += 1
            bvtt1[num_bvtt] = (i, i)
        end
    end

    bvtt1, bvtt2, num_bvtt
end




@inline function traverse_nodes_atomic_range!(
    bvh, src, dst, num_src, num_dst, self_checks, irange,
)
    # For each BVTT pair of nodes, check for contact
    for i in irange[1]:irange[2]
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[i]

        # If self-check (1, 1), sprout children self-checks (2, 2) (3, 3) and pair children (2, 3)
        if implicit1 == implicit2

            # If the right child is virtual, only add left child self-check
            if @inbounds isvirtual(bvh.tree, 2 * implicit1 + 1)
                if self_checks
                    block_start = Threads.atomic_add!(num_dst, 1)
                    dst[block_start + 1] = (implicit1 * 2, implicit1 * 2)
                end
            else
                if self_checks
                    block_start = Threads.atomic_add!(num_dst, 3)
                    dst[block_start + 1] = (implicit1 * 2, implicit1 * 2)
                    dst[block_start + 2] = (implicit1 * 2 + 1, implicit1 * 2 + 1)
                    dst[block_start + 3] = (implicit1 * 2, implicit1 * 2 + 1)
                else
                    block_start = Threads.atomic_add!(num_dst, 1)
                    dst[block_start + 1] = (implicit1 * 2, implicit1 * 2 + 1)
                end
            end

        # Otherwise pair children of the two nodes
        else
            node1 = @inbounds bvh.nodes[memory_index(bvh.tree, implicit1)]
            node2 = @inbounds bvh.nodes[memory_index(bvh.tree, implicit2)]

            # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
            # the nodes' children
            if iscontact(node1, node2)
                # If the right node's right child is virtual, don't add that check. Guaranteed to
                # always have node1 to the left of node2, hence its children will always be real
                if @inbounds isvirtual(bvh.tree, 2 * implicit2 + 1)
                    block_start = Threads.atomic_add!(num_dst, 2)
                    dst[block_start + 1] = (implicit1 * 2, implicit2 * 2)
                    dst[block_start + 2] = (implicit1 * 2 + 1, implicit2 * 2)
                else
                    block_start = Threads.atomic_add!(num_dst, 4)
                    dst[block_start + 1] = (implicit1 * 2, implicit2 * 2)
                    dst[block_start + 2] = (implicit1 * 2, implicit2 * 2 + 1)
                    dst[block_start + 3] = (implicit1 * 2 + 1, implicit2 * 2)
                    dst[block_start + 4] = (implicit1 * 2 + 1, implicit2 * 2 + 1)
                end
            end
        end
    end

    nothing
end



function traverse_nodes_atomic!(bvh, src, dst, num_src, self_checks=true)
    # Traverse levels above leaves => no contacts, only further BVTT sprouting
    # @show num_src src[1:num_src]

    # Index of current number of pair checks sprouted in `dst`; will be updated atomically by each
    # thread as new blocks of pair checks are added
    num_dst = Threads.Atomic{Int}(0)

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = TaskPartitioner(num_src, Threads.nthreads(), 100)
    if tp.num_tasks == 1
        @inbounds traverse_nodes_atomic_range!(
            bvh, src, dst, num_src, num_dst, self_checks, (1, num_src),
        )
    else
        tasks = Vector{Task}(undef, tp.num_tasks)
        for i in 1:tp.num_tasks
            @inbounds tasks[i] = Threads.@spawn traverse_nodes_atomic_range!(
                bvh, src, dst, num_src, num_dst, self_checks, tp[i],
            )
        end
        for i in 1:tp.num_tasks
            wait(tasks[i])
        end
    end

    num_dst[]
end



@inline function traverse_leaves_atomic_range!(
    bvh, src, contacts, num_src, num_contacts, irange
)
    # Number of indices above leaf-level to subtract from real index
    num_above = bvh.tree.real_nodes - bvh.tree.real_leaves

    # For each BVTT pair of nodes, check for contact
    for i in irange[1]:irange[2]
        # Extract implicit indices of BVH leaves to test
        implicit1, implicit2 = src[i]

        leaf1 = @inbounds bvh.leaves[memory_index(bvh.tree, implicit1) - num_above]
        leaf2 = @inbounds bvh.leaves[memory_index(bvh.tree, implicit2) - num_above]

        # If two leaves are touching, save in contacts
        if iscontact(leaf1.volume, leaf2.volume)
            block_start = Threads.atomic_add!(num_contacts, 1)
            contacts[block_start + 1] = (leaf1.index, leaf2.index)
        end
    end

end


function traverse_leaves_atomic!(bvh, src, contacts, num_src)
    # Traverse final level, only doing leaf-leaf checks
    num_contacts = Threads.Atomic{Int}(0)

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = TaskPartitioner(num_src, Threads.nthreads(), 100)
    if tp.num_tasks == 1
        @inbounds traverse_leaves_atomic_range!(
            bvh, src, contacts, num_src, num_contacts, (1, num_src),
        )
    else
        tasks = Vector{Task}(undef, tp.num_tasks)
        for i in 1:tp.num_tasks
            @inbounds tasks[i] = Threads.@spawn traverse_leaves_atomic_range!(
                bvh, src, contacts, num_src, num_contacts, tp[i],
            )
        end
        for i in 1:tp.num_tasks
            wait(tasks[i])
        end
    end

    num_contacts[]
end
