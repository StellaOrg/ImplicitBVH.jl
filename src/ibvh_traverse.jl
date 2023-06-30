"""
    $(TYPEDEF)

Alias for a tuple of two indices representing e.g. a contacting pair.
"""
const IndexPair = Tuple{Int, Int}


"""
    $(TYPEDEF)

Collected BVH traversal `contacts` list, plus the two buffers `cache1` and `cache2` which can be
reused for future traversals to minimise memory allocations.
"""
@with_kw struct BVHTraversal{VC <: AbstractVector}
    num_contacts::Int
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
using IBVH: BBox, BSphere
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


@inline function initial_bvtt(bvh, start_level, cache)
    # Generate all possible contact checks for the given start_level to avoid the very little
    # work to do at the top
    level_nodes = 2^(start_level - 1)
    level_checks = level_nodes * (level_nodes + 1) ÷ 2

    # If we're not at leaf-level, allocate enough memory for next BVTT expansion
    initial_number = start_level == bvh.tree.levels ? level_checks : 4 * level_checks

    if isnothing(cache)
        bvtt1 = similar(bvh.nodes, IndexPair, initial_number)
        bvtt2 = similar(bvh.nodes, IndexPair, initial_number)
    else
        bvtt1 = cache.cache1
        bvtt2 = cache.cache2

        length(bvtt1) < initial_number && resize!(bvtt1, initial_number)
        length(bvtt2) < initial_number && resize!(bvtt2, initial_number)
    end

    # Insert all node-node checks - i.e. no self-checks
    num_bvtt = 0
    num_real = level_nodes - bvh.tree.virtual_leaves >> (bvh.tree.levels - start_level)
    @inbounds for i in level_nodes:level_nodes + num_real - 2
        for j in i + 1:level_nodes + num_real - 1
            num_bvtt += 1
            bvtt1[num_bvtt] = (i, j)
        end
    end

    # Only insert self-checks if we still have nodes below us; leaf-level self-checks aren't needed
    if start_level != bvh.tree.levels
        @inbounds for i in level_nodes:level_nodes + num_real - 1
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
    @inbounds for i in irange[1]:irange[2]
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[i]

        # If self-check (1, 1), sprout children self-checks (2, 2) (3, 3) and pair children (2, 3)
        if implicit1 == implicit2

            # If the right child is virtual, only add left child self-check
            if isvirtual(bvh.tree, 2 * implicit1 + 1)
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
            node1 = bvh.nodes[memory_index(bvh.tree, implicit1)]
            node2 = bvh.nodes[memory_index(bvh.tree, implicit2)]

            # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
            # the nodes' children
            if iscontact(node1, node2)
                # If the right node's right child is virtual, don't add that check. Guaranteed to
                # always have node1 to the left of node2, hence its children will always be real
                if isvirtual(bvh.tree, 2 * implicit2 + 1)
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
        traverse_nodes_atomic_range!(
            bvh, src, dst, num_src, num_dst, self_checks, (1, num_src),
        )
    else
        tasks = Vector{Task}(undef, tp.num_tasks)
        @inbounds for i in 1:tp.num_tasks
            tasks[i] = Threads.@spawn traverse_nodes_atomic_range!(
                bvh, src, dst, num_src, num_dst, self_checks, tp[i],
            )
        end
        @inbounds for i in 1:tp.num_tasks
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
    @inbounds for i in irange[1]:irange[2]
        # Extract implicit indices of BVH leaves to test
        implicit1, implicit2 = src[i]

        real1 = bvh.order[memory_index(bvh.tree, implicit1) - num_above]
        real2 = bvh.order[memory_index(bvh.tree, implicit2) - num_above]

        leaf1 = bvh.leaves[real1]
        leaf2 = bvh.leaves[real2]

        # If two leaves are touching, save in contacts
        if iscontact(leaf1, leaf2)
            block_start = Threads.atomic_add!(num_contacts, 1)
            contacts[block_start + 1] = real1 < real2 ? (real1, real2) : (real2, real1)
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
        traverse_leaves_atomic_range!(
            bvh, src, contacts, num_src, num_contacts, (1, num_src),
        )
    else
        tasks = Vector{Task}(undef, tp.num_tasks)
        @inbounds for i in 1:tp.num_tasks
            tasks[i] = Threads.@spawn traverse_leaves_atomic_range!(
                bvh, src, contacts, num_src, num_contacts, tp[i],
            )
        end
        @inbounds for i in 1:tp.num_tasks
            wait(tasks[i])
        end
    end

    num_contacts[]
end
