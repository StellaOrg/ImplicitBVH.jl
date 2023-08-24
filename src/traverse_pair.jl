"""
    default_start_level(bvh1::BVH, bvh2::BVH)::Int

Compute the default start level when traversing two BVH trees of potentially different sizes.
"""
function default_start_level(bvh1::BVH, bvh2::BVH)
    m1 = maximum2(bvh1.built_level, bvh1.tree.levels ÷ 2)
    m2 = maximum2(bvh2.built_level, bvh2.tree.levels ÷ 2)
    minimum2(m1, m2)
end


"""
    traverse(
        bvh1::BVH,
        bvh2::BVH,
        start_level::Int=default_start_level(bvh1, bvh2),
        cache::Union{Nothing, BVHTraversal}=nothing,
    )::BVHTraversal

Return all the `bvh1` bounding volume leaves that are in contact with any in `bvh2`. The returned
[`BVHTraversal`](@ref) also contains two contact buffers that can be reused on future traversals.

# Examples

```jldoctest
using ImplicitBVH
using ImplicitBVH: BBox, BSphere

# Generate some simple bounding spheres
bounding_spheres1 = [
    BSphere{Float32}([0., 0., 0.], 0.5),
    BSphere{Float32}([0., 0., 3.], 0.4),
]

bounding_spheres2 = [
    BSphere{Float32}([0., 0., 1.], 0.6),
    BSphere{Float32}([0., 0., 2.], 0.5),
    BSphere{Float32}([0., 0., 4.], 0.6),
]

# Build BVHs
bvh1 = BVH(bounding_spheres1, BBox{Float32}, UInt32)
bvh2 = BVH(bounding_spheres2, BBox{Float32}, UInt32)

# Traverse BVH for contact detection
traversal = traverse(bvh1, bvh2, default_start_level(bvh1, bvh2))

# Reuse traversal buffers for future contact detection - possibly with different BVHs
traversal = traverse(bvh1, bvh2, default_start_level(bvh1, bvh2), traversal)
@show traversal.contacts;
;

# output
traversal.contacts = [(1, 1), (2, 3)]
```
"""
function traverse(
    bvh1::BVH{V1, V2, V3},
    bvh2::BVH{V1, V2, V3},
    start_level::Int,
    cache::Union{Nothing, BVHTraversal}=nothing,
) where {V1, V2, V3}

    @boundscheck begin
        if bvh1.tree.levels == bvh2.tree.levels
            @assert bvh1.tree.levels >= start_level >= bvh1.built_level
            @assert bvh2.tree.levels >= start_level >= bvh2.built_level
        else
            @assert bvh1.tree.levels > start_level >= bvh1.built_level
            @assert bvh2.tree.levels > start_level >= bvh2.built_level
        end
    end

    # Always have bigger bvh1.tree.levels > bvh2.tree.levels on the left
    if bvh1.tree.levels < bvh2.tree.levels
        bvh1, bvh2 = bvh2, bvh1
        swapped = true
    else
        swapped = false
    end

    # Explanation: say BVH1 has 10 levels, BVH2 has 8 levels; the last level has "leaves" (the
    # actual bounding volumes for contact detection), levels above have "nodes". Both BVHs are
    # aligned to start at level 1. We have two buffers, bvtt1 (src) and bvtt2 (dst); bvtt1 stores
    # the current level's pairs of BVs we need to check for contact. For each contacting pair of
    # BVs in bvtt1, we pair their children for checking contacts at the next level, which we write
    # into bvtt2. Then bvtt1 and bvtt2 are swapped, we advance to the next level, and repeat.
    #
    # A complicating factor is the fact that the two BVHs may have different heights. We then split
    # contact detection into 4 stages: first, we traverse both in sync - that is, at level 1 we have
    # in bvtt1=[(1, 1)] and we write into bvtt2=[(2, 2), (2, 3), (3, 2), (3, 3)] (if the root of
    # BVH1 is in contact with the root of BVH2). We continue at level 2, 3... until we reach level
    # 7, the level above BVH2 leaves. Now we only traverse BVH1, keeping BVH2's level fixed, and
    # doing node-node checks until we reach level 9 in BVH1; now both BVHs have reached the level
    # above leaves. We do another in-sync pairwise contact detection, this time having nodes in
    # bvtt1 (src) and writing possible contacts to check for leaves in bvtt2 (dst). Now we have
    # reached the leaf-level of both BVHs and we write all contacts found into the final contacts
    # vector.

    # Allocate and add all possible BVTT contact pairs to start with
    bvtt1, bvtt2, num_bvtt = initial_bvtt(bvh1, bvh2, start_level, cache)
    num_checks = num_bvtt

    # Compute node-node contacts while both BVHs are at node levels
    level = start_level
    while level < bvh2.tree.levels - 1
        # We can have maximum 4 new checks per contact-pair; resize destination BVTT accordingly
        length(bvtt2) < 4 * num_bvtt && resize!(bvtt2, 4 * num_bvtt)

        # Check contacts in bvtt1 and add future checks in bvtt2
        num_bvtt = traverse_nodes_pair!(bvh1, bvh2, bvtt1, bvtt2, num_bvtt)
        num_checks += num_bvtt

        # Swap source and destination buffers for next iteration
        bvtt1, bvtt2 = bvtt2, bvtt1
        level += 1
    end

    # Compute node-node contacts while right BVH is at level above leaves
    while level < bvh1.tree.levels - 1
        # We can have maximum 2 new checks per contact-pair; resize destination BVTT accordingly
        length(bvtt2) < 2 * num_bvtt && resize!(bvtt2, 2 * num_bvtt)

        # Check contacts in bvtt1 and add future checks in bvtt2
        num_bvtt = traverse_nodes_left!(bvh1, bvh2, bvtt1, bvtt2, num_bvtt)
        num_checks += num_bvtt

        # Swap source and destination buffers for next iteration
        bvtt1, bvtt2 = bvtt2, bvtt1
        level += 1
    end

    # Compute node-node contacts when both BVHs are at level above leaves
    if level < bvh1.tree.levels
        # We can have maximum 4 new checks per contact-pair; resize destination BVTT accordingly
        length(bvtt2) < 4 * num_bvtt && resize!(bvtt2, 4 * num_bvtt)

        # Check contacts in bvtt1 and add future checks in bvtt2
        num_bvtt = traverse_nodes_pair!(bvh1, bvh2, bvtt1, bvtt2, num_bvtt)
        num_checks += num_bvtt

        # Swap source and destination buffers for next iteration
        bvtt1, bvtt2 = bvtt2, bvtt1
        level += 1
    end

    # Arrived at final leaf level with both BVHs, now populating contact list
    length(bvtt2) < num_bvtt && resize!(bvtt2, num_bvtt)
    num_bvtt = traverse_leaves_pair!(bvh1, bvh2, bvtt1, bvtt2, num_bvtt, swapped)

    # Return contact list and the other buffer as possible cache
    BVHTraversal(start_level, num_checks, num_bvtt, bvtt2, bvtt1)
end


# Needed for compiler disambiguation; user interface is the same as for a default argument
function traverse(bvh1::BVH{V1, V2, V3}, bvh2::BVH{V1, V2, V3}) where {V1, V2, V3}
    traverse(bvh1, bvh2, default_start_level(bvh1, bvh2), nothing)
end


function initial_bvtt(bvh1, bvh2, start_level, cache)
    # Generate all possible contact checks at the given start_level
    level_nodes = pow2(start_level - 1)

    # Number of real nodes at the given start_level and number of checks we'll do
    num_real1 = level_nodes - bvh1.tree.virtual_leaves >> (bvh1.tree.levels - start_level)
    num_real2 = level_nodes - bvh2.tree.virtual_leaves >> (bvh2.tree.levels - start_level)
    level_checks = num_real1 * num_real2

    # If we're not at leaf-level, allocate enough memory for next BVTT expansion
    if bvh2.tree.levels == start_level
        if bvh1.tree.levels > bvh2.tree.levels
            initial_number = 2 * level_checks           # Only bvh2 at leaf level
        else
            initial_number = level_checks               # Both at leaf level
        end
    else
        initial_number = 4 * level_checks               # Neither at leaf level
    end

    # Reuse cache if given
    if isnothing(cache)
        bvtt1 = similar(bvh1.nodes, IndexPair, initial_number)
        bvtt2 = similar(bvh1.nodes, IndexPair, initial_number)
    else
        bvtt1 = cache.cache1
        bvtt2 = cache.cache2

        length(bvtt1) < initial_number && resize!(bvtt1, initial_number)
        length(bvtt2) < initial_number && resize!(bvtt2, initial_number)
    end

    # Insert all checks at this level
    num_bvtt = 0
    @inbounds for i in level_nodes:level_nodes + num_real1 - 1

        # Node-node pair checks
        for j in level_nodes:level_nodes + num_real2 - 1
            num_bvtt += 1
            bvtt1[num_bvtt] = (i, j)
        end
    end

    bvtt1, bvtt2, num_bvtt
end


function traverse_nodes_pair!(bvh1, bvh2, src, dst, num_src)
    # Traverse nodes when level is above leaves for both BVH1 and BVH2

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = TaskPartitioner(num_src, Threads.nthreads(), 100)
    if tp.num_tasks == 1
        num_dst = traverse_nodes_pair_range!(
            bvh1, bvh2,
            src, dst, nothing,
            (1, num_src),
        )
    else
        # Keep track of tasks launched and number of elements written by each task in their unique
        # memory region. The unique region is equal to 4 dst elements per src element
        tasks = Vector{Task}(undef, tp.num_tasks)
        num_written = Vector{Int}(undef, tp.num_tasks)
        @inbounds for i in 1:tp.num_tasks
            istart, iend = tp[i]
            tasks[i] = Threads.@spawn traverse_nodes_pair_range!(
                bvh1, bvh2,
                src, view(dst, 4istart - 3:4iend), view(num_written, i),
                (istart, iend),
            )
        end

        # As tasks finish sequentially, move the new written contacts into contiguous region
        num_dst = 0
        @inbounds for i in 1:tp.num_tasks
            wait(tasks[i])
            task_num_written = num_written[i]

            # Repack written contacts by the second, third thread, etc.
            if i > 1
                istart, iend = tp[i]
                for j in 1:task_num_written
                    dst[num_dst + j] = dst[4istart - 3 + j - 1]
                end
            end
            num_dst += task_num_written
        end
    end

    num_dst
end


function traverse_nodes_pair_range!(
    bvh1, bvh2, src, dst, num_written, irange,
)
    # Check src[irange[1]:irange[2]] and write to dst[1:num_dst]; dst should be given as a view
    num_dst = 0

    # For each BVTT pair of nodes, check for contact
    @inbounds for i in irange[1]:irange[2]
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[i]

        # TODO: can we avoid calling memory_index every time? the number of virtual nodes we
        # need to skip is constant for each level
        node1 = bvh1.nodes[memory_index(bvh1.tree, implicit1)]
        node2 = bvh2.nodes[memory_index(bvh2.tree, implicit2)]

        # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
        # the nodes' children
        if iscontact(node1, node2)
            # If a node's right child is virtual, don't add that check. Guaranteed to always have
            # at least one real child

            # BVH1 node's right child is virtual
            if isvirtual(bvh1.tree, 2 * implicit1 + 1)

                # BVH2 node's right child is virtual too
                if isvirtual(bvh2.tree, 2 * implicit2 + 1)
                    dst[num_dst + 1] = (implicit1 * 2, implicit2 * 2)
                    num_dst += 1

                # Only BVH1 node's right child is virtual
                else
                    dst[num_dst + 1] = (implicit1 * 2, implicit2 * 2)
                    dst[num_dst + 2] = (implicit1 * 2, implicit2 * 2 + 1)
                    num_dst += 2
                end

            # Only BVH2 node's right child is virtual
            elseif isvirtual(bvh2.tree, 2 * implicit2 + 1)
                dst[num_dst + 1] = (implicit1 * 2, implicit2 * 2)
                dst[num_dst + 2] = (implicit1 * 2 + 1, implicit2 * 2)
                num_dst += 2

            # All children are real
            else
                dst[num_dst + 1] = (implicit1 * 2, implicit2 * 2)
                dst[num_dst + 2] = (implicit1 * 2, implicit2 * 2 + 1)
                dst[num_dst + 3] = (implicit1 * 2 + 1, implicit2 * 2)
                dst[num_dst + 4] = (implicit1 * 2 + 1, implicit2 * 2 + 1)
                num_dst += 4
            end
        end
    end

    # Known at compile-time; no return if called in multithreaded context
    if isnothing(num_written)
        return num_dst
    else
        num_written[] = num_dst
        return nothing
    end
end


function traverse_nodes_left!(bvh1, bvh2, src, dst, num_src)
    # Traverse nodes when BVH2 is already at leaf-level - i.e. only BVH1 is sprouted further

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = TaskPartitioner(num_src, Threads.nthreads(), 100)
    if tp.num_tasks == 1
        num_dst = traverse_nodes_left_range!(
            bvh1, bvh2,
            src, dst, nothing,
            (1, num_src),
        )
    else
        # Keep track of tasks launched and number of elements written by each task in their unique
        # memory region. The unique region is equal to 2 dst elements per src element
        tasks = Vector{Task}(undef, tp.num_tasks)
        num_written = Vector{Int}(undef, tp.num_tasks)
        @inbounds for i in 1:tp.num_tasks
            istart, iend = tp[i]
            tasks[i] = Threads.@spawn traverse_nodes_left_range!(
                bvh1, bvh2,
                src, view(dst, 2istart - 1:2iend), view(num_written, i),
                (istart, iend),
            )
        end

        # As tasks finish sequentially, move the new written contacts into contiguous region
        num_dst = 0
        @inbounds for i in 1:tp.num_tasks
            wait(tasks[i])
            task_num_written = num_written[i]

            # Repack written contacts by the second, third thread, etc.
            if i > 1
                istart, iend = tp[i]
                for j in 1:task_num_written
                    dst[num_dst + j] = dst[2istart - 1 + j - 1]
                end
            end
            num_dst += task_num_written
        end
    end

    num_dst
end


function traverse_nodes_left_range!(
    bvh1, bvh2, src, dst, num_written, irange,
)
    # Check src[irange[1]:irange[2]] and write to dst[1:num_dst]; dst should be given as a view
    num_dst = 0

    # For each BVTT pair of nodes, check for contact. Only expand BVTT for BVH1, as BVH2 is already
    # at leaf level
    @inbounds for i in irange[1]:irange[2]
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[i]

        # TODO: can we avoid calling memory_index every time? the number of virtual nodes we
        # need to skip is constant for each level
        node1 = bvh1.nodes[memory_index(bvh1.tree, implicit1)]
        node2 = bvh2.nodes[memory_index(bvh2.tree, implicit2)]

        # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
        # the nodes' children
        if iscontact(node1, node2)
            # If a node's right child is virtual, don't add that check. Guaranteed to always have
            # at least one real child

            # BVH1 node's right child is virtual
            if isvirtual(bvh1.tree, 2 * implicit1 + 1)
                dst[num_dst + 1] = (implicit1 * 2, implicit2)
                num_dst += 1
            else
                dst[num_dst + 1] = (implicit1 * 2, implicit2)
                dst[num_dst + 2] = (implicit1 * 2 + 1, implicit2)
                num_dst += 2
            end
        end
    end

    # Known at compile-time; no return if called in multithreaded context
    if isnothing(num_written)
        return num_dst
    else
        num_written[] = num_dst
        return nothing
    end
end


function traverse_leaves_pair!(bvh1, bvh2, src, contacts, num_src, swapped)
    # Traverse final level, only doing leaf-leaf checks

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = TaskPartitioner(num_src, Threads.nthreads(), 100)
    if tp.num_tasks == 1
        num_contacts = traverse_leaves_pair_range!(
            bvh1, bvh2,
            src, view(contacts, :), nothing,
            swapped,
            (1, num_src),
        )
    else
        num_contacts = 0

        # Keep track of tasks launched and number of elements written by each task in their unique
        # memory region. The unique region is equal to 1 dst elements per src element
        tasks = Vector{Task}(undef, tp.num_tasks)
        num_written = Vector{Int}(undef, tp.num_tasks)
        @inbounds for i in 1:tp.num_tasks
            istart, iend = tp[i]
            tasks[i] = Threads.@spawn traverse_leaves_pair_range!(
                bvh1, bvh2,
                src, view(contacts, istart:iend), view(num_written, i),
                swapped,
                (istart, iend),
            )
        end
        @inbounds for i in 1:tp.num_tasks
            wait(tasks[i])
            task_num_written = num_written[i]

            # Repack written contacts by the second, third thread, etc.
            if i > 1
                istart, iend = tp[i]
                for j in 1:task_num_written
                    contacts[num_contacts + j] = contacts[istart + j - 1]
                end
            end
            num_contacts += task_num_written
        end
    end

    num_contacts
end


function traverse_leaves_pair_range!(
    bvh1, bvh2, src, contacts, num_written, swapped, irange
)
    # Check src[irange[1]:irange[2]] and write to dst[1:num_dst]; dst should be given as a view
    num_dst = 0

    # Number of implicit indices above leaf-level
    num_above1 = pow2(bvh1.tree.levels - 1) - 1
    num_above2 = pow2(bvh2.tree.levels - 1) - 1

    # For each BVTT pair of nodes, check for contact
    @inbounds for i in irange[1]:irange[2]
        # Extract implicit indices of BVH leaves to test
        implicit1, implicit2 = src[i]

        iorder1 = bvh1.order[implicit1 - num_above1]
        iorder2 = bvh2.order[implicit2 - num_above2]

        leaf1 = bvh1.leaves[iorder1]
        leaf2 = bvh2.leaves[iorder2]

        # If two leaves are touching, save in contacts
        if iscontact(leaf1, leaf2)
            contacts[num_dst + 1] = swapped ? (iorder2, iorder1) : (iorder1, iorder2)
            num_dst += 1
        end
    end

    # Known at compile-time; no return if called in multithreaded context
    if isnothing(num_written)
        return num_dst
    else
        num_written[] = num_dst
        return nothing
    end
end
