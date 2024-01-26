"""
    default_start_level(bvh::BVH)::Int
    default_start_level(num_leaves::Integer)::Int

Compute the default start level when traversing a single BVH tree.
"""
function default_start_level(bvh::BVH)
    maximum2(bvh.tree.levels ÷ 2, bvh.built_level)
end


function default_start_level(num_leaves::Integer)
    # Compute the default start level from the number of leaves (geometries) only
    @boundscheck if num_leaves < 1
        throw(DomainError(num_leaves, "must have at least one geometry!"))
    end

    levels = @inbounds ilog2(num_leaves, RoundUp) + 1   # number of binary tree levels
    maximum2(levels ÷ 2, 1)
end


"""
    traverse(
        bvh::BVH,
        start_level::Int=default_start_level(bvh),
        cache::Union{Nothing, BVHTraversal}=nothing;
        num_threads=Threads.nthreads(),
    )::BVHTraversal

Traverse `bvh` downwards from `start_level`, returning all contacting bounding volume leaves. The
returned [`BVHTraversal`](@ref) also contains two contact buffers that can be reused on future
traversals.
    
# Examples

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
traversal = traverse(bvh, 2)

# Reuse traversal buffers for future contact detection - possibly with different BVHs
traversal = traverse(bvh, 2, traversal)
@show traversal.contacts;
;

# output
traversal.contacts = [(1, 2), (2, 3), (4, 5)]
```
"""
function traverse(
    bvh::BVH,
    start_level::Int,
    cache::Union{Nothing, BVHTraversal}=nothing;
    num_threads=Threads.nthreads(),
)

    @assert bvh.tree.levels >= start_level >= bvh.built_level

    # No contacts / traversal for a single node
    if bvh.tree.real_nodes <= 1
        return BVHTraversal(start_level, 0, 0,
                            similar(bvh.nodes, IndexPair, 0),
                            similar(bvh.nodes, IndexPair, 0))
    end

    # Allocate and add all possible BVTT contact pairs to start with
    bvtt1, bvtt2, num_bvtt = initial_bvtt(bvh, start_level, cache)
    num_checks = num_bvtt

    level = start_level
    while level < bvh.tree.levels
        # We can have maximum 4 new checks per contact-pair; resize destination BVTT accordingly
        length(bvtt2) < 4 * num_bvtt && resize!(bvtt2, 4 * 4 * num_bvtt)

        # Check contacts in bvtt1 and add future checks in bvtt2; only sprout self-checks before
        # second-to-last level as leaf self-checks are pointless
        self_checks = level < bvh.tree.levels - 1
        num_bvtt = traverse_nodes!(bvh, bvtt1, bvtt2, num_bvtt, level, self_checks, num_threads)

        num_checks += num_bvtt

        # Swap source and destination buffers for next iteration
        bvtt1, bvtt2 = bvtt2, bvtt1
        level += 1
    end

    # Arrived at final leaf level, now populating contact list
    length(bvtt2) < num_bvtt && resize!(bvtt2, num_bvtt)
    num_bvtt = traverse_leaves!(bvh, bvtt1, bvtt2, num_bvtt, num_threads)

    # Return contact list and the other buffer as possible cache
    BVHTraversal(start_level, num_checks, num_bvtt, bvtt2, bvtt1)
end


# Needed for compiler disambiguation; user interface is the same as for a default argument
function traverse(bvh::BVH)
    traverse(bvh, default_start_level(bvh), nothing)
end


function initial_bvtt(bvh, start_level, cache)
    # Generate all possible contact checks at the given start_level
    level_nodes = pow2(start_level - 1)
    level_checks = (level_nodes - 1) * level_nodes ÷ 2 + level_nodes

    # If we're not at leaf-level, allocate enough memory for next BVTT expansion
    initial_number = start_level == bvh.tree.levels ? level_checks : 4 * level_checks

    # Reuse cache if given
    if isnothing(cache)
        bvtt1 = similar(bvh.nodes, IndexPair, initial_number)
        bvtt2 = similar(bvh.nodes, IndexPair, initial_number)
    else
        bvtt1 = cache.cache1
        bvtt2 = cache.cache2

        length(bvtt1) < initial_number && resize!(bvtt1, initial_number)
        length(bvtt2) < initial_number && resize!(bvtt2, initial_number)
    end

    # Insert all checks at this level
    num_bvtt = 0
    num_real = level_nodes - bvh.tree.virtual_leaves >> (bvh.tree.levels - start_level)
    @inbounds for i in level_nodes:level_nodes + num_real - 1

        # Only insert self-checks if we still have nodes below us; leaf self-checks are not needed
        if start_level != bvh.tree.levels
            num_bvtt += 1
            bvtt1[num_bvtt] = (i, i)
        end

        # Node-node pair checks
        for j in i + 1:level_nodes + num_real - 1
            num_bvtt += 1
            bvtt1[num_bvtt] = (i, j)
        end
    end

    bvtt1, bvtt2, num_bvtt
end


function traverse_nodes_range!(
    bvh, src, dst, num_written, num_skips, self_checks, irange,
)
    # Check src[irange[1]:irange[2]] and write to dst[1:num_dst]; dst should be given as a view
    num_dst = 0

    # For each BVTT pair of nodes, check for contact
    @inbounds for i in irange[1]:irange[2]
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[i]

        # If self-check (1, 1), sprout children self-checks (2, 2) (3, 3) and pair children (2, 3)
        if implicit1 == implicit2

            # If the right child is virtual, only add left child self-check
            if isvirtual(bvh.tree, 2 * implicit1 + 1)
                if self_checks
                    dst[num_dst + 1] = (implicit1 * 2, implicit1 * 2)
                    num_dst += 1
                end
            else
                if self_checks
                    dst[num_dst + 1] = (implicit1 * 2, implicit1 * 2)
                    dst[num_dst + 2] = (implicit1 * 2, implicit1 * 2 + 1)
                    dst[num_dst + 3] = (implicit1 * 2 + 1, implicit1 * 2 + 1)
                    num_dst += 3
                else
                    dst[num_dst + 1] = (implicit1 * 2, implicit1 * 2 + 1)
                    num_dst += 1
                end
            end

        # Otherwise pair children of the two nodes
        else
            node1 = bvh.nodes[implicit1 - num_skips]
            node2 = bvh.nodes[implicit2 - num_skips]

            # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
            # the nodes' children
            if iscontact(node1, node2)
                # If the right node's right child is virtual, don't add that check. Guaranteed to
                # always have node1 to the left of node2, hence its children will always be real
                if isvirtual(bvh.tree, 2 * implicit2 + 1)
                    dst[num_dst + 1] = (implicit1 * 2, implicit2 * 2)
                    dst[num_dst + 2] = (implicit1 * 2 + 1, implicit2 * 2)
                    num_dst += 2
                else
                    dst[num_dst + 1] = (implicit1 * 2, implicit2 * 2)
                    dst[num_dst + 2] = (implicit1 * 2, implicit2 * 2 + 1)
                    dst[num_dst + 3] = (implicit1 * 2 + 1, implicit2 * 2)
                    dst[num_dst + 4] = (implicit1 * 2 + 1, implicit2 * 2 + 1)
                    num_dst += 4
                end
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



function traverse_nodes!(bvh, src, dst, num_src, level, self_checks, num_threads)
    # Traverse levels above leaves => no contacts, only further BVTT sprouting

    # Compute number of virtual elements before this level to skip when computing the memory index
    virtual_nodes_level = bvh.tree.virtual_leaves >> (bvh.tree.levels - (level - 1))
    virtual_nodes_before = 2 * virtual_nodes_level - count_ones(virtual_nodes_level)

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = TaskPartitioner(num_src, num_threads, 100)
    if tp.num_tasks == 1
        num_dst = traverse_nodes_range!(
            bvh,
            src, dst, nothing,
            virtual_nodes_before,
            self_checks,
            (1, num_src),
        )
    else
        # Keep track of tasks launched and number of elements written by each task in their unique
        # memory region. The unique region is equal to 4 dst elements per src element
        tasks = Vector{Task}(undef, tp.num_tasks)
        num_written = Vector{Int}(undef, tp.num_tasks)
        @inbounds for i in 1:tp.num_tasks
            istart, iend = tp[i]
            tasks[i] = Threads.@spawn traverse_nodes_range!(
                bvh,
                src, view(dst, 4istart - 3:4iend), view(num_written, i),
                virtual_nodes_before,
                self_checks,
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



function traverse_leaves_range!(
    bvh, src, contacts, num_written, irange
)
    # Check src[irange[1]:irange[2]] and write to dst[1:num_dst]; dst should be given as a view
    num_dst = 0

    # Number of implicit indices above leaf-level
    num_above = pow2(bvh.tree.levels - 1) - 1

    # For each BVTT pair of nodes, check for contact
    @inbounds for i in irange[1]:irange[2]

        # Extract implicit indices of BVH leaves to test
        implicit1, implicit2 = src[i]

        iorder1 = bvh.order[implicit1 - num_above]
        iorder2 = bvh.order[implicit2 - num_above]

        leaf1 = bvh.leaves[iorder1]
        leaf2 = bvh.leaves[iorder2]

        # If two leaves are touching, save in contacts
        if iscontact(leaf1, leaf2)

            # While it's guaranteed that implicit1 < implicit2, the bvh.order may not be
            # ascending, so we add this comparison to output ordered contact indices
            contacts[num_dst + 1] = iorder1 < iorder2 ? (iorder1, iorder2) : (iorder2, iorder1)
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


function traverse_leaves!(bvh, src, contacts, num_src, num_threads)
    # Traverse final level, only doing leaf-leaf checks

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = TaskPartitioner(num_src, num_threads, 100)
    if tp.num_tasks == 1
        num_contacts = traverse_leaves_range!(
            bvh,
            src, view(contacts, :), nothing,
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
            tasks[i] = Threads.@spawn traverse_leaves_range!(
                bvh,
                src, view(contacts, istart:iend), view(num_written, i),
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
