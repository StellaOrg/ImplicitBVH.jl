function traverse_nodes!(bvh, src, dst, num_src, ::Nothing, level, self_checks, options)
    # Traverse levels above leaves => no contacts, only further BVTT sprouting

    # Compute number of virtual elements before this level to skip when computing the memory index
    virtual_nodes_level = bvh.tree.virtual_leaves >> (bvh.tree.levels - (level - 1))
    virtual_nodes_before = 2 * virtual_nodes_level - count_ones(virtual_nodes_level)

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = AK.TaskPartitioner(num_src, options.num_threads, options.min_traversals_per_thread)
    if tp.num_tasks == 1
        num_dst = traverse_nodes_range!(
            bvh,
            src, dst, nothing,
            virtual_nodes_before,
            self_checks,
            1:num_src,
        )
    else
        # Keep track of tasks launched and number of elements written by each task in their unique
        # memory region. The unique region is equal to 4 dst elements per src element
        tasks = Vector{Task}(undef, tp.num_tasks)
        num_written = Vector{Int}(undef, tp.num_tasks)
        @inbounds for i in 1:tp.num_tasks
            irange = tp[i]
            istart = irange.start
            iend = irange.stop
            tasks[i] = Threads.@spawn traverse_nodes_range!(
                bvh,
                src, view(dst, 4istart - 3:4iend), view(num_written, i),
                virtual_nodes_before,
                self_checks,
                irange,
            )
        end

        # As tasks finish sequentially, move the new written contacts into contiguous region
        num_dst = 0
        @inbounds for i in 1:tp.num_tasks
            wait(tasks[i])
            task_num_written = num_written[i]

            # Repack written contacts by the second, third thread, etc.
            if i > 1
                istart = tp[i].start
                for j in 1:task_num_written
                    dst[num_dst + j] = dst[4istart - 3 + j - 1]
                end
            end
            num_dst += task_num_written
        end
    end

    num_dst
end



function traverse_nodes_range!(
    bvh, src, dst, num_written, num_skips, self_checks, irange,
)
    # Check src[irange[1]:irange[2]] and write to dst[1:num_dst]; dst should be given as a view
    num_dst = 0

    # For each BVTT pair of nodes, check for contact
    @inbounds for i in irange
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[i]

        # If self-check (1, 1), sprout children self-checks (2, 2) (3, 3) and pair children (2, 3)
        if implicit1 == implicit2

            # If the right child is virtual, only add left child self-check
            if unsafe_isvirtual(bvh.tree, 2 * implicit1 + 1)
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
                if unsafe_isvirtual(bvh.tree, 2 * implicit2 + 1)
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



function traverse_leaves!(bvh, src, contacts, num_src, ::Nothing, options)
    # Traverse final level, only doing leaf-leaf checks

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = AK.TaskPartitioner(num_src, options.num_threads, options.min_traversals_per_thread)
    if tp.num_tasks == 1
        num_contacts = traverse_leaves_range!(
            bvh,
            src, view(contacts, :), nothing,
            1:num_src,
        )
    else
        num_contacts = 0

        # Keep track of tasks launched and number of elements written by each task in their unique
        # memory region. The unique region is equal to 1 dst elements per src element
        tasks = Vector{Task}(undef, tp.num_tasks)
        num_written = Vector{Int}(undef, tp.num_tasks)
        @inbounds for i in 1:tp.num_tasks
            irange = tp[i]
            istart = irange.start
            iend = irange.stop
            tasks[i] = Threads.@spawn traverse_leaves_range!(
                bvh,
                src, view(contacts, istart:iend), view(num_written, i),
                irange,
            )
        end
        @inbounds for i in 1:tp.num_tasks
            wait(tasks[i])
            task_num_written = num_written[i]

            # Repack written contacts by the second, third thread, etc.
            if i > 1
                istart = tp[i].start
                for j in 1:task_num_written
                    contacts[num_contacts + j] = contacts[istart + j - 1]
                end
            end
            num_contacts += task_num_written
        end
    end

    num_contacts
end



function traverse_leaves_range!(
    bvh, src, contacts, num_written, irange
)
    # Check src[irange[1]:irange[2]] and write to dst[1:num_dst]; dst should be given as a view
    num_dst = 0

    # Number of implicit indices above leaf-level
    num_above = pow2(bvh.tree.levels - 1) - 1

    # For each BVTT pair of nodes, check for contact
    @inbounds for i in irange

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
