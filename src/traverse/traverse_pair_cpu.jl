function traverse_nodes_pair!(
    bvh1, bvh2, src, dst, num_src,
    cpu_extra::Tuple{Vector, Vector},
    level1, level2,
    options,
)
    # Traverse nodes when level is above leaves for both BVH1 and BVH2
    tasks, num_written = cpu_extra

    # Compute number of virtual elements before this level to skip when computing the memory index
    virtual_nodes_level1 = bvh1.tree.virtual_leaves >> (bvh1.tree.levels - (level1 - 1))
    virtual_nodes_before1 = 2 * virtual_nodes_level1 - count_ones(virtual_nodes_level1)

    virtual_nodes_level2 = bvh2.tree.virtual_leaves >> (bvh2.tree.levels - (level2 - 1))
    virtual_nodes_before2 = 2 * virtual_nodes_level2 - count_ones(virtual_nodes_level2)

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = AK.TaskPartitioner(num_src, options.num_threads, options.min_traversals_per_thread)
    if tp.num_tasks == 1
        num_dst = traverse_nodes_pair_range!(
            bvh1, bvh2,
            src, dst, nothing,
            virtual_nodes_before1,
            virtual_nodes_before2,
            1:num_src,
        )
    else
        # Keep track of tasks launched and number of elements written by each task in their unique
        # memory region. The unique region is equal to 4 dst elements per src element
        @inbounds for i in 1:tp.num_tasks
            irange = tp[i]
            istart = irange.start
            iend = irange.stop
            tasks[i] = Threads.@spawn traverse_nodes_pair_range!(
                bvh1, bvh2,
                src, view(dst, 4istart - 3:4iend), view(num_written, i),
                virtual_nodes_before1,
                virtual_nodes_before2,
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


function traverse_nodes_pair_range!(
    bvh1, bvh2, src, dst, num_written, num_skips1, num_skips2, irange,
)
    # Check src[irange[1]:irange[2]] and write to dst[1:num_dst]; dst should be given as a view
    num_dst = 0

    # For each BVTT pair of nodes, check for contact
    @inbounds for i in irange
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[i]

        node1 = bvh1.nodes[implicit1 - num_skips1]
        node2 = bvh2.nodes[implicit2 - num_skips2]

        # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
        # the nodes' children
        if iscontact(node1, node2)
            # If a node's right child is virtual, don't add that check. Guaranteed to always have
            # at least one real child

            # BVH1 node's right child is virtual
            if isvirtual(bvh1.tree, 2 * implicit1 + 1)

                # BVH2 node's right child is virtual too
                if isvirtual(bvh2.tree, 2 * implicit2 + 1)
                    dst[num_dst + 1] = _leftleft(implicit1, implicit2)
                    num_dst += 1

                # Only BVH1 node's right child is virtual
                else
                    dst[num_dst + 1] = _leftleft(implicit1, implicit2)
                    dst[num_dst + 2] = _leftright(implicit1, implicit2)
                    num_dst += 2
                end

            # Only BVH2 node's right child is virtual
            elseif isvirtual(bvh2.tree, 2 * implicit2 + 1)
                dst[num_dst + 1] = _leftleft(implicit1, implicit2)
                dst[num_dst + 2] = _rightleft(implicit1, implicit2)
                num_dst += 2

            # All children are real
            else
                dst[num_dst + 1] = _leftleft(implicit1, implicit2)
                dst[num_dst + 2] = _leftright(implicit1, implicit2)
                dst[num_dst + 3] = _rightleft(implicit1, implicit2)
                dst[num_dst + 4] = _rightright(implicit1, implicit2)
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


function traverse_nodes_left!(
    bvh1, bvh2, src, dst, num_src,
    cpu_extra::Tuple{Vector, Vector},
    level1, level2,
    options,
)
    # Traverse nodes when BVH2 is already one above leaf-level - i.e. only BVH1 is sprouted further
    tasks, num_written = cpu_extra

    # Compute number of virtual elements before this level to skip when computing the memory index
    virtual_nodes_level1 = bvh1.tree.virtual_leaves >> (bvh1.tree.levels - (level1 - 1))
    virtual_nodes_before1 = 2 * virtual_nodes_level1 - count_ones(virtual_nodes_level1)

    virtual_nodes_level2 = bvh2.tree.virtual_leaves >> (bvh2.tree.levels - (level2 - 1))
    virtual_nodes_before2 = 2 * virtual_nodes_level2 - count_ones(virtual_nodes_level2)

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = AK.TaskPartitioner(num_src, options.num_threads, options.min_traversals_per_thread)
    if tp.num_tasks == 1
        num_dst = traverse_nodes_left_range!(
            bvh1, bvh2,
            src, dst, nothing,
            virtual_nodes_before1,
            virtual_nodes_before2,
            1:num_src,
        )
    else
        # Keep track of tasks launched and number of elements written by each task in their unique
        # memory region. The unique region is equal to 2 dst elements per src element
        @inbounds for i in 1:tp.num_tasks
            irange = tp[i]
            istart = irange.start
            iend = irange.stop
            tasks[i] = Threads.@spawn traverse_nodes_left_range!(
                bvh1, bvh2,
                src, view(dst, 2istart - 1:2iend), view(num_written, i),
                virtual_nodes_before1,
                virtual_nodes_before2,
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
                    dst[num_dst + j] = dst[2istart - 1 + j - 1]
                end
            end
            num_dst += task_num_written
        end
    end

    num_dst
end


function traverse_nodes_left_range!(
    bvh1, bvh2, src, dst, num_written, num_skips1, num_skips2, irange,
)
    # Check src[irange[1]:irange[2]] and write to dst[1:num_dst]; dst should be given as a view
    num_dst = 0

    # For each BVTT pair of nodes, check for contact. Only expand BVTT for BVH1, as BVH2 is already
    # one above leaf level
    @inbounds for i in irange
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[i]

        node1 = bvh1.nodes[implicit1 - num_skips1]
        node2 = bvh2.nodes[implicit2 - num_skips2]

        # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
        # the nodes' children
        if iscontact(node1, node2)
            # If a node's right child is virtual, don't add that check. Guaranteed to always have
            # at least one real child

            # BVH1 node's right child is virtual
            if isvirtual(bvh1.tree, 2 * implicit1 + 1)
                dst[num_dst + 1] = _leftnoop(implicit1, implicit2)
                num_dst += 1
            else
                dst[num_dst + 1] = _leftnoop(implicit1, implicit2)
                dst[num_dst + 2] = _rightnoop(implicit1, implicit2)
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


function traverse_nodes_right!(
    bvh1, bvh2, src, dst, num_src,
    cpu_extra::Tuple{Vector, Vector},
    level1, level2,
    options,
)
    # Traverse nodes when BVH2 is already one above leaf-level - i.e. only BVH1 is sprouted further
    tasks, num_written = cpu_extra

    # Compute number of virtual elements before this level to skip when computing the memory index
    virtual_nodes_level1 = bvh1.tree.virtual_leaves >> (bvh1.tree.levels - (level1 - 1))
    virtual_nodes_before1 = 2 * virtual_nodes_level1 - count_ones(virtual_nodes_level1)

    virtual_nodes_level2 = bvh2.tree.virtual_leaves >> (bvh2.tree.levels - (level2 - 1))
    virtual_nodes_before2 = 2 * virtual_nodes_level2 - count_ones(virtual_nodes_level2)

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = AK.TaskPartitioner(num_src, options.num_threads, options.min_traversals_per_thread)
    if tp.num_tasks == 1
        num_dst = traverse_nodes_right_range!(
            bvh1, bvh2,
            src, dst, nothing,
            virtual_nodes_before1,
            virtual_nodes_before2,
            1:num_src,
        )
    else
        # Keep track of tasks launched and number of elements written by each task in their unique
        # memory region. The unique region is equal to 2 dst elements per src element
        @inbounds for i in 1:tp.num_tasks
            irange = tp[i]
            istart = irange.start
            iend = irange.stop
            tasks[i] = Threads.@spawn traverse_nodes_right_range!(
                bvh1, bvh2,
                src, view(dst, 2istart - 1:2iend), view(num_written, i),
                virtual_nodes_before1,
                virtual_nodes_before2,
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
                    dst[num_dst + j] = dst[2istart - 1 + j - 1]
                end
            end
            num_dst += task_num_written
        end
    end

    num_dst
end


function traverse_nodes_right_range!(
    bvh1, bvh2, src, dst, num_written,num_skips1, num_skips2, irange,
)
    # Check src[irange[1]:irange[2]] and write to dst[1:num_dst]; dst should be given as a view
    num_dst = 0

    # For each BVTT pair of nodes, check for contact. Only expand BVTT for BVH2, as BVH1 is already
    # one above leaf level
    @inbounds for i in irange
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[i]

        node1 = bvh1.nodes[implicit1 - num_skips1]
        node2 = bvh2.nodes[implicit2 - num_skips2]

        # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
        # the nodes' children
        if iscontact(node1, node2)
            # If a node's right child is virtual, don't add that check. Guaranteed to always have
            # at least one real child

            # BVH2 node's right child is virtual
            if isvirtual(bvh2.tree, 2 * implicit2 + 1)
                dst[num_dst + 1] = _noopleft(implicit1, implicit2)
                num_dst += 1
            else
                dst[num_dst + 1] = _noopleft(implicit1, implicit2)
                dst[num_dst + 2] = _noopright(implicit1, implicit2)
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


function traverse_nodes_leaves_left!(
    bvh1, bvh2, src, dst, num_src,
    cpu_extra::Tuple{Vector, Vector},
    level1, level2,
    options,
)
    # Special case: BVH2 is at leaf level; only BVH1 is sprouted further with node-leaf checks
    tasks, num_written = cpu_extra

    # Compute number of virtual elements before this level to skip when computing the memory index
    virtual_nodes_level1 = bvh1.tree.virtual_leaves >> (bvh1.tree.levels - (level1 - 1))
    virtual_nodes_before1 = 2 * virtual_nodes_level1 - count_ones(virtual_nodes_level1)

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = AK.TaskPartitioner(num_src, options.num_threads, options.min_traversals_per_thread)
    if tp.num_tasks == 1
        num_dst = traverse_nodes_leaves_left_range!(
            bvh1, bvh2,
            src, dst, nothing,
            virtual_nodes_before1,
            1:num_src,
        )
    else
        # Keep track of tasks launched and number of elements written by each task in their unique
        # memory region. The unique region is equal to 2 dst elements per src element
        @inbounds for i in 1:tp.num_tasks
            irange = tp[i]
            istart = irange.start
            iend = irange.stop
            tasks[i] = Threads.@spawn traverse_nodes_leaves_left_range!(
                bvh1, bvh2,
                src, view(dst, 2istart - 1:2iend), view(num_written, i),
                virtual_nodes_before1,
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
                    dst[num_dst + j] = dst[2istart - 1 + j - 1]
                end
            end
            num_dst += task_num_written
        end
    end

    num_dst
end


function traverse_nodes_leaves_left_range!(
    bvh1, bvh2, src, dst, num_written, num_skips1, irange,
)
    # Check src[irange[1]:irange[2]] and write to dst[1:num_dst]; dst should be given as a view
    num_dst = 0

    # Number of implicit indices above leaf-level
    num_above2 = pow2(bvh2.tree.levels - 1) - 1

    # For each BVTT pair of nodes, check for contact. Only expand BVTT for BVH1, as BVH2 is already
    # at leaf level
    @inbounds for i in irange
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[i]

        node1 = bvh1.nodes[implicit1 - num_skips1]

        iorder2 = bvh2.order[implicit2 - num_above2]
        leaf2 = bvh2.leaves[iorder2]

        # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
        # the nodes' children
        if iscontact(node1, leaf2)
            # If a node's right child is virtual, don't add that check. Guaranteed to always have
            # at least one real child

            # BVH1 node's right child is virtual
            if isvirtual(bvh1.tree, 2 * implicit1 + 1)
                dst[num_dst + 1] = _leftnoop(implicit1, implicit2)
                num_dst += 1
            else
                dst[num_dst + 1] = _leftnoop(implicit1, implicit2)
                dst[num_dst + 2] = _rightnoop(implicit1, implicit2)
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


function traverse_nodes_leaves_right!(
    bvh1, bvh2, src, dst, num_src,
    cpu_extra::Tuple{Vector, Vector},
    level1, level2,
    options,
)
    # Special case: BVH1 is at leaf level; only BVH2 is sprouted further with node-leaf checks
    tasks, num_written = cpu_extra

    # Compute number of virtual elements before this level to skip when computing the memory index
    virtual_nodes_level2 = bvh2.tree.virtual_leaves >> (bvh2.tree.levels - (level2 - 1))
    virtual_nodes_before2 = 2 * virtual_nodes_level2 - count_ones(virtual_nodes_level2)

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = AK.TaskPartitioner(num_src, options.num_threads, options.min_traversals_per_thread)
    if tp.num_tasks == 1
        num_dst = traverse_nodes_leaves_right_range!(
            bvh1, bvh2,
            src, dst, nothing,
            virtual_nodes_before2,
            1:num_src,
        )
    else
        # Keep track of tasks launched and number of elements written by each task in their unique
        # memory region. The unique region is equal to 2 dst elements per src element
        @inbounds for i in 1:tp.num_tasks
            irange = tp[i]
            istart = irange.start
            iend = irange.stop
            tasks[i] = Threads.@spawn traverse_nodes_leaves_right_range!(
                bvh1, bvh2,
                src, view(dst, 2istart - 1:2iend), view(num_written, i),
                virtual_nodes_before2,
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
                    dst[num_dst + j] = dst[2istart - 1 + j - 1]
                end
            end
            num_dst += task_num_written
        end
    end

    num_dst
end


function traverse_nodes_leaves_right_range!(
    bvh1, bvh2, src, dst, num_written, num_skips2, irange,
)
    # Check src[irange[1]:irange[2]] and write to dst[1:num_dst]; dst should be given as a view
    num_dst = 0

    # Number of implicit indices above leaf-level
    num_above1 = pow2(bvh1.tree.levels - 1) - 1

    # For each BVTT pair of nodes, check for contact. Only expand BVTT for BVH2, as BVH1 is already
    # at leaf level
    @inbounds for i in irange
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[i]

        iorder1 = bvh1.order[implicit1 - num_above1]
        leaf1 = bvh1.leaves[iorder1]

        node2 = bvh2.nodes[implicit2 - num_skips2]

        # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
        # the nodes' children
        if iscontact(leaf1, node2)
            # If a node's right child is virtual, don't add that check. Guaranteed to always have
            # at least one real child

            # BVH2 node's right child is virtual
            if isvirtual(bvh2.tree, 2 * implicit2 + 1)
                dst[num_dst + 1] = _noopleft(implicit1, implicit2)
                num_dst += 1
            else
                dst[num_dst + 1] = _noopleft(implicit1, implicit2)
                dst[num_dst + 2] = _noopright(implicit1, implicit2)
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


function traverse_leaves_pair!(
    bvh1, bvh2, src, contacts, num_src,
    cpu_extra::Tuple{Vector, Vector},
    options,
)
    # Traverse final level, only doing leaf-leaf checks
    tasks, num_written = cpu_extra

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = AK.TaskPartitioner(num_src, options.num_threads, options.min_traversals_per_thread)
    if tp.num_tasks == 1
        num_contacts = traverse_leaves_pair_range!(
            bvh1, bvh2,
            src, view(contacts, :), nothing,
            1:num_src,
        )
    else
        num_contacts = 0

        # Keep track of tasks launched and number of elements written by each task in their unique
        # memory region. The unique region is equal to 1 dst elements per src element
        @inbounds for i in 1:tp.num_tasks
            irange = tp[i]
            istart = irange.start
            iend = irange.stop
            tasks[i] = Threads.@spawn traverse_leaves_pair_range!(
                bvh1, bvh2,
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


function traverse_leaves_pair_range!(
    bvh1, bvh2, src, contacts, num_written, irange
)
    # Check src[irange[1]:irange[2]] and write to dst[1:num_dst]; dst should be given as a view
    num_dst = 0

    # Number of implicit indices above leaf-level
    num_above1 = pow2(bvh1.tree.levels - 1) - 1
    num_above2 = pow2(bvh2.tree.levels - 1) - 1

    # For each BVTT pair of nodes, check for contact
    @inbounds for i in irange
        # Extract implicit indices of BVH leaves to test
        implicit1, implicit2 = src[i]

        iorder1 = bvh1.order[implicit1 - num_above1]
        iorder2 = bvh2.order[implicit2 - num_above2]

        leaf1 = bvh1.leaves[iorder1]
        leaf2 = bvh2.leaves[iorder2]

        # If two leaves are touching, save in contacts
        if iscontact(leaf1, leaf2)
            contacts[num_dst + 1] = (iorder1, iorder2)
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

