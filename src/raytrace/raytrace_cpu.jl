function traverse_rays_nodes!(bvh, points, directions, src, dst, num_src, ::Nothing, level, options)
    # Traverse nodes when level is above leaves

    # Compute number of virtual elements before this level to skip when computing the memory index
    virtual_nodes_level = bvh.tree.virtual_leaves >> (bvh.tree.levels - (level - 1))
    virtual_nodes_before = 2 * virtual_nodes_level - count_ones(virtual_nodes_level)

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = AK.TaskPartitioner(num_src, options.num_threads, options.min_traversals_per_thread)
    if tp.num_tasks == 1
        num_dst = traverse_rays_nodes_range!(
            bvh, points, directions,
            src, dst, nothing,
            virtual_nodes_before,
            (1, num_src),
        )
    else
        # Keep track of tasks launched and number of elements written by each task in their unique
        # memory region. The unique region is equal to 2 dst elements per src element
        tasks = Vector{Task}(undef, tp.num_tasks)
        num_written = Vector{Int}(undef, tp.num_tasks)
        @inbounds for i in 1:tp.num_tasks
            istart, iend = tp[i]
            tasks[i] = Threads.@spawn traverse_rays_nodes_range!(
                bvh, points, directions,
                src, view(dst, 2istart - 1:2iend), view(num_written, i),
                virtual_nodes_before,
                (istart, iend),
            )
        end

        # As tasks finish sequentially, move the new written intersctions into contiguous region
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


function traverse_rays_nodes_range!(
    bvh, points, directions, src, dst, num_written, num_skips, irange,
)
    # Check src[irange[1]:irange[2]] and write to dst[1:num_dst]; dst should be given as a view
    num_dst = 0

    # For each BVTT node-ray pair, check for intersection
    @inbounds for i in irange[1]:irange[2]
        # Extract implicit indices of BVH nodes to test
        implicit, iray = src[i]
        node = bvh.nodes[implicit - num_skips]

        # Extract ray
        p = @view points[:, iray]
        d = @view directions[:, iray]

        # If the node and ray is touching, expand BVTT with new possible contacts - i.e. pair
        if isintersection(node, p, d)
            # If a node's right child is virtual, don't add that check. Guaranteed to always have
            # at least one real child

            # BVH node's right child is virtual
            if isvirtual(bvh.tree, 2 * implicit + 1)
                dst[num_dst + 1] = (implicit * 2, iray)
                num_dst += 1
            else
                dst[num_dst + 1] = (implicit * 2, iray)
                dst[num_dst + 2] = (implicit * 2 + 1, iray)
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


function traverse_rays_leaves!(bvh, points, directions, src, intersections, num_src, ::Nothing, options)
    # Traverse final level, only doing ray-leaf checks

    # Split computation into contiguous ranges of minimum 100 elements each; if only single thread
    # is needed, inline call
    tp = TaskPartitioner(num_src, options.num_threads, options.min_traversals_per_thread)
    if tp.num_tasks == 1
        num_intersections = traverse_rays_leaves_range!(
            bvh, points, directions,
            src, intersections, nothing,
            (1, num_src),
        )
    else
        num_intersections = 0

        # Keep track of tasks launched and number of elements written by each task in their unique
        # memory region. The unique region is equal to 1 dst elements per src element
        tasks = Vector{Task}(undef, tp.num_tasks)
        num_written = Vector{Int}(undef, tp.num_tasks)
        @inbounds for i in 1:tp.num_tasks
            istart, iend = tp[i]
            tasks[i] = Threads.@spawn traverse_rays_leaves_range!(
                bvh, points, directions,
                src, view(intersections, istart:iend), view(num_written, i),
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
                    intersections[num_intersections + j] = intersections[istart + j - 1]
                end
            end
            num_intersections += task_num_written
        end
    end

    num_intersections
end


function traverse_rays_leaves_range!(
    bvh, points, directions, src, intersections, num_written, irange
)
    # Check src[irange[1]:irange[2]] and write to dst[1:num_dst]; dst should be given as a view
    num_dst = 0

    # Number of implicit indices above leaf-level
    num_above = pow2(bvh.tree.levels - 1) - 1

    # For each BVTT node-ray pair, check for intersection
    @inbounds for i in irange[1]:irange[2]
        # Extract implicit indices of BVH leaves to test
        implicit, iray = src[i]

        iorder = bvh.order[implicit - num_above]
        leaf = bvh.leaves[iorder]

        p = @view points[:, iray]
        d = @view directions[:, iray]

        # If leaf-ray intersection, save in intersections
        if isintersection(leaf, p, d)
            intersections[num_dst + 1] = (iorder, iray)
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
