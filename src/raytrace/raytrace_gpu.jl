function traverse_rays_nodes!(bvh, points, directions, src, dst, num_src, dst_offsets, level, options)
    # Traverse nodes when level is above leaves

    # Compute number of virtual elements before this level to skip when computing the memory index
    virtual_nodes_level = bvh.tree.virtual_leaves >> (bvh.tree.levels - (level - 1))
    virtual_nodes_before = 2 * virtual_nodes_level - count_ones(virtual_nodes_level)

    block_size = options.block_size
    num_blocks = (num_src + block_size - 1) ÷ block_size
    backend = get_backend(src)

    kernel! = _traverse_rays_nodes_gpu!(backend, block_size)
    kernel!(
        bvh.tree, bvh.nodes, points, directions,
        src, dst, num_src, dst_offsets, 
        virtual_nodes_before,
        ndrange=num_blocks * block_size,
    )
    
    # We need to know how many checks we have written into dst
    @allowscalar dst_offsets[level]
end


@kernel cpu=false inbounds=true function _traverse_rays_nodes_gpu!(
    tree, nodes, points, directions,
    src, dst, num_src, dst_offsets,
    num_skips,
)
    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear)
    ithread = @index(Local, Linear)

    block_size = @groupsize()[1]

    # At most 2N sprouted checks from N src
    temp = @localmem eltype(dst) (2 * block_size,)
    temp_offset = @localmem typeof(iblock) (1,)
    block_dst_offset = @localmem typeof(iblock) (1,)

    # Write the initial offset for this block as zero. This will be atomically incremented as new
    # pairs are written to temp
    if ithread == 1
        temp_offset[1] = 0
    end
    @synchronize()

    index = ithread + (iblock - 1) * block_size
    if index <= num_src

        # Extract implicit indices of BVH nodes and rays to test
        implicit, iray = src[index]

        node = nodes[implicit - num_skips]
        p = @view points[:, iray]
        d = @view directions[:, iray]

        # If a ray and node are touching, expand BVTT with new possible intersections - i.e. pair
        # the nodes' children with the ray

        if isintersection(node, p, d)
            # If a node's right child is virtual, don't add that check. Guaranteed to always have
            # at least one real child

            # BVH node's right child is virtual
            if unsafe_isvirtual(tree, 2 * implicit + 1)
                new_temp_offset = @atomic temp_offset[1] += 1
                temp[new_temp_offset - 1 + 1] = (implicit * 2, iray)
            # BVH node's right child is real
            else
                new_temp_offset = @atomic temp_offset[1] += 2
                temp[new_temp_offset - 2 + 1] = (implicit * 2, iray)
                temp[new_temp_offset - 2 + 2] = (implicit * 2 + 1, iray)
            end
        end
    end
    @synchronize()

    # Now we have to move the indices from temp to dst in chunks, at offsets reserved via atomic
    # incrementing of dst_offset
    num_temp = temp_offset[1]                               # Number of indices to write
    if ithread == 1
        block_dst_offset[1] = @atomic dst_offsets[end] += num_temp
    end

    @synchronize()
    offset = block_dst_offset[1] - num_temp
    if i <= num_temp
        dst[offset + ithread] = temp[ithread]
    end
end


function traverse_rays_leaves!(
    bvh, points, directions, 
    src::AbstractGPUVector, intersections::AbstractGPUVector,
    num_src, dst_offsets, options
)

    # Traverse final level, only doing ray-leaf checks
    num_above = pow2(bvh.tree.levels - 1) - 1

    block_size = options.block_size
    num_blocks = (num_src + block_size - 1) ÷ block_size
    backend = get_backend(src)

    kernel! = _traverse_rays_leaves!(backend, block_size)
    kernel!(
        bvh.leaves, bvh.order, points, directions,
        src, intersections, num_src, dst_offsets,
        num_above, ndrange=num_blocks * block_size,
    )

    # We need to know how many checks we have written into dst
    @allowscalar dst_offsets[end]
end


@kernel cpu=false inbounds=true function _traverse_rays_leaves_gpu!(
    leaves, order, points, directions,
    src, dst,
    num_src, dst_offsets,
    num_above,
)
    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear)
    ithread = @index(Local, Linear)

    block_size = @groupsize()[1]

    # At most N sprouted checks from N src
    temp = @localmem eltype(dst) (block_size,)
    temp_offset = @localmem typeof(iblock) (1,)
    block_dst_offset = @localmem typeof(iblock) (1,)

    # Write the initial offset for this block as zero. This will be atomically incremented as new
    # pairs are written to temp
    if ithread == 1
        temp_offset[1] = 0
    end
    @synchronize()

    # For each BVTT node-ray pair, check for intersection
    index = ithread + (iblock - 1) * block_size
    if index <= num_src

        # Extract implicit indices of BVH leaves to test
        implicit, iray = src[index]
        iorder = order[implicit - num_above]

        leaf = leaves[iorder]
        p = @view points[:, iray]
        d = @view directions[:, iray]

        # If leaf-ray intersects save the intersection
        if isintersection(leaf, p, d)
            new_temp_offset = @atomic temp_offset[1] += 1
            temp[new_temp_offset - 1 + 1] = (iorder, iray)
        end
    end
    @synchronize()

    # Now we have to move the indices from temp to dst in chunks, at offsets reserved via atomic
    # incrementing of dst_offset
    num_temp = temp_offset[1]                              # Number of indices to write

    if ithread == 1
        block_dst_offset[1] = @atomic dst_offsets[end] += num_temp
    end
    @synchronize()

    offset = block_dst_offset[1] - num_temp
    if ithread <= num_temp
        dst[offset + ithread] = temp[ithread]
    end
end