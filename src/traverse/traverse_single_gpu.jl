function traverse_nodes!(bvh, src::AbstractGPUVector, dst::AbstractGPUVector,
                         num_src, dst_offsets, level, self_checks, options)
    # Traverse levels above leaves => no contacts, only further BVTT sprouting
    I = get_index_type(options)

    # Compute number of virtual elements before this level to skip when computing the memory index
    virtual_nodes_level = bvh.tree.virtual_leaves >> (bvh.tree.levels - (level - 1))
    virtual_nodes_before = 2 * virtual_nodes_level - count_ones(virtual_nodes_level)

    block_size = options.block_size
    num_blocks = (num_src + block_size - 1) ÷ block_size
    backend = get_backend(src)

    kernel! = _traverse_nodes_gpu!(backend, block_size)
    kernel!(
        bvh.tree, bvh.nodes,
        src, dst,
        I(num_src), dst_offsets, I(level),
        I(virtual_nodes_before), self_checks,
        ndrange=num_blocks * block_size,
    )

    # We need to know how many checks we have written into dst
    src, dst, @allowscalar(dst_offsets[level])
end




@kernel cpu=false inbounds=true unsafe_indices=true function _traverse_nodes_gpu!(
    tree, @Const(nodes),
    @Const(src), dst,
    num_src, dst_offsets, level,
    num_skips, self_checks,
)
    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear)
    ithread = @index(Local, Linear)

    @uniform block_size = @groupsize()[0x1]

    # At most 4N sprouted checks from N src
    temp = @localmem eltype(dst) (0x4 * block_size,)
    temp_offset = @localmem typeof(iblock) (1,)
    block_dst_offset = @localmem typeof(iblock) (1,)

    # Write the initial offset for this block as zero. This will be atomically incremented as new
    # pairs are written to temp
    if ithread == 0x1
        temp_offset[0x1] = 0x0
    end
    @synchronize()

    index = ithread + (iblock - 0x1) * block_size
    if index <= num_src
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[index]

        # If self-check (1, 1), sprout children self-checks (2, 2) (3, 3) and pair children (2, 3)
        if implicit1 == implicit2

            # If the right child is virtual, only add left child self-check
            if unsafe_isvirtual(tree, 0x2 * implicit1 + 0x1)
                if self_checks
                    new_temp_offset = @atomic temp_offset[0x1] += 0x1
                    temp[new_temp_offset - 0x1 + 0x1] = _leftleft(implicit1, implicit1)
                end
            else
                if self_checks
                    new_temp_offset = @atomic temp_offset[0x1] += 0x3
                    temp[new_temp_offset - 0x3 + 0x1] = _leftleft(implicit1, implicit1)
                    temp[new_temp_offset - 0x3 + 0x2] = _leftright(implicit1, implicit1)
                    temp[new_temp_offset - 0x3 + 0x3] = _rightright(implicit1, implicit1)
                else
                    new_temp_offset = @atomic temp_offset[0x1] += 0x1
                    temp[new_temp_offset - 0x1 + 0x1] = _leftright(implicit1, implicit1)
                end
            end

        # Otherwise pair children of the two nodes
        else
            node1 = nodes[implicit1 - num_skips]
            node2 = nodes[implicit2 - num_skips]

            # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
            # the nodes' children
            if iscontact(node1, node2)
                # If the right node's right child is virtual, don't add that check. Guaranteed to
                # always have node1 to the left of node2, hence its children will always be real
                if unsafe_isvirtual(tree, 0x2 * implicit2 + 0x1)
                    new_temp_offset = @atomic temp_offset[0x1] += 0x2
                    temp[new_temp_offset - 0x2 + 0x1] = _leftleft(implicit1, implicit2)
                    temp[new_temp_offset - 0x2 + 0x2] = _rightleft(implicit1, implicit2)
                else
                    new_temp_offset = @atomic temp_offset[0x1] += 0x4
                    temp[new_temp_offset - 0x4 + 0x1] = _leftleft(implicit1, implicit2)
                    temp[new_temp_offset - 0x4 + 0x2] = _leftright(implicit1, implicit2)
                    temp[new_temp_offset - 0x4 + 0x3] = _rightleft(implicit1, implicit2)
                    temp[new_temp_offset - 0x4 + 0x4] = _rightright(implicit1, implicit2)
                end
            end
        end
    end
    @synchronize()

    # Now we have to move the indices from temp to dst in chunks, at offsets reserved via atomic
    # incrementing of dst_offset
    num_temp = temp_offset[0x1]                             # Number of indices to write
    if ithread == 0x1
        block_dst_offset[0x1] = @atomic dst_offsets[level] += num_temp
    end
    @synchronize()

    offset = block_dst_offset[0x1] - num_temp
    i = ithread
    while i <= num_temp
        dst[offset + i] = temp[i]
        i += block_size
    end
end




function traverse_leaves!(bvh, src::AbstractGPUVector, contacts::AbstractGPUVector,
                          num_src, dst_offsets, options)
    # Traverse final level, only doing leaf-leaf checks
    I = get_index_type(options)

    # Number of implicit indices above leaf-level
    num_skips = pow2(bvh.tree.levels - 1) - 1

    block_size = options.block_size
    num_blocks = (num_src + block_size - 1) ÷ block_size
    backend = get_backend(src)

    kernel! = _traverse_leaves_gpu!(backend, block_size)
    kernel!(
        bvh.leaves, bvh.order,
        src, contacts,
        I(num_src), dst_offsets, I(num_skips),
        ndrange=num_blocks * block_size,
    )

    # We need to know how many pairs we have written into contacts
    src, contacts, @allowscalar(dst_offsets[end])
end




@kernel cpu=false inbounds=true unsafe_indices=true function _traverse_leaves_gpu!(
    @Const(leaves), @Const(order),
    @Const(src), dst,
    num_src, dst_offsets,
    num_skips,
)
    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear)
    ithread = @index(Local, Linear)

    @uniform block_size = @groupsize()[0x1]

    # At most N contacts in dst from N src
    temp = @localmem eltype(dst) (block_size,)
    temp_offset = @localmem typeof(iblock) (1,)
    block_dst_offset = @localmem typeof(iblock) (1,)

    # Write the initial offset for this block as zero. This will be atomically incremented as new
    # pairs are written to temp
    if ithread == 0x1
        temp_offset[0x1] = 0x0
    end
    @synchronize()

    index = ithread + (iblock - 0x1) * block_size
    if index <= num_src

        # Extract implicit indices of BVH leaves to test
        implicit1, implicit2 = src[index]

        iorder1 = order[implicit1 - num_skips]
        iorder2 = order[implicit2 - num_skips]

        leaf1 = leaves[iorder1]
        leaf2 = leaves[iorder2]

        # If two leaves are touching, save in contacts
        if iscontact(leaf1, leaf2)

            # While it's guaranteed that implicit1 < implicit2, the bvh.order may not be
            # ascending, so we add this comparison to output ordered contact indices
            new_temp_offset = @atomic temp_offset[0x1] += 0x1
            temp[new_temp_offset - 0x1 + 0x1] = iorder1 < iorder2 ? (iorder1, iorder2) : (iorder2, iorder1)
        end
    end
    @synchronize()

    # Now we have to move the indices from temp to dst in chunks, at offsets reserved via atomic
    # incrementing of dst_offset
    num_temp = temp_offset[0x1]                             # Number of indices to write

    if ithread == 0x1
        block_dst_offset[0x1] = @atomic dst_offsets[end] += num_temp
    end
    @synchronize()

    offset = block_dst_offset[0x1] - num_temp
    if ithread <= num_temp
        dst[offset + ithread] = temp[ithread]
    end
end
