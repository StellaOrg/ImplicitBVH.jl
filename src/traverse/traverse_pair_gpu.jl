function traverse_nodes_pair!(bvh1, bvh2, src::AbstractGPUVector, dst::AbstractGPUVector,
                              num_src, dst_offsets, level1, level2, options)
    # Traverse nodes when level is above leaves for both BVH1 and BVH2

    # Compute number of virtual elements before this level to skip when computing the memory index
    virtual_nodes_level1 = bvh1.tree.virtual_leaves >> (bvh1.tree.levels - (level1 - 1))
    virtual_nodes_before1 = 2 * virtual_nodes_level1 - count_ones(virtual_nodes_level1)

    virtual_nodes_level2 = bvh2.tree.virtual_leaves >> (bvh2.tree.levels - (level2 - 1))
    virtual_nodes_before2 = 2 * virtual_nodes_level2 - count_ones(virtual_nodes_level2)

    block_size = options.block_size
    num_blocks = (num_src + block_size - 1) ÷ block_size
    backend = get_backend(src)

    # Compute which dst offset this kernel will use
    idst_offsets = level2 + (level1 - 1) * bvh2.tree.levels

    kernel! = _traverse_nodes_pair_gpu!(backend, block_size)
    kernel!(
        bvh1.tree, bvh2.tree,
        bvh1.nodes, bvh2.nodes,
        src, dst, num_src, dst_offsets, idst_offsets,
        virtual_nodes_before1, virtual_nodes_before2,
        ndrange=num_blocks * block_size,
    )

    # We need to know how many checks we have written into dst
    @allowscalar dst_offsets[idst_offsets]
end


@kernel cpu=false inbounds=true function _traverse_nodes_pair_gpu!(
    tree1, tree2,
    @Const(nodes1), @Const(nodes2),
    @Const(src), dst, num_src, dst_offsets, idst_offsets,
    num_skips1, num_skips2,
)
    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear)
    ithread = @index(Local, Linear)

    block_size = @groupsize()[1]

    # At most 4N sprouted checks from N src
    temp = @localmem eltype(dst) (4 * block_size,)
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
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[index]

        node1 = nodes1[implicit1 - num_skips1]
        node2 = nodes2[implicit2 - num_skips2]

        # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
        # the nodes' children
        if iscontact(node1, node2)
            # If a node's right child is virtual, don't add that check. Guaranteed to always have
            # at least one real child

            # BVH1 node's right child is virtual
            if unsafe_isvirtual(tree1, 2 * implicit1 + 1)

                # BVH2 node's right child is virtual too
                if unsafe_isvirtual(tree2, 2 * implicit2 + 1)
                    new_temp_offset = @atomic temp_offset[1] += 1
                    temp[new_temp_offset - 1 + 1] = (implicit1 * 2, implicit2 * 2)

                # Only BVH1 node's right child is virtual
                else
                    new_temp_offset = @atomic temp_offset[1] += 2
                    temp[new_temp_offset - 2 + 1] = (implicit1 * 2, implicit2 * 2)
                    temp[new_temp_offset - 2 + 2] = (implicit1 * 2, implicit2 * 2 + 1)
                end

            # Only BVH2 node's right child is virtual
            elseif unsafe_isvirtual(tree2, 2 * implicit2 + 1)
                new_temp_offset = @atomic temp_offset[1] += 2
                temp[new_temp_offset - 2 + 1] = (implicit1 * 2, implicit2 * 2)
                temp[new_temp_offset - 2 + 2] = (implicit1 * 2 + 1, implicit2 * 2)

            # All children are real
            else
                new_temp_offset = @atomic temp_offset[1] += 4
                temp[new_temp_offset - 4 + 1] = (implicit1 * 2, implicit2 * 2)
                temp[new_temp_offset - 4 + 2] = (implicit1 * 2, implicit2 * 2 + 1)
                temp[new_temp_offset - 4 + 3] = (implicit1 * 2 + 1, implicit2 * 2)
                temp[new_temp_offset - 4 + 4] = (implicit1 * 2 + 1, implicit2 * 2 + 1)
            end
        end
    end
    @synchronize()

    # Now we have to move the indices from temp to dst in chunks, at offsets reserved via atomic
    # incrementing of dst_offset
    num_temp = temp_offset[1]                               # Number of indices to write
    if ithread == 1
        block_dst_offset[1] = @atomic dst_offsets[idst_offsets] += num_temp
    end
    @synchronize()

    offset = block_dst_offset[1] - num_temp
    i = ithread
    while i <= num_temp
        dst[offset + i] = temp[i]
        i += block_size
    end
end


function traverse_nodes_left!(bvh1, bvh2, src::AbstractGPUVector, dst::AbstractGPUVector,
                              num_src, dst_offsets, level1, level2, options)
    # Traverse nodes when BVH2 is already one above leaf-level - i.e. only BVH1 is sprouted further

    # Compute number of virtual elements before this level to skip when computing the memory index
    virtual_nodes_level1 = bvh1.tree.virtual_leaves >> (bvh1.tree.levels - (level1 - 1))
    virtual_nodes_before1 = 2 * virtual_nodes_level1 - count_ones(virtual_nodes_level1)

    virtual_nodes_level2 = bvh2.tree.virtual_leaves >> (bvh2.tree.levels - (level2 - 1))
    virtual_nodes_before2 = 2 * virtual_nodes_level2 - count_ones(virtual_nodes_level2)

    block_size = options.block_size
    num_blocks = (num_src + block_size - 1) ÷ block_size
    backend = get_backend(src)

    # Compute which dst offset this kernel will use
    idst_offsets = level2 + (level1 - 1) * bvh2.tree.levels

    kernel! = _traverse_nodes_left_gpu!(backend, block_size)
    kernel!(
        bvh1.tree, bvh2.tree,
        bvh1.nodes, bvh2.nodes,
        src, dst, num_src, dst_offsets, idst_offsets,
        virtual_nodes_before1, virtual_nodes_before2,
        ndrange=num_blocks * block_size,
    )

    # We need to know how many checks we have written into dst
    @allowscalar dst_offsets[idst_offsets]
end


@kernel cpu=false inbounds=true function _traverse_nodes_left_gpu!(
    tree1, tree2,
    @Const(nodes1), @Const(nodes2),
    @Const(src), dst, num_src, dst_offsets, idst_offsets,
    num_skips1, num_skips2,
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

    # For each BVTT pair of nodes, check for contact. Only expand BVTT for BVH1, as BVH2 is already
    # one above leaf level
    index = ithread + (iblock - 1) * block_size
    if index <= num_src
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[index]

        node1 = nodes1[implicit1 - num_skips1]
        node2 = nodes2[implicit2 - num_skips2]

        # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
        # the nodes' children
        if iscontact(node1, node2)
            # If a node's right child is virtual, don't add that check. Guaranteed to always have
            # at least one real child

            # BVH1 node's right child is virtual
            if unsafe_isvirtual(tree1, 2 * implicit1 + 1)
                new_temp_offset = @atomic temp_offset[1] += 1
                temp[new_temp_offset - 1 + 1] = (implicit1 * 2, implicit2)
            else
                new_temp_offset = @atomic temp_offset[1] += 2
                temp[new_temp_offset - 2 + 1] = (implicit1 * 2, implicit2)
                temp[new_temp_offset - 2 + 2] = (implicit1 * 2 + 1, implicit2)
            end
        end
    end
    @synchronize()

    # Now we have to move the indices from temp to dst in chunks, at offsets reserved via atomic
    # incrementing of dst_offset
    num_temp = temp_offset[1]                               # Number of indices to write
    if ithread == 1
        block_dst_offset[1] = @atomic dst_offsets[idst_offsets] += num_temp
    end
    @synchronize()

    offset = block_dst_offset[1] - num_temp
    i = ithread
    while i <= num_temp
        dst[offset + i] = temp[i]
        i += block_size
    end
end


function traverse_nodes_right!(bvh1, bvh2, src::AbstractGPUVector, dst::AbstractGPUVector,
                               num_src, dst_offsets, level1, level2, options)
    # Traverse nodes when BVH2 is already one above leaf-level - i.e. only BVH1 is sprouted further

    # Compute number of virtual elements before this level to skip when computing the memory index
    virtual_nodes_level1 = bvh1.tree.virtual_leaves >> (bvh1.tree.levels - (level1 - 1))
    virtual_nodes_before1 = 2 * virtual_nodes_level1 - count_ones(virtual_nodes_level1)

    virtual_nodes_level2 = bvh2.tree.virtual_leaves >> (bvh2.tree.levels - (level2 - 1))
    virtual_nodes_before2 = 2 * virtual_nodes_level2 - count_ones(virtual_nodes_level2)

    block_size = options.block_size
    num_blocks = (num_src + block_size - 1) ÷ block_size
    backend = get_backend(src)

    # Compute which dst offset this kernel will use
    idst_offsets = level2 + (level1 - 1) * bvh2.tree.levels

    kernel! = _traverse_nodes_right_gpu!(backend, block_size)
    kernel!(
        bvh1.tree, bvh2.tree,
        bvh1.nodes, bvh2.nodes,
        src, dst, num_src, dst_offsets, idst_offsets,
        virtual_nodes_before1, virtual_nodes_before2,
        ndrange=num_blocks * block_size,
    )

    # We need to know how many checks we have written into dst
    @allowscalar dst_offsets[idst_offsets]
end


@kernel cpu=false inbounds=true function _traverse_nodes_right_gpu!(
    tree1, tree2,
    @Const(nodes1), @Const(nodes2),
    @Const(src), dst, num_src, dst_offsets, idst_offsets,
    num_skips1, num_skips2,
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

    # For each BVTT pair of nodes, check for contact. Only expand BVTT for BVH2, as BVH1 is already
    # one above leaf level
    index = ithread + (iblock - 1) * block_size
    if index <= num_src
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[index]

        node1 = nodes1[implicit1 - num_skips1]
        node2 = nodes2[implicit2 - num_skips2]

        # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
        # the nodes' children
        if iscontact(node1, node2)
            # If a node's right child is virtual, don't add that check. Guaranteed to always have
            # at least one real child

            # BVH2 node's right child is virtual
            if unsafe_isvirtual(tree2, 2 * implicit2 + 1)
                new_temp_offset = @atomic temp_offset[1] += 1
                temp[new_temp_offset - 1 + 1] = (implicit1, implicit2 * 2)
            else
                new_temp_offset = @atomic temp_offset[1] += 2
                temp[new_temp_offset - 2 + 1] = (implicit1, implicit2 * 2)
                temp[new_temp_offset - 2 + 2] = (implicit1, implicit2 * 2 + 1)
            end
        end
    end
    @synchronize()

    # Now we have to move the indices from temp to dst in chunks, at offsets reserved via atomic
    # incrementing of dst_offset
    num_temp = temp_offset[1]                               # Number of indices to write
    if ithread == 1
        block_dst_offset[1] = @atomic dst_offsets[idst_offsets] += num_temp
    end
    @synchronize()

    offset = block_dst_offset[1] - num_temp
    i = ithread
    while i <= num_temp
        dst[offset + i] = temp[i]
        i += block_size
    end
end


function traverse_nodes_leaves_left!(bvh1, bvh2, src::AbstractGPUVector, dst::AbstractGPUVector,
                                     num_src, dst_offsets, level1, level2, options)
    # Special case: BVH2 is at leaf level; only BVH1 is sprouted further with node-leaf checks

    # Compute number of virtual elements before this level to skip when computing the memory index
    virtual_nodes_level1 = bvh1.tree.virtual_leaves >> (bvh1.tree.levels - (level1 - 1))
    virtual_nodes_before1 = 2 * virtual_nodes_level1 - count_ones(virtual_nodes_level1)

    num_above2 = pow2(bvh2.tree.levels - 1) - 1

    block_size = options.block_size
    num_blocks = (num_src + block_size - 1) ÷ block_size
    backend = get_backend(src)

    # Compute which dst offset this kernel will use
    idst_offsets = level2 + (level1 - 1) * bvh2.tree.levels

    kernel! = _traverse_nodes_leaves_left_gpu!(backend, block_size)
    kernel!(
        bvh1.tree, bvh2.tree,
        bvh1.nodes, bvh2.leaves, bvh2.order,
        src, dst, num_src, dst_offsets, idst_offsets,
        virtual_nodes_before1, num_above2,
        ndrange=num_blocks * block_size,
    )

    # We need to know how many checks we have written into dst
    @allowscalar dst_offsets[idst_offsets]
end


@kernel cpu=false inbounds=true function _traverse_nodes_leaves_left_gpu!(
    tree1, tree2,
    @Const(nodes1), @Const(leaves2), @Const(order2),
    @Const(src), dst, num_src, dst_offsets, idst_offsets,
    num_skips1, num_above2,
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

    # For each BVTT pair of nodes, check for contact. Only expand BVTT for BVH1, as BVH2 is already
    # at leaf level
    index = ithread + (iblock - 1) * block_size
    if index <= num_src
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[index]

        node1 = nodes1[implicit1 - num_skips1]

        iorder2 = order2[implicit2 - num_above2]
        leaf2 = leaves2[iorder2]

        # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
        # the nodes' children
        if iscontact(node1, leaf2)
            # If a node's right child is virtual, don't add that check. Guaranteed to always have
            # at least one real child

            # BVH1 node's right child is virtual
            if unsafe_isvirtual(tree1, 2 * implicit1 + 1)
                new_temp_offset = @atomic temp_offset[1] += 1
                temp[new_temp_offset - 1 + 1] = (implicit1 * 2, implicit2)
            else
                new_temp_offset = @atomic temp_offset[1] += 2
                temp[new_temp_offset - 2 + 1] = (implicit1 * 2, implicit2)
                temp[new_temp_offset - 2 + 2] = (implicit1 * 2 + 1, implicit2)
            end
        end
    end
    @synchronize()

    # Now we have to move the indices from temp to dst in chunks, at offsets reserved via atomic
    # incrementing of dst_offset
    num_temp = temp_offset[1]                               # Number of indices to write
    if ithread == 1
        block_dst_offset[1] = @atomic dst_offsets[idst_offsets] += num_temp
    end
    @synchronize()

    offset = block_dst_offset[1] - num_temp
    i = ithread
    while i <= num_temp
        dst[offset + i] = temp[i]
        i += block_size
    end
end


function traverse_nodes_leaves_right!(bvh1, bvh2, src::AbstractGPUVector, dst::AbstractGPUVector,
                                      num_src, dst_offsets, level1, level2, options)
    # Special case: BVH1 is at leaf level; only BVH2 is sprouted further with node-leaf checks

    # Compute number of virtual elements before this level to skip when computing the memory index
    num_above1 = pow2(bvh1.tree.levels - 1) - 1

    virtual_nodes_level2 = bvh2.tree.virtual_leaves >> (bvh2.tree.levels - (level2 - 1))
    virtual_nodes_before2 = 2 * virtual_nodes_level2 - count_ones(virtual_nodes_level2)

    block_size = options.block_size
    num_blocks = (num_src + block_size - 1) ÷ block_size
    backend = get_backend(src)

    # Compute which dst offset this kernel will use
    idst_offsets = level2 + (level1 - 1) * bvh2.tree.levels

    kernel! = _traverse_nodes_leaves_right_gpu!(backend, block_size)
    kernel!(
        bvh1.tree, bvh2.tree,
        bvh1.leaves, bvh2.nodes, bvh1.order,
        src, dst, num_src, dst_offsets, idst_offsets,
        num_above1, virtual_nodes_before2,
        ndrange=num_blocks * block_size,
    )

    # We need to know how many checks we have written into dst
    @allowscalar dst_offsets[idst_offsets]
end


@kernel cpu=false inbounds=true function _traverse_nodes_leaves_right_gpu!(
    tree1, tree2,
    @Const(leaves1), @Const(nodes2), @Const(order1),
    @Const(src), dst, num_src, dst_offsets, idst_offsets,
    num_above1, num_skips2,
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

    # For each BVTT pair of nodes, check for contact. Only expand BVTT for BVH2, as BVH1 is already
    # at leaf level
    index = ithread + (iblock - 1) * block_size
    if index <= num_src
        # Extract implicit indices of BVH nodes to test
        implicit1, implicit2 = src[index]

        iorder1 = order1[implicit1 - num_above1]
        leaf1 = leaves1[iorder1]

        node2 = nodes2[implicit2 - num_skips2]

        # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
        # the nodes' children
        if iscontact(leaf1, node2)
            # If a node's right child is virtual, don't add that check. Guaranteed to always have
            # at least one real child

            # BVH2 node's right child is virtual
            if unsafe_isvirtual(tree2, 2 * implicit2 + 1)
                new_temp_offset = @atomic temp_offset[1] += 1
                temp[new_temp_offset - 1 + 1] = (implicit1, implicit2 * 2)
            else
                new_temp_offset = @atomic temp_offset[1] += 2
                temp[new_temp_offset - 2 + 1] = (implicit1, implicit2 * 2)
                temp[new_temp_offset - 2 + 2] = (implicit1, implicit2 * 2 + 1)
            end
        end
    end
    @synchronize()

    # Now we have to move the indices from temp to dst in chunks, at offsets reserved via atomic
    # incrementing of dst_offset
    num_temp = temp_offset[1]                               # Number of indices to write
    if ithread == 1
        block_dst_offset[1] = @atomic dst_offsets[idst_offsets] += num_temp
    end
    @synchronize()

    offset = block_dst_offset[1] - num_temp
    i = ithread
    while i <= num_temp
        dst[offset + i] = temp[i]
        i += block_size
    end
end


function traverse_leaves_pair!(bvh1, bvh2, src::AbstractGPUVector, contacts::AbstractGPUVector,
                               num_src, dst_offsets, options)
    # Traverse final level, only doing leaf-leaf checks
    num_above1 = pow2(bvh1.tree.levels - 1) - 1
    num_above2 = pow2(bvh2.tree.levels - 1) - 1

    block_size = options.block_size
    num_blocks = (num_src + block_size - 1) ÷ block_size
    backend = get_backend(src)

    kernel! = _traverse_leaves_pair_gpu!(backend, block_size)
    kernel!(
        bvh1.leaves, bvh2.leaves, bvh1.order, bvh2.order,
        src, contacts, num_src, dst_offsets,
        num_above1, num_above2,
        ndrange=num_blocks * block_size,
    )

    # We need to know how many checks we have written into dst
    @allowscalar dst_offsets[end]
end


@kernel cpu=false inbounds=true function _traverse_leaves_pair_gpu!(
    @Const(leaves1), @Const(leaves2), @Const(order1), @Const(order2),
    @Const(src), dst,
    num_src, dst_offsets,
    num_above1, num_above2,
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

    # For each BVTT pair of nodes, check for contact
    index = ithread + (iblock - 1) * block_size
    if index <= num_src
        # Extract implicit indices of BVH leaves to test
        implicit1, implicit2 = src[index]

        iorder1 = order1[implicit1 - num_above1]
        iorder2 = order2[implicit2 - num_above2]

        leaf1 = leaves1[iorder1]
        leaf2 = leaves2[iorder2]

        # If two leaves are touching, save in contacts
        if iscontact(leaf1, leaf2)
            new_temp_offset = @atomic temp_offset[1] += 1
            temp[new_temp_offset - 1 + 1] = (iorder1, iorder2)
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
    if ithread <= num_temp
        dst[offset + ithread] = temp[ithread]
    end
end
