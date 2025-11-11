function traverse(
    bvh1::BVH, bvh2::BVH, alg::LVTTraversal;
    start_level1::Int=default_start_level(bvh1, alg),
    start_level2::Int=default_start_level(bvh2, alg),
    narrow=(bv1, bv2) -> true,
    cache::Union{Nothing, BVHTraversal}=nothing,
    options=BVHOptions(),
)
    # Correctness checks
    @boundscheck begin
        @argcheck bvh1.built_level <= start_level1 <= bvh1.tree.levels <= 32
        @argcheck bvh2.built_level <= start_level2 <= bvh2.tree.levels <= 32
    end

    # Delegate to the BVH with more leaves to be the first argument
    if length(bvh1.leaves) >= length(bvh2.leaves)
        return traverse_lvt(
            bvh1, bvh2, LVTTraversal();
            start_level1=start_level1,
            start_level2=start_level2,
            narrow=narrow,
            cache=cache,
            flip=Val(false),
            options=options,
        )
    else
        return traverse_lvt(
            bvh2, bvh1, LVTTraversal();
            start_level1=start_level2,
            start_level2=start_level1,
            narrow=narrow,
            cache=cache,
            flip=Val(true),
            options=options,
        )
    end
end


function traverse_lvt(
    bvh1::BVH, bvh2::BVH, ::LVTTraversal;
    start_level1::Int=default_start_level(bvh1),
    start_level2::Int=default_start_level(bvh2),
    narrow=(bv1, bv2) -> true,
    cache::Union{Nothing, BVHTraversal}=nothing,
    flip=Val(false),
    options=BVHOptions(),
)
    # Get index type from exemplar; compile-time safety checks
    I = get_index_type(bvh1)
    @argcheck get_index_type(bvh2) === I
    @argcheck AK.get_backend(bvh1.nodes) === AK.get_backend(bvh2.nodes)

    # Allocate buffer for number of contacts to be written, one for each thread
    if AK.get_backend(bvh1.nodes) isa typeof(AK.CPU_BACKEND)
        # CPU backend - one per thread (one thread handles multiple input bounding volumes)
        npadding = div(64, sizeof(I), RoundUp)
        thread_ncontacts_length = npadding * options.num_threads
    else
        # GPU backend - one per input bounding volume
        npadding = 1
        thread_ncontacts_length = length(bvh1.leaves)
    end

    # Reuse from cache if possible
    if isnothing(cache)
        thread_ncontacts_raw = similar(bvh1.nodes, I, thread_ncontacts_length)
    else
        @argcheck eltype(cache.cache2) === I
        thread_ncontacts_raw = cache.cache2
        if length(thread_ncontacts_raw) < thread_ncontacts_length
            resize!(thread_ncontacts_raw, thread_ncontacts_length)
        end
    end

    # On CPUs take every npadding-th element to avoid false sharing
    if AK.get_backend(bvh1.nodes) isa typeof(AK.CPU_BACKEND)
        thread_ncontacts = view(reshape(thread_ncontacts_raw, npadding, :), 1, :)
    else
        thread_ncontacts = thread_ncontacts_raw
    end

    # On the first pass, just count how many contacts we will have per thread
    thread_ncontacts .= I(0)
    traverse_contacts_lvt!(bvh1, bvh2, start_level2, thread_ncontacts, nothing, narrow, flip, options)

    # Accumulate numbers of contacts to get thread offsets to write contacts based on
    AK.accumulate!(+, thread_ncontacts, init=I(0), max_tasks=1, block_size=options.block_size)

    # Allocate contacts vector
    total_contacts = @allowscalar thread_ncontacts[end]
    if isnothing(cache)
        contacts = similar(bvh1.nodes, IndexPair{I}, total_contacts)
    else
        @argcheck eltype(cache.cache1) === IndexPair{I}
        contacts = cache.cache1
        length(contacts) < total_contacts && resize!(contacts, total_contacts)
    end

    if total_contacts == 0
        # No contacts, return empty traversal
        return BVHTraversal(
            Int(start_level1), Int(start_level2), 0,
            contacts, thread_ncontacts_raw,
        )
    end

    # Do second pass where contacts are written
    traverse_contacts_lvt!(bvh1, bvh2, start_level2, thread_ncontacts, contacts, narrow, flip, options)

    # Return contact list and the other buffer as possible cache
    BVHTraversal(
        Int(start_level1), Int(start_level2), Int(total_contacts),
        contacts, thread_ncontacts_raw,
    )
end


function traverse_contacts_lvt!(
    bvh1::BVH, bvh2::BVH,
    start_level2::Int,
    thread_ncontacts::AbstractVector,
    contacts::Union{Nothing, AbstractVector{<:IndexPair}},      # Nothing on first (counting) pass
    narrow,
    flip::Val,
    options::BVHOptions,
)
    backend = AK.get_backend(bvh1.nodes)
    num_checks = length(bvh1.leaves)

    if backend isa typeof(AK.CPU_BACKEND)
        AK.itask_partition(num_checks, options.num_threads, options.min_traversals_per_thread) do itask, irange
            I = get_index_type(bvh1)
            stack = SimpleMVector{32, I}(undef)
            iwrite = if isnothing(contacts)
                Ref(I(itask))
            else
                Ref(itask == 0x1 ? I(0x1) : I(thread_ncontacts[itask - 0x1] + 0x1))
            end
            for i in irange
                bv = @inbounds bvh1.leaves[i]
                @inline traverse_lvt_pair!(
                    bv, bvh2, stack,
                    I(start_level2), iwrite,
                    thread_ncontacts, contacts,
                    narrow,
                    flip,
                )
            end
        end
    else
        # GPU implementation
        AK.foreachindex(1:num_checks, backend, block_size=options.block_size) do i
            I = get_index_type(bvh1)
            stack = SimpleMVector{32, I}(undef)
            iwrite = if isnothing(contacts)
                Ref(I(i))
            else
                Ref(i == 0x1 ? I(0x1) : I(thread_ncontacts[i - 0x1] + 0x1))
            end
            bv = @inbounds bvh1.leaves[i]
            @inline traverse_lvt_pair!(
                bv, bvh2, stack,
                I(start_level2), iwrite,
                thread_ncontacts, contacts,
                narrow,
                flip,
            )
        end
    end

    nothing
end


function traverse_lvt_pair!(
    bv::BoundingVolume,
    bvh::BVH, stack::SimpleMVector,
    start_level::I,
    iwrite::Ref{I},
    thread_ncontacts::AbstractVector{I},
    contacts::Union{Nothing, AbstractVector{IndexPair{I}}},
    narrow,
    ::Val{FLIP},
) where {I <: Integer, FLIP}

    # Traverse a single bounding volume against the BVH using leaf-vs-tree traversal; descend
    # down the left child for as long as there is intersection, then escape back to the next
    # sibling to the right (saved on the stack)
    inode_start = pow2(start_level - 0x1)
    level_num_real = pow2(start_level - 0x1) - bvh.tree.virtual_leaves >> (bvh.tree.levels - start_level)
    inode_end = inode_start + level_num_real - 0x1

    # If the leaf type is not the same as the node type, convert locally the leaf to node type so
    # node-node checks are faster / uniform; at the leaf level we use the leaf's bounding volume
    NodeType = eltype(bvh.nodes)
    bv_node = bv.volume isa NodeType ? bv.volume : NodeType(bv.volume)

    @inbounds for inode_root in inode_start:inode_end
        istack = I(0)
        inode = inode_root
        while true
            ilevel = unsafe_ilog2(inode, RoundDown) + 0x1
            if ilevel == bvh.tree.levels
                leaf = @inbounds bvh.leaves[inode - pow2(bvh.tree.levels - 0x1) + 0x1]
                if iscontact(bv.volume, leaf.volume) && (FLIP ? narrow(leaf, bv) : narrow(bv, leaf))
                    if isnothing(contacts)
                        # Counting pass - known at compile-time
                        @inbounds thread_ncontacts[iwrite[]] += I(1)
                    else
                        # Writing pass
                        if FLIP
                            @inbounds contacts[iwrite[]] = (leaf.index, bv.index)
                        else
                            @inbounds contacts[iwrite[]] = (bv.index, leaf.index)
                        end
                        iwrite[] += I(1)
                    end
                end
            else
                node = @inbounds bvh.nodes[inode - bvh.skips[ilevel]]
                if iscontact(bv_node, node)
                    # If the right child exists, push to the stack and carry on with the left child
                    if !unsafe_isvirtual(bvh.tree, 0x2 * inode + 0x1)
                        istack += 0x1
                        @inbounds stack[istack] = 0x2 * inode + 0x1
                    end
                    inode = 0x2 * inode
                    continue
                end
            end

            if istack == 0x0
                break
            else
                # Pop from stack
                inode = @inbounds stack[istack]
                istack -= 0x1
            end
        end
    end

    nothing
end
