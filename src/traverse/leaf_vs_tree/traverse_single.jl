function traverse(
    bvh::BVH, alg::LVTTraversal;
    start_level::Int=default_start_level(bvh, alg),
    narrow=(bv1, bv2) -> true,
    cache::Union{Nothing, BVHTraversal}=nothing,
    options=BVHOptions(),
)
    # Correctness checks
    @boundscheck begin
        @argcheck bvh.built_level <= start_level <= bvh.tree.levels <= 32
    end

    # Get index type from exemplar
    I = get_index_type(bvh)

    # No contacts / traversal for a single node
    if bvh.tree.real_nodes <= 1
        return BVHTraversal(Int(start_level), 0, 0,
                            similar(bvh.nodes, IndexPair{I}, 0),
                            similar(bvh.nodes, I, 0))
    end

    # Allocate buffer for number of contacts to be written, one for each thread
    if AK.get_backend(bvh.nodes) isa typeof(AK.CPU_BACKEND)
        # CPU backend - one per thread (one thread handles multiple input bounding volumes)
        npadding = div(64, sizeof(I), RoundUp)
        thread_ncontacts_length = npadding * options.num_threads
    else
        # GPU backend - one per input bounding volume
        npadding = 1
        thread_ncontacts_length = length(bvh.leaves)
    end

    # Reuse from cache if possible
    if isnothing(cache)
        thread_ncontacts_raw = similar(bvh.nodes, I, thread_ncontacts_length)
    else
        @argcheck eltype(cache.cache2) === I
        thread_ncontacts_raw = cache.cache2
        if length(thread_ncontacts_raw) < thread_ncontacts_length
            resize!(thread_ncontacts_raw, thread_ncontacts_length)
        end
    end

    # On CPUs take every npadding-th element to avoid false sharing
    if AK.get_backend(bvh.nodes) isa typeof(AK.CPU_BACKEND)
        thread_ncontacts = view(reshape(thread_ncontacts_raw, npadding, :), 1, :)
    else
        thread_ncontacts = thread_ncontacts_raw
    end

    # On the first pass, just count how many contacts we will have per thread
    thread_ncontacts .= I(0)
    traverse_contacts_lvt!(bvh, start_level, thread_ncontacts, nothing, narrow, options)

    # Accumulate numbers of contacts to get thread offsets to write contacts based on
    AK.accumulate!(+, thread_ncontacts, init=I(0), max_tasks=1, block_size=options.block_size)

    # Allocate contacts vector
    total_contacts = @allowscalar thread_ncontacts[end]
    if isnothing(cache)
        contacts = similar(bvh.nodes, IndexPair{I}, total_contacts)
    else
        @argcheck eltype(cache.cache1) === IndexPair{I}
        contacts = cache.cache1
        length(contacts) < total_contacts && resize!(contacts, total_contacts)
    end

    if total_contacts == 0
        # No contacts, return empty traversal
        return BVHTraversal(Int(start_level), 0, 0, contacts, thread_ncontacts_raw)
    end

    # Do second pass where contacts are written
    traverse_contacts_lvt!(bvh, start_level, thread_ncontacts, contacts, narrow, options)

    # Return contact list and the other buffer as possible cache
    BVHTraversal(Int(start_level), 0, Int(total_contacts), contacts, thread_ncontacts_raw)
end


function traverse_contacts_lvt!(
    bvh::BVH,
    start_level::Int,
    thread_ncontacts::AbstractVector,
    contacts::Union{Nothing, AbstractVector{<:IndexPair}},      # Nothing on first (counting) pass
    narrow,
    options::BVHOptions,
)
    backend = AK.get_backend(bvh.nodes)
    num_checks = length(bvh.leaves)

    if backend isa typeof(AK.CPU_BACKEND)
        AK.itask_partition(num_checks, options.num_threads, options.min_traversals_per_thread) do itask, irange
            I = get_index_type(bvh)
            stack = SimpleMVector{32, I}(undef)
            iwrite = if isnothing(contacts)
                Ref(I(itask))
            else
                Ref(itask == 0x1 ? I(0x1) : I(thread_ncontacts[itask - 0x1] + 0x1))
            end
            for i in irange
                bv = @inbounds bvh.leaves[i]
                @inline traverse_lvt_single!(
                    bv, I(i), bvh, stack,
                    I(start_level), iwrite,
                    thread_ncontacts, contacts,
                    narrow,
                )
            end
        end
    else
        # GPU implementation
        AK.foreachindex(1:num_checks, backend, block_size=options.block_size) do i
            I = get_index_type(bvh)
            stack = SimpleMVector{32, I}(undef)
            iwrite = if isnothing(contacts)
                Ref(I(i))
            else
                Ref(i == 0x1 ? I(0x1) : I(thread_ncontacts[i - 0x1] + 0x1))
            end
            bv = @inbounds bvh.leaves[i]
            @inline traverse_lvt_single!(
                bv, I(i), bvh, stack,
                I(start_level), iwrite,
                thread_ncontacts, contacts,
                narrow,
            )
        end
    end

    nothing
end


@inbounds function traverse_lvt_single!(
    bv::BoundingVolume, ileaf::I,
    bvh::BVH, stack::SimpleMVector,
    start_level::I,
    iwrite::Ref{I},
    thread_ncontacts::AbstractVector{I},
    contacts::Union{Nothing, AbstractVector{IndexPair{I}}},
    narrow,
) where {I <: Integer}
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
            ilevel::I = unsafe_ilog2(inode, RoundDown) + 0x1

            # To avoid double counting, ignore checks if inode subtree is fully to the left of bv,
            # i.e. the inode rightmost reachable leaf is > ibv
            irightmost_reachable = ((inode + 0x1) << (bvh.tree.levels - ilevel)) - 0x1
            if irightmost_reachable <= ileaf + pow2(bvh.tree.levels - 0x1) - 0x1
                # Skip this subtree
            elseif ilevel == bvh.tree.levels
                leaf = @inbounds bvh.leaves[inode - pow2(bvh.tree.levels - 0x1) + 0x1]
                if iscontact(bv.volume, leaf.volume) && narrow(bv, leaf)
                    if isnothing(contacts)
                        # Counting pass - known at compile-time
                        @inbounds thread_ncontacts[iwrite[]] += I(1)
                    else
                        # Writing pass; for single-tree traversal, place indices in sorted order
                        @inbounds contacts[iwrite[]] = if bv.index > leaf.index
                            (leaf.index, bv.index)
                        else
                            (bv.index, leaf.index)
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

