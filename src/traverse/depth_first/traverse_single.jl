function traverse(
    bvh::BVH, ::DFSTraversal;
    start_level::Int=default_start_level(bvh),
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
        # CPU backend - one per thread (one thread handles multiple start_level pairs here)
        padding = div(64, sizeof(I), RoundUp)
        thread_ncontacts_length = padding * options.num_threads
    else
        # GPU backend - one per start_level pair to check (one thread per pair here)
        padding = 1
        thread_ncontacts_length = compute_num_checks(bvh, start_level)
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
    thread_ncontacts = reshape(thread_ncontacts_raw, padding, :)
    thread_ncontacts[1, :] .= I(0)

    # On the first pass, just count how many contacts we will have per thread
    traverse_contacts!(bvh, start_level, thread_ncontacts, nothing, options)

    # Accumulate numbers of contacts to get thread offsets to write contacts based on
    AK.accumulate!(+, @view(thread_ncontacts[1, :]), init=I(0), max_tasks=1, block_size=options.block_size)

    # Allocate contacts vector
    total_contacts = @allowscalar thread_ncontacts[1, end]
    if isnothing(cache)
        contacts = similar(bvh.nodes, IndexPair{I}, total_contacts)
    else
        @argcheck eltype(cache.cache1) === IndexPair{I}
        contacts = cache.cache1
        length(contacts) < total_contacts && resize!(contacts, total_contacts)
    end

    if total_contacts == 0
        # No contacts, return empty traversal
        return BVHTraversal(Int(start_level), 0, Int(total_contacts), contacts, thread_ncontacts_raw)
    end

    # Do second pass where contacts are written
    traverse_contacts!(bvh, start_level, thread_ncontacts, contacts, options)

    # Return contact list and the other buffer as possible cache
    BVHTraversal(Int(start_level), 0, Int(total_contacts), contacts, thread_ncontacts_raw)
end


function traverse_contacts!(
    bvh,
    start_level,
    thread_ncontacts,
    contacts::Union{Nothing, AbstractVector},
    options,
)
    # Pre-compute number of skips to do in memory for each level
    skips = compute_skips(bvh)

    # Compute number of initial checks at the start_level
    level_nodes = pow2(start_level - 1)
    num_real = level_nodes - bvh.tree.virtual_leaves >> (bvh.tree.levels - start_level)
    num_checks = compute_num_checks(bvh, start_level)

    backend = AK.get_backend(bvh.nodes)
    if backend isa typeof(AK.CPU_BACKEND)
        AK.itask_partition(num_checks, options.num_threads) do itask, irange

            I = get_index_type(bvh)

            # Stack buffer of contact pairs - hardcoded for at most 32 levels deep (1 level per
            # column), and at most 4 pairs per level
            checks = MMatrix{4, 32, IndexPair{I}}(undef)

            # Number of pair checks to do at this level
            nchecks = MVector{32, I}(undef)

            # Index for the current pair at a given level
            ichecks = MVector{32, I}(undef)

            # Index into contacts for writing during the second pass
            icontact = if isnothing(contacts)
                nothing
            else
                Ref(itask == 1 ? I(1) : I(thread_ncontacts[1, itask - 1] + 1))
            end

            # For last 3 levels switch to breadth-first traversal, allocate stack buffers
            checks_last2 = MVector{4, IndexPair{I}}(undef)      # Leaf-level minus 2
            checks_last1 = MVector{16, IndexPair{I}}(undef)     # Leaf-level minus 1
            checks_last0 = MVector{64, IndexPair{I}}(undef)     # Leaf-level

            for i in irange

                # Interleave thread accesses in strides of num_threads to balance loads, as later
                # Morton indices will be closer in space and thus have more contacts
                k = interleaved_index(i, num_checks, options.num_threads)
                @inline _traverse_depth!(
                    bvh, skips, I(itask),
                    contacts, thread_ncontacts, icontact,
                    checks, nchecks, ichecks,
                    checks_last2, checks_last1, checks_last0,
                    I(start_level), I(level_nodes), I(num_real), I(k),
                )
            end
        end
    else
        # GPU implementation
        AK.foreachindex(1:num_checks, backend, block_size=options.block_size) do i
            I = get_index_type(bvh)

            # Stack buffer of contact pairs - hardcoded for at most 32 levels deep (1 level per
            # column), and at most 4 pairs per level
            checks = MMatrix{4, 32, IndexPair{I}}(undef)

            # Number of pair checks to do at this level
            nchecks = MVector{32, I}(undef)

            # Index for the current pair at a given level
            ichecks = MVector{32, I}(undef)

            # Index into contacts for writing during the second pass
            icontact = if isnothing(contacts)
                nothing
            else
                Ref(i == 0x1 ? I(0x1) : I(thread_ncontacts[0x1, i - 0x1] + 0x1))
            end

            # For last 3 levels switch to breadth-first traversal, allocate stack buffers
            checks_last2 = MVector{4, IndexPair{I}}(undef)      # Leaf-level minus 2
            checks_last1 = MVector{16, IndexPair{I}}(undef)     # Leaf-level minus 1
            checks_last0 = MVector{64, IndexPair{I}}(undef)     # Leaf-level

            @inline _traverse_depth!(
                bvh, skips, I(i),
                contacts, thread_ncontacts, icontact,
                checks, nchecks, ichecks,
                checks_last2, checks_last1, checks_last0,
                I(start_level), I(level_nodes), I(num_real), I(i),
            )
        end
    end

    nothing
end


# Compute number of initial checks at the start_level
function compute_num_checks(bvh, start_level)
    level_nodes = pow2(start_level - 1)
    num_real = level_nodes - bvh.tree.virtual_leaves >> (bvh.tree.levels - start_level)

    if start_level != bvh.tree.levels
        # If we are not at the last level, we have n * (n + 1) ÷ 2 checks (including self-checks)
        return num_real * (num_real + 1) ÷ 2
    else
        # If we are at the last level, we only have n * (n - 1) ÷ 2 checks (i.e. no self-checks)
        return num_real * (num_real - 1) ÷ 2
    end

end


@inbounds function _traverse_depth!(
    bvh, skips, itask::I,
    contacts::Union{Nothing, AbstractVector}, thread_ncontacts, icontact::Union{Nothing, <:Ref{I}},
    checks, nchecks, ichecks,
    checks_last2, checks_last1, checks_last0,
    start_level::I, level_nodes::I, num_real::I, i::I,
) where I <: Integer

    # Initialise: first, single check at given start_level from which we will do the depth-first
    # traversal
    ilevel = I(start_level)
    @inbounds nchecks[ilevel] = I(1)
    @inbounds ichecks[ilevel] = I(1)

    # Get the (i, j) pair for the current check
    if start_level != bvh.tree.levels
        @inbounds checks[1, ilevel] = _initial_level_pair_inclusive(level_nodes, num_real, i)
    else
        @inbounds checks[1, ilevel] = _initial_level_pair_exclusive(level_nodes, num_real, i)
    end

    # Do depth-first traversal of the tree
    while ilevel >= start_level

        if @inbounds ichecks[ilevel] > nchecks[ilevel]
            # No more checks at this level, go up
            ilevel -= I(1)
            continue
        end

        pair = @inbounds checks[ichecks[ilevel], ilevel]

        # Switch to breadth-first traversal for the last 3 levels to improve cache locality while
        # keeping the stack size known - we get about 20% speed up
        if ilevel == bvh.tree.levels - I(3)
            @inline _traverse_breadth_minus3!(
                bvh, skips, itask,
                contacts, thread_ncontacts, icontact,
                pair,
                checks_last2, checks_last1, checks_last0,
            )
            # Increment index for the next check at this level; remain at leaf-level minus 3
            @inbounds ichecks[ilevel] += I(1)

        elseif ilevel == bvh.tree.levels - I(2)
            @inline _traverse_breadth_minus2!(
                bvh, skips, itask,
                contacts, thread_ncontacts, icontact,
                pair,
                checks_last1, checks_last0,
            )
            # Increment index for the next check at this level; stay at leaf-level minus 2
            @inbounds ichecks[ilevel] += I(1)
        
        # At leaf-level minus 1 breadth-first is equivalent to depth-first
        elseif ilevel == bvh.tree.levels

            # At leaf level, we have our two different passes:
            # - First pass, when contacts::Nothing (compile-time known) we only count how many
            #   contacts we will have
            # - Second pass, when contacts::AbstractVector we write contacts at offset icontact
            if isnothing(contacts)
                @inline _pair_leaf_leaf_count!(bvh, pair, thread_ncontacts, itask)
            else
                @inline _pair_leaf_leaf_write!(bvh, pair, contacts, icontact)
            end

            # Increment index for the next check at this level; stay at last level
            @inbounds ichecks[ilevel] += I(1)

        else
            # Normal depth-first pairing
            @inline _depth_pair!(bvh, pair, checks, nchecks, skips, ilevel)

            # Increment index for the next check at this level
            @inbounds ichecks[ilevel] += I(1)

            # Advance to the next level
            ilevel += I(1)
            @inbounds ichecks[ilevel] = I(1)
        end
    end

    nothing
end




# Return the k-th element of the interleaved order of 1:N by stride M:
# 1, 1+M, 1+2M, …, 2, 2+M, …, M, M+M, …
# Arguments are 1-based. Requires N ≥ 1, M ≥ 1, and 1 ≤ k ≤ N.
function interleaved_index(k::I, N::I, M::I) where I <: Integer
    q = div(N, M)                                   # floor(N/M)
    r = rem(N, M)                                   # N % M, with 0 ≤ r < M
    longlen = q + I(1)                              # length of the first r columns
    longtotal = r * longlen

    if k <= longtotal
        c = div(k - I(1), longlen) + I(1)           # column (1..r)
        t = rem(k - I(1), longlen)                  # row within column
        return c + t * M
    else
        k2 = k - longtotal
        shortlen = q
        c = r + div(k2 - I(1), shortlen) + I(1)     # column (r+1..M)
        t = rem(k2 - I(1), shortlen)
        return c + t * M
    end
end




# Get i_lin-th (i, j) for level_nodes if we have num_real nodes at the given level
# i_lin between 1 and n_lin + n
@inline function _initial_level_pair_exclusive(
    level_nodes::I,
    num_real::I,
    i_lin::I,
) where I <: Integer

    i, j = _k2ij_exclusive(num_real, i_lin - I(1))
    i += level_nodes
    j += level_nodes
    return (i, j)
end


# Get i_lin-th (i, j) for level_nodes if we have num_real nodes at the given level
# i_lin between 1 and n_lin + n
@inline function _initial_level_pair_inclusive(
    level_nodes::I,
    num_real::I,
    i_lin::I,
) where I <: Integer

    i, j = _k2ij_inclusive(num_real, i_lin - I(1))
    i += level_nodes
    j += level_nodes
    return (i, j)
end




# Depth-first traversal
function _depth_pair!(bvh, pair, checks, nchecks, skips, ilevel)
    I = get_index_type(bvh)

    # When dealing with a self-check (1, 1) we pair the children (2, 3); we sprout
    # further self-checks (2, 2) and (3, 3) only if we are not at level above
    # leaves (leaf-leaf self checks are not needed)
    if pair[1] == pair[2]
        if ilevel < bvh.tree.levels - I(1)
            @inline _depth_pair_self_self_checks!(bvh, pair, checks, nchecks, ilevel)
        else
            @inline _depth_pair_self!(bvh, pair, checks, nchecks, ilevel)
        end

    # Node-node check, only pair children
    else
        @inline _depth_pair_node_node!(bvh, pair, checks, nchecks, skips, ilevel)
    end
end


# If self-check (1, 1), sprout children self-checks (2, 2) (3, 3) and pair children (2, 3)
function _depth_pair_self_self_checks!(bvh, pair, checks, nchecks, ilevel)
    # Extract implicit indices of BVH nodes to test
    implicit1, implicit2 = pair

    # If the right child is virtual, only add left child self-check
    if unsafe_isvirtual(bvh.tree, 0x2 * implicit1 + 0x1)
        @inbounds checks[0x1, ilevel + 0x1] = _leftleft(implicit1, implicit1)
        @inbounds nchecks[ilevel + 0x1] = 0x1
    else
        @inbounds checks[0x1, ilevel + 0x1] = _leftleft(implicit1, implicit1)
        @inbounds checks[0x2, ilevel + 0x1] = _leftright(implicit1, implicit1)
        @inbounds checks[0x3, ilevel + 0x1] = _rightright(implicit1, implicit1)
        @inbounds nchecks[ilevel + 0x1] = 0x3
    end
end


# If self-check (1, 1), only pair children (2, 3)
function _depth_pair_self!(bvh, pair, checks, nchecks, ilevel)
    # Extract implicit indices of BVH nodes to test
    implicit1, implicit2 = pair

    # If the right child is virtual, nothing to pair
    if unsafe_isvirtual(bvh.tree, 0x2 * implicit1 + 0x1)
        @inbounds nchecks[ilevel + 0x1] = 0
    else
        @inbounds checks[0x1, ilevel + 0x1] = _leftright(implicit1, implicit1)
        @inbounds nchecks[ilevel + 0x1] = 0x1
    end
end


# Pair children of the two nodes
function _depth_pair_node_node!(bvh, pair, checks, nchecks, skips, ilevel)
    # Extract implicit indices of BVH nodes to test
    implicit1, implicit2 = pair

    # Number of virtual nodes to skip in memory to get to the real node
    num_skips = @inbounds skips[ilevel]
    node1 = @inbounds bvh.nodes[implicit1 - num_skips]
    node2 = @inbounds bvh.nodes[implicit2 - num_skips]

    # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
    # the nodes' children
    if iscontact(node1, node2)
        # If the right node's right child is virtual, don't add that check. Guaranteed to
        # always have node1 to the left of node2, hence its children will always be real
        if unsafe_isvirtual(bvh.tree, 0x2 * implicit2 + 0x1)
            @inbounds checks[0x1, ilevel + 0x1] = _leftleft(implicit1, implicit2)
            @inbounds checks[0x2, ilevel + 0x1] = _rightleft(implicit1, implicit2)
            @inbounds nchecks[ilevel + 0x1] = 0x2
        else
            @inbounds checks[0x1, ilevel + 0x1] = _leftleft(implicit1, implicit2)
            @inbounds checks[0x2, ilevel + 0x1] = _leftright(implicit1, implicit2)
            @inbounds checks[0x3, ilevel + 0x1] = _rightleft(implicit1, implicit2)
            @inbounds checks[0x4, ilevel + 0x1] = _rightright(implicit1, implicit2)
            @inbounds nchecks[ilevel + 0x1] = 0x4
        end
    else
        # No contact, no children to pair
        @inbounds nchecks[ilevel + 0x1] = 0x0
    end
end




# Breadth-first traversal for the last 3 levels
function _traverse_breadth_minus3!(
    bvh, skips, itask,
    contacts, thread_ncontacts, icontact,
    pair,
    checks_last2, checks_last1, checks_last0,
)
    I = get_index_type(bvh)

    ilevel = bvh.tree.levels - I(3)
    num_written2 = I(0)
    num_written2 = @inline _breadth_pair!(bvh, skips, pair, ilevel, checks_last2, num_written2)

    ilevel = bvh.tree.levels - I(2)
    num_written1 = I(0)
    for i in I(1):num_written2
        pair = @inbounds checks_last2[i]
        num_written1 += @inline _breadth_pair!(bvh, skips, pair, ilevel, checks_last1, num_written1)
    end

    ilevel = bvh.tree.levels - I(1)
    num_written0 = I(0)
    for i in I(1):num_written1
        pair = @inbounds checks_last1[i]
        num_written0 += @inline _breadth_pair!(bvh, skips, pair, ilevel, checks_last0, num_written0)
    end

    # At leaf level, we have our two different passes:
    # - First pass, when contacts::Nothing (compile-time known) we only count how many
    #   contacts we will have
    # - Second pass, when contacts::AbstractVector we write contacts at offset icontact
    for i in I(1):num_written0
        pair = @inbounds checks_last0[i]
        if isnothing(contacts)
            @inline _pair_leaf_leaf_count!(bvh, pair, thread_ncontacts, itask)
        else
            @inline _pair_leaf_leaf_write!(bvh, pair, contacts, icontact)
        end
    end
end


# Breadth-first traversal for the last 2 levels
function _traverse_breadth_minus2!(
    bvh, skips, itask,
    contacts, thread_ncontacts, icontact,
    pair,
    checks_last1, checks_last0,
)
    I = get_index_type(bvh)

    ilevel = bvh.tree.levels - I(2)
    num_written1 = I(0)
    num_written1 = @inline _breadth_pair!(bvh, skips, pair, ilevel, checks_last1, num_written1)

    ilevel = bvh.tree.levels - I(1)
    num_written0 = I(0)
    for i in I(1):num_written1
        pair = @inbounds checks_last1[i]
        num_written0 += @inline _breadth_pair!(bvh, skips, pair, ilevel, checks_last0, num_written0)
    end

    # At leaf level, we have our two different passes:
    # - First pass, when contacts::Nothing (compile-time known) we only count how many
    #   contacts we will have
    # - Second pass, when contacts::AbstractVector we write contacts at offset icontact
    for i in I(1):num_written0
        pair = @inbounds checks_last0[i]
        if isnothing(contacts)
            @inline _pair_leaf_leaf_count!(bvh, pair, thread_ncontacts, itask)
        else
            @inline _pair_leaf_leaf_write!(bvh, pair, contacts, icontact)
        end
    end
end


function _breadth_pair!(bvh, skips, pair, ilevel, checks, offset)
    I = get_index_type(bvh)

    # When dealing with a self-check (1, 1) we pair the children (2, 3); we sprout
    # further self-checks (2, 2) and (3, 3) only if we are not at level above
    # leaves (leaf-leaf self checks are not needed)
    if pair[1] == pair[2]
        if ilevel < bvh.tree.levels - I(1)
            num_written = @inline _breadth_pair_self_self_checks!(bvh, pair, checks, offset)
        else
            num_written = @inline _breadth_pair_self!(bvh, pair, checks, offset)
        end

    # Node-node check, only pair children
    else
        num_written = @inline _breadth_pair_node_node!(bvh, pair, checks, skips, ilevel, offset)
    end

    num_written
end


# If self-check (1, 1), sprout children self-checks (2, 2) (3, 3) and pair children (2, 3)
function _breadth_pair_self_self_checks!(bvh, pair, checks, offset)
    I = get_index_type(bvh)

    # Extract implicit indices of BVH nodes to test
    implicit1, implicit2 = pair

    # If the right child is virtual, only add left child self-check
    if unsafe_isvirtual(bvh.tree, 0x2 * implicit1 + 0x1)
        @inbounds checks[offset + 0x1] = _leftleft(implicit1, implicit1)
        return I(1)
    else
        @inbounds checks[offset + 0x1] = _leftleft(implicit1, implicit1)
        @inbounds checks[offset + 0x2] = _leftright(implicit1, implicit1)
        @inbounds checks[offset + 0x3] = _rightright(implicit1, implicit1)
        return I(3)
    end
end


# If self-check (1, 1), only pair children (2, 3)
function _breadth_pair_self!(bvh, pair, checks, offset)
    I = get_index_type(bvh)

    # Extract implicit indices of BVH nodes to test
    implicit1, implicit2 = pair

    # If the right child is virtual, nothing to pair
    if unsafe_isvirtual(bvh.tree, 0x2 * implicit1 + 0x1)
        return I(0)
    else
        @inbounds checks[offset + 0x1] = _leftright(implicit1, implicit1)
        return I(1)
    end
end


# Pair children of the two nodes
function _breadth_pair_node_node!(bvh, pair, checks, skips, ilevel, offset)
    I = get_index_type(bvh)

    # Extract implicit indices of BVH nodes to test
    implicit1, implicit2 = pair

    # Number of virtual nodes to skip in memory to get to the real node
    num_skips = @inbounds skips[ilevel]
    node1 = @inbounds bvh.nodes[implicit1 - num_skips]
    node2 = @inbounds bvh.nodes[implicit2 - num_skips]

    # If the two nodes are touching, expand BVTT with new possible contacts - i.e. pair
    # the nodes' children
    if iscontact(node1, node2)
        # If the right node's right child is virtual, don't add that check. Guaranteed to
        # always have node1 to the left of node2, hence its children will always be real
        if unsafe_isvirtual(bvh.tree, 0x2 * implicit2 + 0x1)
            @inbounds checks[offset + 0x1] = _leftleft(implicit1, implicit2)
            @inbounds checks[offset + 0x2] = _rightleft(implicit1, implicit2)
            return I(2)
        else
            @inbounds checks[offset + 0x1] = _leftleft(implicit1, implicit2)
            @inbounds checks[offset + 0x2] = _leftright(implicit1, implicit2)
            @inbounds checks[offset + 0x3] = _rightleft(implicit1, implicit2)
            @inbounds checks[offset + 0x4] = _rightright(implicit1, implicit2)
            return I(4)
        end
    else
        # No contact, no children to pair
        return I(0)
    end
end





# Leaf-level traversal
function _pair_leaf_leaf_count!(bvh, pair, thread_ncontacts, itask)
    # Number of implicit indices above leaf-level
    num_above = pow2(bvh.tree.levels - 0x1) - 0x1

    # Extract implicit indices of BVH leaves to test
    implicit1, implicit2 = pair
    leaf1 = @inbounds bvh.leaves[implicit1 - num_above]
    leaf2 = @inbounds bvh.leaves[implicit2 - num_above]

    # First pass: count contacts
    if iscontact(leaf1.volume, leaf2.volume)
        @inbounds thread_ncontacts[0x1, itask] += 0x1
    end
end


function _pair_leaf_leaf_write!(bvh, pair, contacts, icontact)
    # Number of implicit indices above leaf-level
    num_above = pow2(bvh.tree.levels - 0x1) - 0x1

    # Extract implicit indices of BVH leaves to test
    implicit1, implicit2 = pair
    leaf1 = @inbounds bvh.leaves[implicit1 - num_above]
    leaf2 = @inbounds bvh.leaves[implicit2 - num_above]

    # Second pass: write contacts
    if iscontact(leaf1.volume, leaf2.volume)
        @inbounds contacts[icontact[]] = (leaf1.index, leaf2.index)
        icontact[] += 0x1
    end
end










# Traversal implementations
# include("traverse_single_cpu.jl")
# include("traverse_single_gpu.jl")
