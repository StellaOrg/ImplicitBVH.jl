"""
    traverse(
        bvh1::BVH,
        bvh2::BVH,
        start_level1::Int=default_start_level(bvh1),
        start_level2::Int=default_start_level(bvh2),
        cache::Union{Nothing, BVHTraversal}=nothing;
        options=BVHOptions(),
    )::BVHTraversal

Return all the `bvh1` bounding volume leaves that are in contact with any in `bvh2`. The returned
[`BVHTraversal`](@ref) also contains two contact buffers that can be reused on future traversals.

# Examples

```jldoctest
using ImplicitBVH
using ImplicitBVH: BBox, BSphere

# Generate some simple bounding spheres
bounding_spheres1 = [
    BSphere{Float32}([0., 0., 0.], 0.5),
    BSphere{Float32}([0., 0., 3.], 0.4),
]

bounding_spheres2 = [
    BSphere{Float32}([0., 0., 1.], 0.6),
    BSphere{Float32}([0., 0., 2.], 0.5),
    BSphere{Float32}([0., 0., 4.], 0.6),
]

# Build BVHs
bvh1 = BVH(bounding_spheres1, BBox{Float32}, UInt32)
bvh2 = BVH(bounding_spheres2, BBox{Float32}, UInt32)

# Traverse BVH for contact detection
traversal = traverse(bvh1, bvh2, default_start_level(bvh1), default_start_level(bvh2))

# Reuse traversal buffers for future contact detection - possibly with different BVHs
traversal = traverse(bvh1, bvh2, default_start_level(bvh1), default_start_level(bvh2), traversal)
@show traversal.contacts;
;

# output
traversal.contacts = Tuple{Int32, Int32}[(1, 1), (2, 3)]
```
"""
function traverse(
    bvh1::BVH,
    bvh2::BVH,
    start_level1::Int=default_start_level(bvh1),
    start_level2::Int=default_start_level(bvh2),
    cache::Union{Nothing, BVHTraversal}=nothing;
    options=BVHOptions(),
)

    @boundscheck begin
        @argcheck bvh1.tree.levels >= start_level1 >= bvh1.built_level
        @argcheck bvh2.tree.levels >= start_level2 >= bvh2.built_level
    end

    # Get index type from exemplar
    index_type = typeof(options.index_exemplar)

    # Explanation: say BVH1 has 10 levels, BVH2 has 8 levels; the last level has "leaves" (the
    # actual bounding volumes for contact detection), levels above have "nodes". Both BVHs are
    # aligned to start at level 1. We have two buffers, bvtt1 (src) and bvtt2 (dst); bvtt1 stores
    # the current level's pairs of BVs we need to check for contact. For each contacting pair of
    # BVs in bvtt1, we pair their children for checking contacts at the next level, which we write
    # into bvtt2. Then bvtt1 and bvtt2 are swapped, we advance to the next level, and repeat.
    #
    # A complicating factor is the fact that the two BVHs may have different heights. We then split
    # contact detection into 4 stages: first, we traverse both in sync - that is, at level 1 we have
    # in bvtt1=[(1, 1)] and we write into bvtt2=[(2, 2), (2, 3), (3, 2), (3, 3)] (if the root of
    # BVH1 is in contact with the root of BVH2). We continue at level 2, 3... until we reach level
    # 7, the level above BVH2 leaves. Now we only traverse BVH1, keeping BVH2's level fixed, and
    # doing node-node checks until we reach level 9 in BVH1; now both BVHs have reached the level
    # above leaves. We do another in-sync pairwise contact detection, this time having nodes in
    # bvtt1 (src) and writing possible contacts to check for leaves in bvtt2 (dst). Now we have
    # reached the leaf-level of both BVHs and we write all contacts found into the final contacts
    # vector.

    # Allocate and add all possible BVTT contact pairs to start with
    bvtt1, bvtt2, num_bvtt = initial_bvtt(bvh1, bvh2, start_level1, start_level2, cache, options)
    num_checks = num_bvtt

    # Known at compile time, even though branches have different types
    extra = if bvtt1 isa AbstractGPUVector
        # For GPUs we need an additional global offset to coordinate writing results
        backend = get_backend(bvtt1)
        KernelAbstractions.zeros(backend, index_type, Int(bvh1.tree.levels * bvh2.tree.levels))
    else
        # For CPUs we need a contact counter for each task
        Vector{Int}(undef, options.num_threads)
    end

    # Compute node-node contacts while both BVHs are at node levels
    level1 = start_level1
    level2 = start_level2
    while level1 < bvh1.tree.levels - 1 && level2 < bvh2.tree.levels - 1
        # We can have maximum 4 new checks per contact-pair; resize destination BVTT accordingly
        length(bvtt2) < 4 * num_bvtt && resize!(bvtt2, 4 * num_bvtt)

        # Check contacts in bvtt1 and add future checks in bvtt2
        bvtt1, bvtt2, num_bvtt = traverse_nodes_pair!(bvh1, bvh2, bvtt1, bvtt2, num_bvtt, extra,
                                                      level1, level2, options)
        num_checks += num_bvtt

        # Swap source and destination buffers for next iteration
        bvtt1, bvtt2 = bvtt2, bvtt1
        level1 += 1
        level2 += 1
    end

    # Compute node-node contacts while only right BVH is at level above leaves
    while level1 < bvh1.tree.levels - 1 && level2 == bvh2.tree.levels - 1
        # We can have maximum 2 new checks per contact-pair; resize destination BVTT accordingly
        length(bvtt2) < 2 * num_bvtt && resize!(bvtt2, 2 * num_bvtt)

        # Check contacts in bvtt1 and add future checks in bvtt2
        bvtt1, bvtt2, num_bvtt = traverse_nodes_left!(bvh1, bvh2, bvtt1, bvtt2, num_bvtt, extra,
                                                      level1, level2, options)
        num_checks += num_bvtt

        # Swap source and destination buffers for next iteration
        bvtt1, bvtt2 = bvtt2, bvtt1
        level1 += 1
    end

    # Compute node-node contacts while only left BVH is at level above leaves
    while level2 < bvh2.tree.levels - 1 && level1 == bvh1.tree.levels - 1
        # We can have maximum 2 new checks per contact-pair; resize destination BVTT accordingly
        length(bvtt2) < 2 * num_bvtt && resize!(bvtt2, 2 * num_bvtt)

        # Check contacts in bvtt1 and add future checks in bvtt2
        bvtt1, bvtt2, num_bvtt = traverse_nodes_right!(bvh1, bvh2, bvtt1, bvtt2, num_bvtt, extra,
                                                       level1, level2, options)
        num_checks += num_bvtt

        # Swap source and destination buffers for next iteration
        bvtt1, bvtt2 = bvtt2, bvtt1
        level2 += 1
    end

    # Special case: if the right BVH is already at leaf level (i.e. it either had a single leaf or
    # start_level2 == bvh2.tree.levels) then we must do node-leaf checks down to both leaf levels
    while level2 == bvh2.tree.levels && level1 < bvh1.tree.levels
        # We can have maximum 2 new checks per contact-pair; resize destination BVTT accordingly
        length(bvtt2) < 2 * num_bvtt && resize!(bvtt2, 2 * num_bvtt)

        # Check contacts in bvtt1 and add future checks in bvtt2
        bvtt1, bvtt2, num_bvtt = traverse_nodes_leaves_left!(bvh1, bvh2, bvtt1, bvtt2, num_bvtt, extra,
                                                             level1, level2, options)
        num_checks += num_bvtt

        # Swap source and destination buffers for next iteration
        bvtt1, bvtt2 = bvtt2, bvtt1
        level1 += 1
    end

    # Special case: if the left BVH is already at leaf level (i.e. it either had a single leaf or
    # start_level1 == bvh1.tree.levels) then we must do node-leaf checks down to both leaf levels
    while level1 == bvh1.tree.levels && level2 < bvh2.tree.levels
        # We can have maximum 2 new checks per contact-pair; resize destination BVTT accordingly
        length(bvtt2) < 2 * num_bvtt && resize!(bvtt2, 2 * num_bvtt)

        # Check contacts in bvtt1 and add future checks in bvtt2
        bvtt1, bvtt2, num_bvtt = traverse_nodes_leaves_right!(bvh1, bvh2, bvtt1, bvtt2, num_bvtt, extra,
                                                              level1, level2, options)
        num_checks += num_bvtt

        # Swap source and destination buffers for next iteration
        bvtt1, bvtt2 = bvtt2, bvtt1
        level2 += 1
    end

    # Compute node-node contacts when both BVHs are at level above leaves
    if level1 == bvh1.tree.levels - 1 && level2 == bvh2.tree.levels - 1
        # We can have maximum 4 new checks per contact-pair; resize destination BVTT accordingly
        length(bvtt2) < 4 * num_bvtt && resize!(bvtt2, 4 * num_bvtt)

        # Check contacts in bvtt1 and add future checks in bvtt2
        bvtt1, bvtt2, num_bvtt = traverse_nodes_pair!(bvh1, bvh2, bvtt1, bvtt2, num_bvtt, extra,
                                                      level1, level2, options)
        num_checks += num_bvtt

        # Swap source and destination buffers for next iteration
        bvtt1, bvtt2 = bvtt2, bvtt1
        level1 += 1
        level2 += 1
    end

    # Arrived at final leaf level with both BVHs, now populating contact list
    length(bvtt2) < num_bvtt && resize!(bvtt2, num_bvtt)
    bvtt1, bvtt2, num_bvtt = traverse_leaves_pair!(bvh1, bvh2, bvtt1, bvtt2, num_bvtt, extra, options)

    # Return contact list and the other buffer as possible cache
    BVHTraversal(start_level1, start_level2, num_checks, Int(num_bvtt), bvtt2, bvtt1)
end


function initial_bvtt(bvh1::BVH, bvh2::BVH, start_level1::Int, start_level2::Int, cache, options)
    # Get index type from exemplar
    index_type = typeof(options.index_exemplar)

    # Generate all possible contact checks at the given start_level
    level_nodes1 = pow2(start_level1 - 1)
    level_nodes2 = pow2(start_level2 - 1)

    # Number of real nodes at the given start_level and number of checks we'll do
    num_real1 = level_nodes1 - bvh1.tree.virtual_leaves >> (bvh1.tree.levels - start_level1)
    num_real2 = level_nodes2 - bvh2.tree.virtual_leaves >> (bvh2.tree.levels - start_level2)
    level_checks = num_real1 * num_real2

    # If we're not at leaf-level, allocate enough memory for next BVTT expansion
    if start_level1 == bvh1.tree.levels && start_level2 == bvh2.tree.levels
        initial_number = level_checks               # Both at leaf level
    elseif start_level1 == bvh1.tree.levels || start_level2 == bvh2.tree.levels
        initial_number = 2 * level_checks           # Only one at leaf level
    else
        initial_number = 4 * level_checks           # Neither at leaf level
    end

    # Reuse cache if given
    if isnothing(cache)
        bvtt1 = similar(bvh1.nodes, IndexPair{index_type}, initial_number)
        bvtt2 = similar(bvh1.nodes, IndexPair{index_type}, initial_number)
    else
        @argcheck eltype(cache.cache1) === IndexPair{index_type}
        @argcheck eltype(cache.cache2) === IndexPair{index_type}

        bvtt1 = cache.cache1
        bvtt2 = cache.cache2

        length(bvtt1) < initial_number && resize!(bvtt1, initial_number)
        length(bvtt2) < initial_number && resize!(bvtt2, initial_number)
    end

    # Insert all checks to do at this level
    num_bvtt = fill_initial_bvtt_pair!(bvtt1, level_nodes1, num_real1, level_nodes2, num_real2, options)

    bvtt1, bvtt2, num_bvtt
end


function fill_initial_bvtt_pair!(bvtt1, level_nodes1, num_real1, level_nodes2, num_real2, options)
    backend = get_backend(bvtt1)
    if backend isa GPU
        # GPU version with the two for loops (see CPU) linearised
        AK.foreachindex(1:num_real1 * num_real2, backend, block_size=options.block_size) do i
            irow, icol = divrem(i - 1, num_real2)
            bvtt1[i] = (irow + level_nodes1, icol + level_nodes2)
        end
        return num_real1 * num_real2
    else
        # CPU initial checks; this uses such simple instructions that single threading is fastest
        num_bvtt = 0
        @inbounds for i in level_nodes1:level_nodes1 + num_real1 - 1
            # Node-node pair checks
            for j in level_nodes2:level_nodes2 + num_real2 - 1
                num_bvtt += 1
                bvtt1[num_bvtt] = (i, j)
            end
        end
        return num_bvtt
    end
end


# Traversal implementations
include("traverse_pair_cpu.jl")
include("traverse_pair_gpu.jl")

