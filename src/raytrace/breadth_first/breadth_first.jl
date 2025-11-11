function traverse_rays(
    bvh::BVH,
    points::AbstractMatrix, directions::AbstractMatrix,
    alg::BFSTraversal;
    start_level::Int=1,
    narrow=(bv, p, d) -> true,
    cache::Union{Nothing, BVHTraversal}=nothing,
    options=BVHOptions(),
)
    # Correctness checks
    @boundscheck begin
        @argcheck bvh.tree.levels >= start_level >= bvh.built_level
        @argcheck size(points, 1) == size(directions, 1) == 3
        @argcheck size(points, 2) == size(directions, 2)
    end

    num_rays = size(points, 2)

    # Get index type from exemplar; compile-time safety checks
    I = get_index_type(options)
    @argcheck AK.get_backend(bvh.nodes) == AK.get_backend(points) == AK.get_backend(directions)

    # No intersections for no rays
    if num_rays == 0
        return BVHTraversal(start_level, 0, 0,
                            similar(bvh.nodes, IndexPair{I}, 0),
                            similar(bvh.nodes, IndexPair{I}, 0))
    end

    # Allocate and add all possible BVTT contact pairs to start with
    bvtt1, bvtt2, num_bvtt = initial_bvtt(bvh, points, directions, start_level, cache, options)
    num_checks = num_bvtt

    # For GPUs we need an additional global offset to coordinate writing results
    dst_offsets = if bvtt1 isa AbstractGPUVector
        backend = get_backend(bvtt1)
        KernelAbstractions.zeros(backend, I, Int(bvh.tree.levels))
    else
        nothing
    end

    level = start_level
    while level < bvh.tree.levels
        # We can have maximum 2 new checks per BV-ray-pair; resize destination BVTT accordingly
        length(bvtt2) < 2 * num_bvtt && resize!(bvtt2, 2 * num_bvtt)

        # Check intersections in bvtt1 and add future checks in bvtt2
        num_bvtt = traverse_rays_nodes!(bvh, points, directions,
                                        bvtt1, bvtt2, num_bvtt,
                                        dst_offsets, level, options)
        num_checks += num_bvtt

        # Swap source and destination buffers for next iteration
        bvtt1, bvtt2 = bvtt2, bvtt1
        level += 1
    end

    # Arrived at final leaf level, now populating contact list
    length(bvtt2) < num_bvtt && resize!(bvtt2, num_bvtt)
    num_bvtt = traverse_rays_leaves!(bvh, points, directions,
                                     bvtt1, bvtt2, num_bvtt,
                                     dst_offsets, narrow, options)

    # Return contact list and the other buffer as possible cache
    BVHTraversal(start_level, num_checks, Int(num_bvtt), bvtt2, bvtt1)
end


function initial_bvtt(
    bvh::BVH,
    points::AbstractArray,
    directions::AbstractArray,
    start_level,
    cache,
    options,
)
    num_rays = size(points, 2)

    # Get index type from exemplar
    index_type = typeof(options.index_exemplar)

    # Generate all possible contact checks between all nodes at the given start_level and all rays
    level_nodes = pow2(start_level - 1)

    # Number of real nodes at the given start_level and number of checks we'll do
    num_real = level_nodes - bvh.tree.virtual_leaves >> (bvh.tree.levels - start_level)
    level_checks = num_real * num_rays

    # If we're not at leaf-level, allocate enough memory for next BVTT expansion
    if start_level == bvh.tree.levels
        initial_number = level_checks
    else
        initial_number = 2 * level_checks
    end

    # Reuse cache if given
    if isnothing(cache)
        bvtt1 = similar(bvh.nodes, IndexPair{index_type}, initial_number)
        bvtt2 = similar(bvh.nodes, IndexPair{index_type}, initial_number)
    else
        @argcheck eltype(cache.cache1) === IndexPair{index_type}
        @argcheck eltype(cache.cache2) === IndexPair{index_type}

        bvtt1 = cache.cache1
        bvtt2 = cache.cache2

        length(bvtt1) < initial_number && resize!(bvtt1, initial_number)
        length(bvtt2) < initial_number && resize!(bvtt2, initial_number)
    end

    # Insert all checks to do at this level
    fill_initial_bvtt_rays!(bvtt1, level_nodes, num_real, num_rays, options)

    bvtt1, bvtt2, level_checks
end


function fill_initial_bvtt_rays!(bvtt1, level_nodes, num_real, num_rays, options)
    backend = get_backend(bvtt1)
    if backend isa GPU
        # GPU version with the two for loops (see CPU) linearised
        AK.foreachindex(1:num_real * num_rays, backend, block_size=options.block_size) do i
            irow, icol = divrem(i - 1, num_rays)
            bvtt1[i] = (irow + level_nodes, icol + 1)
        end
    else
        # CPU initial checks; this uses such simple instructions that single threading is fastest
        num_bvtt = 0
        @inbounds for i in level_nodes:level_nodes + num_real - 1
            # Node-node pair checks
            for j in 1:num_rays
                num_bvtt += 1
                bvtt1[num_bvtt] = (i, j)
            end
        end
    end
end


# Traversal implementations
include("raytrace_cpu.jl")
include("raytrace_gpu.jl")
