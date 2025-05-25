"""
    traverse(
        bvh::BVH,
        start_level::Int=default_start_level(bvh),
        cache::Union{Nothing, BVHTraversal}=nothing;
        options=BVHOptions(),
    )::BVHTraversal

Traverse `bvh` downwards from `start_level`, returning all contacting bounding volume leaves. The
returned [`BVHTraversal`](@ref) also contains two contact buffers that can be reused on future
traversals.

# Examples

```jldoctest
using ImplicitBVH
using ImplicitBVH: BBox, BSphere

# Generate some simple bounding spheres
bounding_spheres = [
    BSphere{Float32}([0., 0., 0.], 0.5),
    BSphere{Float32}([0., 0., 1.], 0.6),
    BSphere{Float32}([0., 0., 2.], 0.5),
    BSphere{Float32}([0., 0., 3.], 0.4),
    BSphere{Float32}([0., 0., 4.], 0.6),
]

# Build BVH
bvh = BVH(bounding_spheres, BBox{Float32}, UInt32)

# Traverse BVH for contact detection
traversal = traverse(bvh, 2)

# Reuse traversal buffers for future contact detection - possibly with different BVHs
traversal = traverse(bvh, 2, traversal)
@show traversal.contacts;
;

# output
traversal.contacts = Tuple{Int32, Int32}[(1, 2), (2, 3), (4, 5)]
```
"""
function traverse(
    bvh::BVH,
    start_level::Int=default_start_level(bvh),
    cache::Union{Nothing, BVHTraversal}=nothing;
    options=BVHOptions(),
)
    # Correctness checks
    @boundscheck begin
        @argcheck bvh.tree.levels >= start_level >= bvh.built_level
    end

    # Get index type from exemplar
    I = get_index_type(options)

    # No contacts / traversal for a single node
    if bvh.tree.real_nodes <= 1
        return BVHTraversal(start_level, 0, 0,
                            similar(bvh.nodes, IndexPair{I}, 0),
                            similar(bvh.nodes, IndexPair{I}, 0))
    end

    # Allocate and add all possible BVTT contact pairs to start with
    bvtt1, bvtt2, num_bvtt = initial_bvtt(bvh, start_level, cache, options)
    num_checks = num_bvtt

    # Known at compile time, even though branches have different types
    extra = if bvtt1 isa AbstractGPUVector
        # For GPUs we need an additional global offset to coordinate writing results
        backend = get_backend(bvtt1)
        KernelAbstractions.zeros(backend, I, Int(bvh.tree.levels))
    else
        # For CPUs we need a contact counter for each task
        Vector{Int}(undef, options.num_threads)
    end

    level = start_level
    while level < bvh.tree.levels
        # We can have maximum 4 new checks per contact-pair; resize destination BVTT accordingly
        length(bvtt2) < 4 * num_bvtt && resize!(bvtt2, 4 * num_bvtt)

        # Check contacts in bvtt1 and add future checks in bvtt2; only sprout self-checks before
        # second-to-last level as leaf self-checks are pointless
        self_checks = level < bvh.tree.levels - 1
        bvtt1, bvtt2, num_bvtt = traverse_nodes!(bvh, bvtt1, bvtt2,
                                                 num_bvtt, extra, level,
                                                 self_checks, options)
        num_checks += num_bvtt

        # Swap source and destination buffers for next iteration
        bvtt1, bvtt2 = bvtt2, bvtt1
        level += 1
    end

    # Arrived at final leaf level, now populating contact list
    length(bvtt2) < num_bvtt && resize!(bvtt2, num_bvtt)
    bvtt1, bvtt2, num_bvtt = traverse_leaves!(bvh, bvtt1, bvtt2, num_bvtt, extra, options)

    # Return contact list and the other buffer as possible cache
    BVHTraversal(start_level, num_checks, Int(num_bvtt), bvtt2, bvtt1)
end


function initial_bvtt(bvh::BVH, start_level, cache, options)

    # Index type
    I = get_index_type(options)

    # Generate all possible contact checks at the given start_level
    level_nodes = pow2(start_level - 1)
    level_checks = (level_nodes - 1) * level_nodes ÷ 2 + level_nodes

    # If we're not at leaf-level, allocate enough memory for next BVTT expansion
    initial_number = start_level == bvh.tree.levels ? level_checks : 4 * level_checks

    # Reuse cache if given
    if isnothing(cache)
        bvtt1 = similar(bvh.nodes, IndexPair{I}, initial_number)
        bvtt2 = similar(bvh.nodes, IndexPair{I}, initial_number)
    else
        @argcheck eltype(cache.cache1) === IndexPair{I}
        @argcheck eltype(cache.cache2) === IndexPair{I}

        bvtt1 = cache.cache1
        bvtt2 = cache.cache2

        length(bvtt1) < initial_number && resize!(bvtt1, initial_number)
        length(bvtt2) < initial_number && resize!(bvtt2, initial_number)
    end

    # Number of real nodes on this level
    num_levels = bvh.tree.levels
    num_real = level_nodes - bvh.tree.virtual_leaves >> (bvh.tree.levels - start_level)

    # Insert all checks to do at this level
    num_bvtt = fill_initial_bvtt_single!(bvtt1, num_levels, start_level, level_nodes, num_real, options)

    bvtt1, bvtt2, num_bvtt
end


function fill_initial_bvtt_single!(bvtt1, num_levels, start_level, level_nodes, num_real, options)
    # Add initial checks; see the CPU branch for the simple code; triangular for loops like this
    # can be linearised for GPU one-per-thread access
    backend = get_backend(bvtt1)
    if backend isa GPU

        # Convert linear index k to upper triangular (i, j) indices for a matrix of side n; 0-index
        function tri_ij(n::I, k::I) where I <: Integer
            a = Float32(-8 * k + 4 * n * (n - 1) - 7)
            b = unsafe_trunc(I, sqrt(a) / 2.0f0 - 0.5f0)
            i = n - 2 - b
            j = k + i + 1 - n * (n - 1) ÷ 2 + (n - i) * ((n - i) - 1) ÷ 2
            (i, j)
        end

        # Matrix side and number of linear indices in upper triangular side
        n = num_real
        n_lin = n * (n - 1) ÷ 2

        if start_level != num_levels
            # First add n_lin node-node pair checks, then another n self-checks
            AK.foreachindex(1:n_lin + n, backend, block_size=options.block_size) do i_lin
                if i_lin > n_lin
                    i = i_lin - n_lin - 1 + level_nodes
                    @inbounds bvtt1[i_lin] = (i, i)
                else
                    i, j = tri_ij(n, i_lin - 1)
                    i += level_nodes
                    j += level_nodes
                    @inbounds bvtt1[i_lin] = (i, j)
                end
            end
            num_bvtt = n_lin + n
            return num_bvtt
        else
            # Only node-node pair checks at the last level
            AK.foreachindex(1:n_lin, backend, block_size=options.block_size) do i_lin
                i, j = tri_ij(n, i_lin - 1)

                i += level_nodes
                j += level_nodes

                @inbounds bvtt1[i_lin] = (i, j)
            end
            num_bvtt = n_lin
            return num_bvtt
        end
    else
        # CPU initial checks; this uses such simple instructions that single threading is fastest
        num_bvtt = 0
        @inbounds for i in level_nodes:level_nodes + num_real - 1
            # Only insert self-checks if we still have nodes below us; leaf self-checks are not needed
            if start_level != num_levels
                num_bvtt += 1
                bvtt1[num_bvtt] = (i, i)
            end

            # Node-node pair checks
            for j in i + 1:level_nodes + num_real - 1
                num_bvtt += 1
                bvtt1[num_bvtt] = (i, j)
            end
        end
        return num_bvtt
    end
end


# Traversal implementations
include("traverse_single_cpu.jl")
include("traverse_single_gpu.jl")

