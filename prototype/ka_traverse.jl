# File   : ka_morton.jl
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 07.06.2024


using Random
using oneAPI: oneArray
import AcceleratedKernels as AK

using BenchmarkTools

using ImplicitBVH
using ImplicitBVH: BSphere, BBox

using Profile
using PProf


function original(start_level, num_levels, num_real=1 << (num_levels - 1))

    @assert start_level <= num_levels

    # Generate all possible contact checks at the given start_level
    level_nodes = 2^(start_level - 1)
    level_checks = (level_nodes - 1) * level_nodes ÷ 2 + level_nodes

    # If we're not at leaf-level, allocate enough memory for next BVTT expansion
    initial_number = start_level == num_levels ? level_checks : 4 * level_checks
    bvtt1 = Vector{Tuple{Int, Int}}(undef, initial_number)

    # Insert all checks at this level
    num_bvtt = 0
    # num_real = level_nodes - bvh.tree.virtual_leaves >> (bvh.tree.levels - start_level)

    # num_levels = bvh.tree.levels

    for i in level_nodes:level_nodes + num_real - 1
        # Node-node pair checks
        for j in i + 1:level_nodes + num_real - 1
            num_bvtt += 1
            bvtt1[num_bvtt] = (i, j)
        end
    end

    # Only insert self-checks if we still have nodes below us; leaf self-checks are not needed
    if start_level != num_levels
        for i in level_nodes:level_nodes + num_real - 1
            num_bvtt += 1
            bvtt1[num_bvtt] = (i, i)
        end
    end

    @show num_bvtt

    bvtt1
end


@fastmath function trinum(n::I, k::I) where I <: Integer
    a = Float32(-8 * k + 4 * n * (n - 1) - 7)
    b = unsafe_trunc(I, sqrt(a) / 2.0f0 - 0.5f0)
    i = n - 2 - b
    j = k + i + 1 - n * (n - 1) ÷ 2 + (n - i) * ((n - i) - 1) ÷ 2
    (i, j)
end


function test_trinum(n)
    N = n * (n - 1) ÷ 2
    v = Array{Tuple{Int, Int}}(undef, N)
    AK.foreachindex(1:N, get_backend(v)) do i
        v[i] = trinum(n, i - 1)
    end

    v
end


function linearised(start_level, num_levels, num_real=1 << (num_levels - 1))

    @assert start_level <= num_levels

    # Generate all possible contact checks at the given start_level
    level_nodes = 2^(start_level - 1)
    level_checks = (level_nodes - 1) * level_nodes ÷ 2 + level_nodes

    # If we're not at leaf-level, allocate enough memory for next BVTT expansion
    initial_number = start_level == num_levels ? level_checks : 4 * level_checks
    bvtt1 = Vector{Tuple{Int, Int}}(undef, initial_number)

    n = num_real
    n_lin = n * (n - 1) ÷ 2

    if start_level != num_levels
        AK.foreachindex(1:n_lin + n, get_backend(bvtt1)) do i_lin
            if i_lin > n_lin
                i = i_lin - n_lin - 1 + level_nodes
                bvtt1[i_lin] = (i, i)
            else
                i, j = trinum(n, i_lin - 1)
                i += level_nodes
                j += level_nodes
                bvtt1[i_lin] = (i, j)
            end
        end
        @show num_bvtt = n_lin + n
    else
        AK.foreachindex(1:n_lin, get_backend(bvtt1)) do i_lin
            i, j = trinum(n, i_lin - 1)

            i += level_nodes
            j += level_nodes

            bvtt1[i_lin] = (i, j)
        end
        @show num_bvtt = n_lin
    end

    bvtt1
end


function check_isvirtual(tree)

    v = oneArray{Bool}(undef, tree.real_nodes)
    AK.foreachindex(v) do i
        v[i] = @inbounds ImplicitBVH.isvirtual(tree, i)
    end

    v
end


# Test initial filling of array with linearised upper triangular matrix
# v1 = original(4, 5)
# v2 = linearised(4, 5)


Random.seed!(0)

num_bvs = 100_000
bvs = map(BSphere{Float32}, [100 * rand(3) .+ rand(3, 3) for _ in 1:num_bvs])
bvs = oneArray(bvs)

options = BVHOptions(block_size=128, num_threads=8)
bvh = BVH(bvs, BBox{Float32}, UInt32, 1, options=options)
# @benchmark traverse(bvh)
bvt = traverse(bvh)


# check_isvirtual(bvh.tree)


# # Collect a profile
# Profile.clear()
# @profile begin
#     for _ in 1:100
#         ImplicitBVH.morton_encode!(mortons, bvs, mins, maxs, options)
#     end
# end
# 
# # Export pprof profile and open interactive profiling web interface.
# pprof()

