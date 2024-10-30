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


function original(level_nodes1=8, num_real1=7, level_nodes2=4, num_real2=3)

    level_checks = num_real1 * num_real2
    bvtt1 = Vector{Tuple{Int, Int}}(undef, level_checks)

    options = ImplicitBVH.BVHOptions()
    ImplicitBVH.fill_initial_bvtt_pair!(
        bvtt1, level_nodes1, num_real1, level_nodes2, num_real2, options,
    )

    bvtt1
end


function linearised(level_nodes1=8, num_real1=7, level_nodes2=4, num_real2=3)

    level_checks = num_real1 * num_real2
    bvtt1 = oneArray{Tuple{Int, Int}}(undef, level_checks)

    options = ImplicitBVH.BVHOptions()
    ImplicitBVH.fill_initial_bvtt_pair!(
        bvtt1, level_nodes1, num_real1, level_nodes2, num_real2, options,
    )

    bvtt1
end



# v1 = original()
# v2 = linearised()


Random.seed!(0)

num_bvs = 100_000
bvs = map(BSphere{Float32}, [100 * rand(3) .+ rand(3, 3) for _ in 1:num_bvs])
bvs = oneArray(bvs)

options = BVHOptions(block_size=128, num_threads=8)
bvh = BVH(bvs, BBox{Float32}, UInt32, 1, options=options)
# @benchmark traverse(bvh)
bvt = traverse(bvh, bvh)


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

