# File   : ka_morton.jl
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 07.06.2024


using Random
using Metal: MtlArray

using BenchmarkTools

using ImplicitBVH
using ImplicitBVH: BSphere, BBox

using Profile
using PProf


Random.seed!(0)

num_bvs = 100_000
bvs = map(BSphere{Float32}, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_bvs])
bvs = MtlArray(bvs)

options = BVHOptions(block_size=128, num_threads=8)
@benchmark BVH(bvs, BBox{Float32}, UInt32, 1, options=options)


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

