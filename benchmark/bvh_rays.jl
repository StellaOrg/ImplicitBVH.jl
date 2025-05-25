# File   : bvh_contact.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 15.12.2022


using ImplicitBVH
using ImplicitBVH: BSphere, BBox

using MeshIO
using FileIO

using BenchmarkTools
using Profile
using PProf

# using CUDA: CuArray


# Types used
const LeafType = BSphere{Float32}
const NodeType = BBox{Float32}
const MortonType = UInt32


# Load mesh and compute bounding spheres for each triangle. Can download mesh from:
# https://github.com/alecjacobson/common-3d-test-models/blob/master/data/xyzrgb_dragon.obj
mesh = load(joinpath(@__DIR__, "xyzrgb_dragon.obj"))
display(mesh)
@show Threads.nthreads()

bounding_spheres = [LeafType(tri) for tri in mesh]

num_rays = 100_000
points = rand(Float32, 3, num_rays)
directions = rand(Float32, 3, num_rays)


# For GPU tests
# bounding_spheres = CuArray(bounding_spheres)
# points = CuArray(points)
# directions = CuArray(directions)


# Pre-compile BVH traversal
bvh = BVH(bounding_spheres, NodeType, MortonType)
@show traversal = traverse_rays(bvh, points, directions)


# # Benchmark BVH traversal anew
# println("BVH traversal with dynamic buffer resizing:")
# display(@benchmark(traverse_rays(bvh, points, directions), samples=100))

# Benchmark BVH creation reusing previous cache
println("BVH traversal without dynamic buffer resizing:")
display(@benchmark(traverse_rays(bvh, points, directions, 1, traversal), samples=100))



# # Collect a pprof profile of the complete build
# Profile.clear()
# @profile begin
#     for _ in 1:1000
#         check_intersections!(intersections, bvs, points, directions)
#     end
# end
# 
# 
# # Export pprof profile and open interactive profiling web interface.
# pprof(; out="bvh_rays.pb.gz")


# # Test for some coding mistakes
# using Test
# Test.detect_unbound_args(ImplicitBVH, recursive = true)
# Test.detect_ambiguities(ImplicitBVH, recursive = true)


# # More complete report on type stabilities
# using JET
# JET.@report_opt check_intersections!(intersections, bvs, points, directions)


# using Profile
# BVH(bounding_spheres, NodeType, MortonType)
# Profile.clear_malloc_data()
# BVH(bounding_spheres, NodeType, MortonType)
