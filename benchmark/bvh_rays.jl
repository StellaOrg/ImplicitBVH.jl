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
const FloatType = Float32
const LeafType = BSphere{FloatType}
const NodeType = BBox{FloatType}
const MortonType = UInt32
const IndexType = Int32


# Load mesh and compute bounding spheres for each triangle. Can download mesh from:
# https://github.com/alecjacobson/common-3d-test-models/blob/master/data/xyzrgb_dragon.obj
mesh = load(joinpath(@__DIR__, "xyzrgb_dragon.obj"))
display(mesh)
@show Threads.nthreads()

bounding_spheres = [LeafType(tri) for tri in mesh]

num_rays = 100_000
points = rand(FloatType, 3, num_rays)
directions = rand(FloatType, 3, num_rays)


# For GPU tests
# bounding_spheres = CuArray(bounding_spheres)
# points = CuArray(points)
# directions = CuArray(directions)


# Pre-compile BVH traversal
options = BVHOptions(index=IndexType, morton=DefaultMortonAlgorithm(MortonType))
bvh = BVH(bounding_spheres, NodeType, options=options)
@show traversal = traverse_rays(bvh, points, directions)


# Benchmark BVH ray-tracing
for alg in (LVTTraversal(), BFSTraversal())
    println("BVH traversal with $alg:")
    display(@benchmark traverse_rays(bvh, points, directions, $alg))
    println()
end



# # Collect a pprof profile of the complete build
# Profile.clear()
# @profile begin
#     traverse_rays(bvh, points, directions)
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
