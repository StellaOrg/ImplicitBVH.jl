# File   : bvh_build.jl
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
options = BVHOptions(index=IndexType, morton=DefaultMortonAlgorithm(MortonType))


# Load mesh and compute bounding spheres for each triangle. Can download mesh from:
# https://github.com/alecjacobson/common-3d-test-models/blob/master/data/xyzrgb_dragon.obj
mesh = load(joinpath(@__DIR__, "xyzrgb_dragon.obj"))
display(mesh)

bounding_spheres = [LeafType(tri) for tri in mesh]
# bounding_spheres = CuArray(bounding_spheres)

# Pre-compile BVH build
bvh = BVH(bounding_spheres, NodeType, options=options)

# Benchmark BVH creation including Morton encoding
println("BVH creation including Morton encoding:")
display(@benchmark BVH(bounding_spheres, NodeType, options=options))

println("BVH with cached memory reuse:")
display(@benchmark BVH(bvh.leaves, NodeType, cache=bvh, options=options))


# # Collect a pprof profile of the complete build
# BVH(bvh.leaves, NodeType, options=options)
# Profile.clear()
# @profile begin
#     for _ in 1:100
#         BVH(bounding_spheres, NodeType, options=options)
#     end
# end
# 
# # Export pprof profile and open interactive profiling web interface.
# pprof(; out="bvh_build.pb.gz")


# Test for some coding mistakes
# using Test
# Test.detect_unbound_args(ImplicitBVH, recursive = true)
# Test.detect_ambiguities(ImplicitBVH, recursive = true)


# More complete report on type stabilities
# using JET
# JET.@report_opt BVH(bounding_spheres, NodeType, MortonType)


# using Profile
# BVH(bounding_spheres, NodeType, MortonType)
# Profile.clear_malloc_data()
# BVH(bounding_spheres, NodeType, MortonType)
