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

# using Metal: MtlArray


# Types used
const FloatType = Float32
const LeafType = BSphere{FloatType}
const NodeType = BBox{FloatType}
const MortonType = UInt32
const IndexType = Int32
const alg = LVTTraversal()
options = BVHOptions(index=IndexType, morton=DefaultMortonAlgorithm(MortonType))


# Load mesh and compute bounding spheres for each triangle. Can download mesh from:
# https://github.com/alecjacobson/common-3d-test-models/blob/master/data/xyzrgb_dragon.obj
mesh = load(joinpath(@__DIR__, "xyzrgb_dragon.obj"))
display(mesh)
@show Threads.nthreads()

bounding_spheres = [LeafType(tri) for tri in mesh]
# bounding_spheres = MtlArray(bounding_spheres)

# Pre-compile BVH traversal
bvh = BVH(bounding_spheres, NodeType, options=options)
@show traversal = traverse(bvh, alg)

# Benchmark BVH traversal anew
println("BVH traversal:")
display(@benchmark traverse(bvh, alg))


# # Collect a pprof profile
# Profile.clear()
# @profile begin
#     for _ in 1:10
#         traverse(bvh, alg)
#     end
# end
# 
# # Export pprof profile and open interactive profiling web interface.
# pprof(; out="bvh_contact.pb.gz")


# Test for some coding mistakes
# using Test
# Test.detect_unbound_args(ImplicitBVH, recursive = true)
# Test.detect_ambiguities(ImplicitBVH, recursive = true)


# More complete report on type stabilities
# using JET
# JET.@report_opt traverse(bvh)


# using Profile
# traverse(bvh, bvh.tree.levels ÷ 2, traversal)
# Profile.clear_malloc_data()
# traverse(bvh, bvh.tree.levels ÷ 2, traversal)
