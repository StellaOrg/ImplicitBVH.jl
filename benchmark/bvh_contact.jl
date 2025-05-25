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

# Pre-compile BVH traversal
bvh = BVH(bounding_spheres, NodeType, MortonType)
@show traversal = traverse(bvh)

# Print algorithmic efficiency
eff = traversal.num_checks / (length(bounding_spheres) * length(bounding_spheres) / 2)
println("Did $eff of the total checks needed for brute-force contact detection")

# Benchmark BVH traversal anew
println("BVH traversal with dynamic buffer resizing:")
display(@benchmark(traverse(bvh), samples=100))

# Benchmark BVH creation reusing previous cache
println("BVH traversal without dynamic buffer resizing:")
display(@benchmark(traverse(bvh, bvh.tree.levels ÷ 2, traversal), samples=100))

# Collect a pprof profile
# Profile.clear()
# @profile begin
#     for _ in 1:1000
#         traverse(bvh, bvh.tree.levels ÷ 2, traversal)
#     end
# end

# Export pprof profile and open interactive profiling web interface.
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
