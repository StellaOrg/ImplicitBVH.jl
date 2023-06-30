# File   : bvh_contact.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 15.12.2022


using IBVH
using IBVH: BSphere, BBox

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
@show size(mesh) Threads.nthreads()
bounding_spheres = [LeafType(tri) for tri in mesh]

# Pre-compile BVH traversal
bvh = BVH(bounding_spheres, NodeType, MortonType)
traversal = traverse(bvh)
@show bvh.stats

# Print algorithmic efficiency
eff = bvh.stats.num_checks / (length(bounding_spheres) * length(bounding_spheres) / 2)
println("Did $eff of the total checks needed for brute-force contact detection")

# Benchmark BVH creation including Morton encoding
println("BVH traversal with dynamic buffer resizing:")
display(@benchmark(traverse(bvh)))

# Benchmark BVH creation including Morton encoding
println("BVH traversal without dynamic buffer resizing:")
display(@benchmark(traverse(bvh, bvh.tree.levels ÷ 2, traversal)))

# Collect a pprof profile
Profile.clear()
@profile traverse(bvh)

# Export pprof profile and open interactive profiling web interface.
pprof(; out="bvh_contact.pb.gz")
