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


# Types used
const LeafType = BSphere{Float32}
const NodeType = BBox{Float32}
const MortonType = UInt32


# Load mesh and compute bounding spheres for each triangle. Can download mesh from:
# https://github.com/alecjacobson/common-3d-test-models/blob/master/data/xyzrgb_dragon.obj
mesh = load(joinpath(@__DIR__, "xyzrgb_dragon.obj"))
@show size(mesh)
bounding_spheres = [LeafType(tri) for tri in mesh]

# Pre-compile BVH build
bvh = BVH(bounding_spheres, NodeType, MortonType)

# Benchmark BVH creation including Morton encoding
println("BVH creation including Morton encoding:")
display(@benchmark(BVH(bounding_spheres, NodeType, MortonType)))

# Collect a pprof profile of the complete build
Profile.clear()
@profile BVH(bounding_spheres, NodeType, MortonType)

# Export pprof profile and open interactive profiling web interface.
pprof(; out="bvh_build.pb.gz")
