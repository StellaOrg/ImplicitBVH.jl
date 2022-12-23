# File   : bvh_build.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 15.12.2022


using IBVH
using MeshIO
using FileIO

using BenchmarkTools
using Profile
using PProf


# Load mesh and compute bounding spheres for each triangle
mesh = load("cloth_ball70.stl")
bounding_spheres = [BBox{Float32}(tri) for tri in mesh]

# Pre-compile BVH build
bvh = BVH(bounding_spheres, BBox{Float32}, UInt32)

# Benchmark BVH creation including Morton encoding
println("BVH creation including Morton encoding:")
display(@benchmark(BVH(bounding_spheres, BBox{Float32}, UInt32)))

# Benchmark BVH creation including Morton encoding
println("BVH creation with pre-computed Morton codes:")
display(@benchmark(BVH(bvh.leaves, BBox{Float32}, UInt32)))

# Collect a pprof profile
Profile.clear()
@profile BVH(bounding_spheres, BBox{Float32}, UInt32)

# Export pprof profile and open interactive profiling web interface.
pprof()
