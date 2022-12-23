# File   : bvh_contact.jl
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

# Pre-compile BVH traversal
bvh = BVH(bounding_spheres, BBox{Float32}, UInt32)
traversal = traverse(bvh)

# Benchmark BVH creation including Morton encoding
println("BVH traversal with dynamic buffer resizing:")
display(@benchmark(traverse(bvh)))

# Benchmark BVH creation including Morton encoding
println("BVH traversal without dynamic buffer resizing:")
display(@benchmark(traverse(bvh, bvh.tree.levels ÷ 2, traversal)))

# Collect a pprof profile
Profile.clear()
@profile BVH(bounding_spheres, BBox{Float32}, UInt32)

# Export pprof profile and open interactive profiling web interface.
pprof()
