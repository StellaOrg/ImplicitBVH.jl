# File   : morton.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 29.06.2023


using IBVH
using MeshIO
using FileIO

using BenchmarkTools
using Profile
using PProf


const LeafType = BSphere{Float32}
const MortonType = UInt32


# Load mesh and compute bounding spheres for each triangle
mesh = load((@__DIR__) * "/xyzrgb_dragon.obj")
@show size(mesh) Threads.nthreads()
bounding_spheres = [LeafType(tri) for tri in mesh]

# Pre-compile bounding volume extrema computation
IBVH.bounding_volumes_extrema(bounding_spheres)
println("Bounding volume extrema:")
display(@benchmark(IBVH.bounding_volumes_extrema(bounding_spheres)))

# Pre-compile morton encoding
mortons = IBVH.morton_encode(bounding_spheres, MortonType)
println("Morton encoding:")
display(@benchmark(IBVH.morton_encode(bounding_spheres, MortonType)))

# Collect a pprof profile of the complete build
Profile.clear()
@profile IBVH.morton_encode(bounding_spheres, MortonType)

# Export pprof profile and open interactive profiling web interface.
pprof(; out="morton.pb.gz")
