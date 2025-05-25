# File   : bvh_build.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 15.12.2022


using ImplicitBVH
using ImplicitBVH: BSphere, BBox

import AcceleratedKernels as AK

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

# Example single-threaded user code to compute bounding volumes for each triangle in a mesh
function fill_bounding_volumes!(bounding_volumes, mesh)
    AK.foreachindex(bounding_volumes) do i
        @inbounds bounding_volumes[i] = eltype(bounding_volumes)(mesh[i])
    end
end

bounding_spheres = Vector{LeafType}(undef, length(mesh.faces))
display(@benchmark(fill_bounding_volumes!(bounding_spheres, mesh)))


# Collect a pprof profile of the complete build
Profile.clear()
@profile fill_bounding_volumes!(bounding_spheres, mesh)

# Export pprof profile and open interactive profiling web interface.
pprof(; out="bvh_volumes.pb.gz")


# Test for some coding mistakes
# using Test
# Test.detect_unbound_args(ImplicitBVH, recursive = true)
# Test.detect_ambiguities(ImplicitBVH, recursive = true)


# More complete report on type stabilities
# using JET
# JET.@report_opt BVH(bounding_spheres, NodeType, MortonType)


# using Profile
# fill_bounding_volumes!(bounding_spheres, mesh)
# Profile.clear_malloc_data()
# fill_bounding_volumes!(bounding_spheres, mesh)
