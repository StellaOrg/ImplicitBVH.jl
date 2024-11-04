# File   : bvh_build.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 15.12.2022


using ImplicitBVH
using ImplicitBVH: BSphere, BBox

using Random
Random.seed!(42)

using MeshIO
using FileIO

using BenchmarkTools
using Profile
using PProf


# Types used
const LeafType = BBox{Float32}



num_bvs = 1_000_000

bvs = [LeafType(rand(3, 3)) for _ in 1:num_bvs]
points = rand(Float32, 3, num_bvs)
directions = rand(Float32, 3, num_bvs)


# Example usage
function check_intersections!(intersections, bvs, points, directions)
    @inbounds for i in eachindex(intersections)
        intersections[i] = isintersection(bvs[i], @view(points[:, i]), @view(directions[:, i]))
    end
    nothing
end

intersections = Vector{Bool}(undef, length(bvs))
check_intersections!(intersections, bvs, points, directions)


println("Benchmarking $(length(bvs)) intersections:")
display(@benchmark check_intersections!(intersections, bvs, points, directions))


# # Collect a pprof profile of the complete build
# Profile.clear()
# @profile begin
#     for _ in 1:1000
#         check_intersections!(intersections, bvs, points, directions)
#     end
# end
# 
# 
# # Export pprof profile and open interactive profiling web interface.
# pprof(; out="bvh_rays.pb.gz")


# Test for some coding mistakes
using Test
Test.detect_unbound_args(ImplicitBVH, recursive = true)
Test.detect_ambiguities(ImplicitBVH, recursive = true)


# More complete report on type stabilities
using JET
JET.@report_opt check_intersections!(intersections, bvs, points, directions)


# using Profile
# BVH(bounding_spheres, NodeType, MortonType)
# Profile.clear_malloc_data()
# BVH(bounding_spheres, NodeType, MortonType)
