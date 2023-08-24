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

using GLMakie


# Types used
const LeafType = BSphere{Float32}
const NodeType = BBox{Float32}
const MortonType = UInt32


# Load mesh and compute bounding spheres for each triangle. Can download mesh from:
# https://github.com/alecjacobson/common-3d-test-models/blob/master/data/xyzrgb_dragon.obj
mesh = load(joinpath(@__DIR__, "stanford-bunny.obj"))
bounding_spheres = [LeafType(tri) for tri in mesh]

# Pre-compile BVH traversal
bvh = BVH(bounding_spheres, NodeType, MortonType)
@show traversal = traverse(bvh)


function box_lines!(lines, lo, up)
    # Write lines forming an axis-aligned box from lo to up
    @assert ndims(lines) == 2
    @assert size(lines) == (24, 3)

    lines[1:24, 1:3] .= [
        # Bottom sides
        lo[1] lo[2] lo[3]
        up[1] lo[2] lo[3]
        up[1] up[2] lo[3]
        lo[1] up[2] lo[3]
        lo[1] lo[2] lo[3]
        NaN NaN NaN

        # Vertical sides
        lo[1] lo[2] lo[3]
        lo[1] lo[2] up[3]
        NaN NaN NaN

        up[1] lo[2] lo[3]
        up[1] lo[2] up[3]
        NaN NaN NaN

        up[1] up[2] lo[3]
        up[1] up[2] up[3]
        NaN NaN NaN

        lo[1] up[2] lo[3]
        lo[1] up[2] up[3]
        NaN NaN NaN

        # Top sides
        lo[1] lo[2] up[3]
        up[1] lo[2] up[3]
        up[1] up[2] up[3]
        lo[1] up[2] up[3]
        lo[1] lo[2] up[3]
        NaN NaN NaN
    ]

    nothing
end


function boxes_lines(boxes)
    # Create contiguous matrix of lines representing boxes
    lines = Matrix{Float64}(undef, 24 * length(boxes), 3)
    for i in axes(boxes, 1)
        box_lines!(view(lines, 24 * (i - 1) + 1:24i, 1:3), boxes[i].lo, boxes[i].up)
    end
    lines
end


# Plot a wireframe of the mesh and the bounding boxes above leaf level
fig, ax = wireframe(
    mesh,
    color = [tri[1][2] for tri in mesh for i in 1:3],
    colormap=:Spectral,
    ssao=true,
)
lines!(ax, boxes_lines(bvh.nodes), linewidth=0.5)
fig
