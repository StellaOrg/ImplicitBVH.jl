using ImplicitBVH
# using GLMakie
using ImplicitBVH: BBox


#included for plotting

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


const NodeType=BBox{Float32}

leaves=[BBox{Float32}((33.0f0, 3.0f0, 33.0f0), (67.0f0, 28.0f0, 53.0f0))
            BBox{Float32}((33.0f0, 3.0f0, 53.0f0), (67.0f0, 28.0f0, 73.0f0))
            BBox{Float32}((33.0f0, 28.0f0, 33.0f0), (67.0f0, 52.0f0, 53.0f0))
            BBox{Float32}((33.0f0, 28.0f0, 53.0f0), (67.0f0, 52.0f0, 73.0f0))
            BBox{Float32}((33.0f0, 52.0f0, 33.0f0), (67.0f0, 77.0f0, 53.0f0))
            BBox{Float32}((33.0f0, 52.0f0, 53.0f0), (67.0f0, 77.0f0, 73.0f0))
            BBox{Float32}((33.0f0, 77.0f0, 33.0f0), (67.0f0, 102.0f0, 53.0f0))
            BBox{Float32}((33.0f0, 77.0f0, 53.0f0), (67.0f0, 102.0f0, 73.0f0))]

bvh_test=BVH(leaves, NodeType)
@show bvh_test
@show bvh_test.order


stop

begin
    fig = Figure(size = (1200, 800))
    ax = Axis3(fig[1, 1])
    
    lines!(ax, boxes_lines(bvh_test.leaves), linewidth = 2, color = "grey", linestyle=:dash)
    lines!(ax, boxes_lines([bvh_test.nodes[1]]), linewidth = 2, color = "green")
    lines!(ax, boxes_lines([bvh_test.nodes[4]]), linewidth = 5, color = "orange",linestyle=:solid)
    
    lines!(ax, boxes_lines([bvh_test.leaves[1]]), linewidth = 5, color = "blue",linestyle=:solid) # is located inside node 4
    lines!(ax, boxes_lines([bvh_test.leaves[2]]), linewidth = 5, color = "blue",linestyle=:solid) # should be located inside node 4
    
    lines!(ax, boxes_lines([bvh_test.leaves[3]]), linewidth = 5, color = "black",linestyle=:dash) # is actually located inside node 4
    

    fig
end