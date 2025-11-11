struct LVTTraversal <: TraversalAlgorithm end


function default_start_level(bvh::BVH, ::LVTTraversal)::Int
    maximum2(1, bvh.built_level)
end


# Single BVH and BVH-BVH traversal
include("traverse_single.jl")
include("traverse_pair.jl")

