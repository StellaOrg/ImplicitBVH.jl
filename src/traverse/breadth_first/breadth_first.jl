struct BFSTraversal <: TraversalAlgorithm end


function default_start_level(bvh::BVH, ::BFSTraversal)::Int
    maximum2(bvh.tree.levels ÷ 2, bvh.built_level)
end


# Single BVH and BVH-BVH traversal
include("traverse_single.jl")
include("traverse_pair.jl")
