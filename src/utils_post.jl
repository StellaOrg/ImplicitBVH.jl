"""
Get index type from options or derived data types.

# Methods
    get_index_type(::ImplicitTree{I}) where I
    get_index_type(bvh::BVH)
    get_index_type(options::BVHOptions)

"""
get_index_type(::ImplicitTree{I}) where I = I
get_index_type(bvh::BVH) = get_index_type(bvh.tree)
get_index_type(options::BVHOptions) = typeof(options.index_exemplar)
