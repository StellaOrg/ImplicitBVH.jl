var documenterSearchIndex = {"docs":
[{"location":"morton/#Morton-Encoding","page":"Morton Encoding","title":"Morton Encoding","text":"","category":"section"},{"location":"morton/","page":"Morton Encoding","title":"Morton Encoding","text":"ImplicitBVH.MortonUnsigned\nImplicitBVH.morton_encode\nImplicitBVH.morton_encode!\nImplicitBVH.morton_encode_single\nImplicitBVH.morton_scaling\nImplicitBVH.morton_split3\nImplicitBVH.bounding_volumes_extrema\nImplicitBVH.relative_precision","category":"page"},{"location":"morton/#ImplicitBVH.MortonUnsigned","page":"Morton Encoding","title":"ImplicitBVH.MortonUnsigned","text":"Acceptable unsigned integer types for Morton encoding: Union{UInt16, UInt32, UInt64}.\n\n\n\n\n\n","category":"type"},{"location":"morton/#ImplicitBVH.morton_encode","page":"Morton Encoding","title":"ImplicitBVH.morton_encode","text":"morton_encode(\n    bounding_volumes,\n    ::Type{U}=UInt,\n    options=BVHOptions(),\n) where {U <: MortonUnsigned}\n\nEncode the centers of some bounding_volumes as Morton codes of type U <: MortonUnsigned. See morton_encode! for full details. \n\n\n\n\n\n","category":"function"},{"location":"morton/#ImplicitBVH.morton_encode!","page":"Morton Encoding","title":"ImplicitBVH.morton_encode!","text":"morton_encode!(\n    mortons::AbstractVector{U},\n    bounding_volumes,\n    options=BVHOptions(),\n) where {U <: MortonUnsigned}\n\nmorton_encode!(\n    mortons::AbstractVector{U},\n    bounding_volumes::AbstractVector,\n    mins,\n    maxs,\n    options=BVHOptions(),\n) where {U <: MortonUnsigned}\n\nEncode each bounding volume into vector of corresponding Morton codes such that they uniformly cover the maximum Morton range given an unsigned integer type U <: MortonUnsigned.\n\nwarning: Warning\nThe dimension-wise exclusive mins and maxs must be correct; if any bounding volume center is equal to, or beyond mins / maxs, the results will be silently incorrect.\n\n\n\n\n\n","category":"function"},{"location":"morton/#ImplicitBVH.morton_encode_single","page":"Morton Encoding","title":"ImplicitBVH.morton_encode_single","text":"morton_encode_single(centre, mins, maxs, U::MortonUnsignedType=UInt32)\n\nReturn Morton code for a single 3D position centre scaled uniformly between mins and maxs. Works transparently for SVector, Vector, etc. with eltype UInt16, UInt32 or UInt64.\n\n\n\n\n\n","category":"function"},{"location":"morton/#ImplicitBVH.morton_scaling","page":"Morton Encoding","title":"ImplicitBVH.morton_scaling","text":"morton_scaling(::Type{UInt16}) = 2^5\nmorton_scaling(::Type{UInt32}) = 2^10\nmorton_scaling(::Type{UInt64}) = 2^21\n\nExclusive maximum number possible to use for 3D Morton encoding for each type.\n\n\n\n\n\n","category":"function"},{"location":"morton/#ImplicitBVH.morton_split3","page":"Morton Encoding","title":"ImplicitBVH.morton_split3","text":"morton_split3(v::UInt16)\nmorton_split3(v::UInt32)\nmorton_split3(v::UInt64)\n\nShift a number's individual bits such that they have two zeros between them.\n\n\n\n\n\n","category":"function"},{"location":"morton/#ImplicitBVH.bounding_volumes_extrema","page":"Morton Encoding","title":"ImplicitBVH.bounding_volumes_extrema","text":"bounding_volumes_extrema(bounding_volumes)\n\nCompute exclusive lower and upper bounds in iterable of bounding volumes, e.g. Vector{BBox}.\n\n\n\n\n\n","category":"function"},{"location":"morton/#ImplicitBVH.relative_precision","page":"Morton Encoding","title":"ImplicitBVH.relative_precision","text":"relative_precision(::Type{Float16}) = 1e-2\nrelative_precision(::Type{Float32}) = 1e-5\nrelative_precision(::Type{Float64}) = 1e-14\n\nRelative precision value for floating-point types.\n\n\n\n\n\n","category":"function"},{"location":"bounding_volumes/#Bounding-Volumes","page":"Bounding Volumes","title":"Bounding Volumes","text":"","category":"section"},{"location":"bounding_volumes/","page":"Bounding Volumes","title":"Bounding Volumes","text":"ImplicitBVH.BBox\nImplicitBVH.BSphere","category":"page"},{"location":"bounding_volumes/#ImplicitBVH.BBox","page":"Bounding Volumes","title":"ImplicitBVH.BBox","text":"struct BBox{T}\n\nAxis-aligned bounding box, highly optimised for computing bounding volumes for triangles and merging into larger bounding volumes.\n\nCan also be constructed from two spheres to e.g. allow merging BSphere leaves into BBox nodes.\n\nMethods\n\n# Convenience constructors\nBBox(lo::NTuple{3, T}, up::NTuple{3, T}) where T\nBBox{T}(lo::AbstractVector, up::AbstractVector) where T\nBBox(lo::AbstractVector, up::AbstractVector)\n\n# Construct from triangle vertices\nBBox{T}(p1, p2, p3) where T\nBBox(p1, p2, p3)\nBBox{T}(vertices::AbstractMatrix) where T\nBBox(vertices::AbstractMatrix)\nBBox{T}(triangle) where T\nBBox(triangle)\n\n# Merging bounding boxes\nBBox{T}(a::BBox, b::BBox) where T\nBBox(a::BBox{T}, b::BBox{T}) where T\nBase.:+(a::BBox, b::BBox)\n\n# Merging bounding spheres\nBBox{T}(a::BSphere{T}) where T\nBBox(a::BSphere{T}) where T\nBBox{T}(a::BSphere{T}, b::BSphere{T}) where T\nBBox(a::BSphere{T}, b::BSphere{T}) where T\n\n\n\n\n\n","category":"type"},{"location":"bounding_volumes/#ImplicitBVH.BSphere","page":"Bounding Volumes","title":"ImplicitBVH.BSphere","text":"struct BSphere{T}\n\nBounding sphere, highly optimised for computing bounding volumes for triangles and merging into larger bounding volumes.\n\nMethods\n\n# Convenience constructors\nBSphere(x::NTuple{3, T}, r)\nBSphere{T}(x::AbstractVector, r) where T\nBSphere(x::AbstractVector, r)\n\n# Construct from triangle vertices\nBSphere{T}(p1, p2, p3) where T\nBSphere(p1, p2, p3)\nBSphere{T}(vertices::AbstractMatrix) where T\nBSphere(vertices::AbstractMatrix)\nBSphere{T}(triangle) where T\nBSphere(triangle)\n\n# Merging bounding volumes\nBSphere{T}(a::BSphere, b::BSphere) where T\nBSphere(a::BSphere{T}, b::BSphere{T}) where T\nBase.:+(a::BSphere, b::BSphere)\n\n\n\n\n\n","category":"type"},{"location":"bounding_volumes/#Query-Functions","page":"Bounding Volumes","title":"Query Functions","text":"","category":"section"},{"location":"bounding_volumes/","page":"Bounding Volumes","title":"Bounding Volumes","text":"ImplicitBVH.iscontact\nImplicitBVH.center","category":"page"},{"location":"bounding_volumes/#ImplicitBVH.iscontact","page":"Bounding Volumes","title":"ImplicitBVH.iscontact","text":"iscontact(a::BSphere, b::BSphere)\niscontact(a::BBox, b::BBox)\niscontact(a::BSphere, b::BBox)\niscontact(a::BBox, b::BSphere)\n\nCheck if two bounding volumes are touching or inter-penetrating.\n\n\n\n\n\n","category":"function"},{"location":"bounding_volumes/#ImplicitBVH.center","page":"Bounding Volumes","title":"ImplicitBVH.center","text":"center(b::BSphere)\ncenter(b::BBox{T}) where T\n\nGet the coordinates of a bounding volume's centre, as a NTuple{3, T}.\n\n\n\n\n\n","category":"function"},{"location":"bounding_volumes/#Miscellaneous","page":"Bounding Volumes","title":"Miscellaneous","text":"","category":"section"},{"location":"bounding_volumes/","page":"Bounding Volumes","title":"Bounding Volumes","text":"ImplicitBVH.translate","category":"page"},{"location":"bounding_volumes/#ImplicitBVH.translate","page":"Bounding Volumes","title":"ImplicitBVH.translate","text":"translate(b::BSphere{T}, dx) where T\ntranslate(b::BBox{T}, dx) where T\n\nGet a new bounding volume translated by dx; dx can be any iterable with 3 elements.\n\n\n\n\n\n","category":"function"},{"location":"implicit_tree/#Implicit-Binary-Tree","page":"Implicit Binary Tree","title":"Implicit Binary Tree","text":"","category":"section"},{"location":"implicit_tree/","page":"Implicit Binary Tree","title":"Implicit Binary Tree","text":"ImplicitTree\nmemory_index\nlevel_indices\nisvirtual","category":"page"},{"location":"implicit_tree/#ImplicitBVH.ImplicitTree","page":"Implicit Binary Tree","title":"ImplicitBVH.ImplicitTree","text":"struct ImplicitTree{T<:Integer}\n\nImplicit binary tree for num_leaves elements, where nodes are labelled according to a breadth-first search.\n\nMethods\n\nImplicitTree(num_leaves::Integer)\nImplicitTree{T}(num_leaves::Integer)\n\nFields\n\nlevels::Integer: Number of levels in the tree.\nreal_leaves::Integer: Number of real leaves - i.e. the elements from which the tree was constructed.\nreal_nodes::Integer: Total number of real nodes in tree.\nvirtual_leaves::Integer: Number of virtual leaves needed at the bottom level to have a perfect binary tree.\nvirtual_nodes::Integer: Total number of virtual nodes in tree needed for a complete binary tree.\n\nExamples\n\njulia> using ImplicitBVH\n\n# Given 5 geometric elements (e.g. bounding boxes) we construct the following implicit tree\n# having the 5 real leaves at implicit indices 8-12 plus 3 virtual leaves.\n#         Nodes & Leaves                Tree Level\n#               1                       1\n#       2               3               2\n#   4       5       6        7v         3\n# 8   9   10 11   12 13v  14v  15v      4\njulia> tree = ImplicitTree(5)\nImplicitTree{Int64}\n  levels: Int64 4\n  real_leaves: Int64 5\n  real_nodes: Int64 11\n  virtual_leaves: Int64 3\n  virtual_nodes: Int64 4\n\n# We can keep all tree nodes in a contiguous vector with no extra padding for the virtual\n# nodes by computing the real memory index of real nodes; e.g. real memory index of node 8\n# skips node 7 which is virtual:\njulia> memory_index(tree, 8)\n7\n\n# We can get the range of indices of real nodes on a given level\njulia> level_indices(tree, 3)\n(4, 6)\n\n# And we can check if a node at a given implicit index is virtual\njulia> isvirtual(tree, 6)\nfalse\n\njulia> isvirtual(tree, 7)\ntrue\n\n\n\n\n\n","category":"type"},{"location":"implicit_tree/#ImplicitBVH.memory_index","page":"Implicit Binary Tree","title":"ImplicitBVH.memory_index","text":"memory_index(tree::ImplicitTree, implicit_index::Integer)\n\nReturn actual memory index for a node at implicit index i in a perfect BFS-labelled tree.\n\n\n\n\n\n","category":"function"},{"location":"implicit_tree/#ImplicitBVH.level_indices","page":"Implicit Binary Tree","title":"ImplicitBVH.level_indices","text":"level_indices(tree::ImplicitTree, level::Integer)\n\nReturn range Tuple{Int64, Int64} of memory indices of elements at level.\n\n\n\n\n\n","category":"function"},{"location":"implicit_tree/#ImplicitBVH.isvirtual","page":"Implicit Binary Tree","title":"ImplicitBVH.isvirtual","text":"isvirtual(tree::ImplicitTree, implicit_index::Integer)\n\nCheck if given implicit_index corresponds to a virtual node.\n\n\n\n\n\n","category":"function"},{"location":"utilities/#Utilities","page":"Utilities","title":"Utilities","text":"","category":"section"},{"location":"utilities/","page":"Utilities","title":"Utilities","text":"ImplicitBVH.BVHOptions\nImplicitBVH.get_index_type\nImplicitBVH.TaskPartitioner","category":"page"},{"location":"utilities/#ImplicitBVH.BVHOptions","page":"Utilities","title":"ImplicitBVH.BVHOptions","text":"struct BVHOptions{I<:Integer, T}\n\nOptions for building and traversing bounding volume hierarchies, including parallel strategy settings.\n\nAn exemplar of an index (e.g. Int32(0)) is used to deduce the types of indices used in the BVH building (ImplicitTree(@ref), order) and traversal (IndexPair(@ref)).\n\nThe CPU scheduler can be :threads (for base Julia threads) or :polyester (for Polyester.jl threads).\n\nIf compute_extrema=false and mins / maxs are defined, they will not be computed from the distribution of bounding volumes; useful if you have a fixed simulation box, for example.\n\nMethods\n\nBVHOptions(;\n\n    # Example index from which to deduce type\n    index_exemplar::I               = Int32(0),\n\n    # CPU threading\n    scheduler::Symbol               = :threads,\n    num_threads::Int                = Threads.nthreads(),\n    min_mortons_per_thread::Int     = 1000,\n    min_boundings_per_thread::Int   = 1000,\n    min_traversals_per_thread::Int  = 1000,\n\n    # GPU scheduling\n    block_size::Int                 = 256,\n\n    # Minima / maxima\n    compute_extrema::Bool           = true,\n    mins::NTuple{3, T}              = (NaN32, NaN32, NaN32),\n    maxs::NTuple{3, T}              = (NaN32, NaN32, NaN32),\n) where {I <: Integer, T}\n\nFields\n\nindex_exemplar::Integer\nscheduler::Symbol\nnum_threads::Int64\nmin_mortons_per_thread::Int64\nmin_boundings_per_thread::Int64\nmin_traversals_per_thread::Int64\nblock_size::Int64\ncompute_extrema::Bool\nmins::Tuple{T, T, T} where T\nmaxs::Tuple{T, T, T} where T\n\n\n\n\n\n","category":"type"},{"location":"utilities/#ImplicitBVH.get_index_type","page":"Utilities","title":"ImplicitBVH.get_index_type","text":"Get index type from options or derived data types.\n\nMethods\n\nget_index_type(::ImplicitTree{I}) where I\nget_index_type(options::BVHOptions)\n\n\n\n\n\n","category":"function"},{"location":"utilities/#ImplicitBVH.TaskPartitioner","page":"Utilities","title":"ImplicitBVH.TaskPartitioner","text":"struct TaskPartitioner\n\nPartitioning num_elems elements / jobs over maximum max_tasks tasks with minimum min_elems elements per task.\n\nMethods\n\nTaskPartitioner(num_elems, max_tasks=Threads.nthreads(), min_elems=1)\n\nFields\n\nnum_elems::Int64\nmax_tasks::Int64\nmin_elems::Int64\nnum_tasks::Int64\n\nExamples\n\nusing ImplicitBVH: TaskPartitioner\n\n# Divide 10 elements between 4 tasks\ntp = TaskPartitioner(10, 4)\nfor i in 1:tp.num_tasks\n    @show tp[i]\nend\n\n# output\ntp[i] = (1, 3)\ntp[i] = (4, 6)\ntp[i] = (7, 9)\ntp[i] = (10, 10)\n\nusing ImplicitBVH: TaskPartitioner\n\n# Divide 20 elements between 6 tasks with minimum 5 elements per task.\n# Not all tasks will be required\ntp = TaskPartitioner(20, 6, 5)\nfor i in 1:tp.num_tasks\n    @show tp[i]\nend\n\n# output\ntp[i] = (1, 5)\ntp[i] = (6, 10)\ntp[i] = (11, 15)\ntp[i] = (16, 20)\n\n\n\n\n\n","category":"type"},{"location":"#ImplicitBVH.jl-Documentation","page":"ImplicitBVH.jl Documentation","title":"ImplicitBVH.jl Documentation","text":"","category":"section"},{"location":"#BVH-Construction-and-Traversal","page":"ImplicitBVH.jl Documentation","title":"BVH Construction & Traversal","text":"","category":"section"},{"location":"","page":"ImplicitBVH.jl Documentation","title":"ImplicitBVH.jl Documentation","text":"BVH\ntraverse\nBVHTraversal\ndefault_start_level\nImplicitBVH.IndexPair","category":"page"},{"location":"#ImplicitBVH.BVH","page":"ImplicitBVH.jl Documentation","title":"ImplicitBVH.BVH","text":"struct BVH{I<:Integer, VN<:(AbstractVector), VL<:(AbstractVector), VM<:(AbstractVector), VO<:(AbstractVector)}\n\nImplicit bounding volume hierarchy constructed from an iterable of some geometric primitives' (e.g. triangles in a mesh) bounding volumes forming the ImplicitTree leaves. The leaves and merged nodes above them can have different types - e.g. BSphere{Float64} for leaves merged into larger BBox{Float64}.\n\nThe initial geometric primitives are sorted according to their Morton-encoded coordinates; the unsigned integer type used for the Morton encoding can be chosen between Union{UInt16, UInt32, UInt64}.\n\nFinally, the tree can be incompletely-built up to a given built_level and later start contact detection downwards from this level, e.g.:\n\nImplicit tree from 5 bounding volumes - i.e. the real leaves\n\nTree Level          Nodes & Leaves               Build Up    Traverse Down\n    1                     1                         Ʌ              |\n    2             2               3                 |              |\n    3         4       5       6        7v           |              |\n    4       8   9   10 11   12 13v  14v  15v        |              V\n            -------Real------- ---Virtual---\n\nMethods\n\n# Normal constructor which builds BVH\nBVH(\n    bounding_volumes::AbstractVector{L},\n    node_type::Type{N}=L,\n    morton_type::Type{U}=UInt32,\n    built_level=1;\n    options=BVHOptions(),\n) where {L, N, U <: MortonUnsigned}\n\n# Copy constructor reusing previous memory; previous\n# bounding volumes are moved to new_positions.\nBVH(\n    prev::BVH,\n    new_positions::AbstractMatrix,\n    built_level=1;\n    options=BVHOptions(),\n)\n\nFields\n\ntree::ImplicitTree{I <: Integer}\nnodes::VN <: AbstractVector\nleaves::VL <: AbstractVector\nmortons::VM <: AbstractVector\norder::VO <: AbstractVector\nbuilt_level::Int\n\nExamples\n\nSimple usage with bounding spheres and default 64-bit types:\n\nusing ImplicitBVH\nusing ImplicitBVH: BSphere\n\n# Generate some simple bounding spheres\nbounding_spheres = [\n    BSphere([0., 0., 0.], 0.5),\n    BSphere([0., 0., 1.], 0.6),\n    BSphere([0., 0., 2.], 0.5),\n    BSphere([0., 0., 3.], 0.4),\n    BSphere([0., 0., 4.], 0.6),\n]\n\n# Build BVH\nbvh = BVH(bounding_spheres)\n\n# Traverse BVH for contact detection\ntraversal = traverse(bvh)\n@show traversal.contacts;\n;\n\n# output\ntraversal.contacts = Tuple{Int32, Int32}[(1, 2), (2, 3), (4, 5)]\n\nUsing Float32 bounding spheres for leaves, Float32 bounding boxes for nodes above, and UInt32 Morton codes:\n\nusing ImplicitBVH\nusing ImplicitBVH: BBox, BSphere\n\n# Generate some simple bounding spheres\nbounding_spheres = [\n    BSphere{Float32}([0., 0., 0.], 0.5),\n    BSphere{Float32}([0., 0., 1.], 0.6),\n    BSphere{Float32}([0., 0., 2.], 0.5),\n    BSphere{Float32}([0., 0., 3.], 0.4),\n    BSphere{Float32}([0., 0., 4.], 0.6),\n]\n\n# Build BVH\nbvh = BVH(bounding_spheres, BBox{Float32}, UInt32)\n\n# Traverse BVH for contact detection\ntraversal = traverse(bvh)\n@show traversal.contacts;\n;\n\n# output\ntraversal.contacts = Tuple{Int32, Int32}[(1, 2), (2, 3), (4, 5)]\n\nBuild BVH up to level 2 and start traversing down from level 3, reusing the previous traversal cache:\n\nbvh = BVH(bounding_spheres, BBox{Float32}, UInt32, 2)\ntraversal = traverse(bvh, 3, traversal)\n\nUpdate previous BVH bounding volumes' positions and rebuild BVH reusing previous memory:\n\nnew_positions = rand(3, 5)\nbvh_rebuilt = BVH(bvh, new_positions)\n\n\n\n\n\n","category":"type"},{"location":"#ImplicitBVH.traverse","page":"ImplicitBVH.jl Documentation","title":"ImplicitBVH.traverse","text":"traverse(\n    bvh::BVH,\n    start_level::Int=default_start_level(bvh),\n    cache::Union{Nothing, BVHTraversal}=nothing;\n    options=BVHOptions(),\n)::BVHTraversal\n\nTraverse bvh downwards from start_level, returning all contacting bounding volume leaves. The returned BVHTraversal also contains two contact buffers that can be reused on future traversals.\n\nExamples\n\nusing ImplicitBVH\nusing ImplicitBVH: BBox, BSphere\n\n# Generate some simple bounding spheres\nbounding_spheres = [\n    BSphere{Float32}([0., 0., 0.], 0.5),\n    BSphere{Float32}([0., 0., 1.], 0.6),\n    BSphere{Float32}([0., 0., 2.], 0.5),\n    BSphere{Float32}([0., 0., 3.], 0.4),\n    BSphere{Float32}([0., 0., 4.], 0.6),\n]\n\n# Build BVH\nbvh = BVH(bounding_spheres, BBox{Float32}, UInt32)\n\n# Traverse BVH for contact detection\ntraversal = traverse(bvh, 2)\n\n# Reuse traversal buffers for future contact detection - possibly with different BVHs\ntraversal = traverse(bvh, 2, traversal)\n@show traversal.contacts;\n;\n\n# output\ntraversal.contacts = Tuple{Int32, Int32}[(1, 2), (2, 3), (4, 5)]\n\n\n\n\n\ntraverse(\n    bvh1::BVH,\n    bvh2::BVH,\n    start_level1::Int=default_start_level(bvh1),\n    start_level2::Int=default_start_level(bvh2),\n    cache::Union{Nothing, BVHTraversal}=nothing;\n    options=BVHOptions(),\n)::BVHTraversal\n\nReturn all the bvh1 bounding volume leaves that are in contact with any in bvh2. The returned BVHTraversal also contains two contact buffers that can be reused on future traversals.\n\nExamples\n\nusing ImplicitBVH\nusing ImplicitBVH: BBox, BSphere\n\n# Generate some simple bounding spheres\nbounding_spheres1 = [\n    BSphere{Float32}([0., 0., 0.], 0.5),\n    BSphere{Float32}([0., 0., 3.], 0.4),\n]\n\nbounding_spheres2 = [\n    BSphere{Float32}([0., 0., 1.], 0.6),\n    BSphere{Float32}([0., 0., 2.], 0.5),\n    BSphere{Float32}([0., 0., 4.], 0.6),\n]\n\n# Build BVHs\nbvh1 = BVH(bounding_spheres1, BBox{Float32}, UInt32)\nbvh2 = BVH(bounding_spheres2, BBox{Float32}, UInt32)\n\n# Traverse BVH for contact detection\ntraversal = traverse(bvh1, bvh2, default_start_level(bvh1), default_start_level(bvh2))\n\n# Reuse traversal buffers for future contact detection - possibly with different BVHs\ntraversal = traverse(bvh1, bvh2, default_start_level(bvh1), default_start_level(bvh2), traversal)\n@show traversal.contacts;\n;\n\n# output\ntraversal.contacts = Tuple{Int32, Int32}[(1, 1), (2, 3)]\n\n\n\n\n\n","category":"function"},{"location":"#ImplicitBVH.BVHTraversal","page":"ImplicitBVH.jl Documentation","title":"ImplicitBVH.BVHTraversal","text":"struct BVHTraversal{C1<:(AbstractVector), C2<:(AbstractVector)}\n\nCollected BVH traversal contacts vector, some stats, plus the two buffers cache1 and cache2 which can be reused for future traversals to minimise memory allocations.\n\nFields\n\nstart_level1::Int: the level at which the single/pair-tree traversal started for the first BVH.\nstart_level2::Int: the level at which the pair-tree traversal started for the second BVH.\nnum_checks::Int: the total number of contact checks done.\nnum_contacts::Int: the number of contacts found.\ncontacts::view(cache1, 1:num_contacts): the contacting pairs found, as a view into cache1.\ncache1::C1{IndexPair} <: AbstractVector: first BVH traversal buffer.\ncache2::C2{IndexPair} <: AbstractVector: second BVH traversal buffer.\n\n\n\n\n\n","category":"type"},{"location":"#ImplicitBVH.default_start_level","page":"ImplicitBVH.jl Documentation","title":"ImplicitBVH.default_start_level","text":"default_start_level(bvh::BVH)::Int\ndefault_start_level(num_leaves::Integer)::Int\n\nCompute the default start level when traversing a single BVH tree.\n\n\n\n\n\n","category":"function"},{"location":"#ImplicitBVH.IndexPair","page":"ImplicitBVH.jl Documentation","title":"ImplicitBVH.IndexPair","text":"struct Tuple{I, I}\n\nAlias for a tuple of two indices representing e.g. a contacting pair.\n\n\n\n\n\n","category":"type"},{"location":"#Index","page":"ImplicitBVH.jl Documentation","title":"Index","text":"","category":"section"},{"location":"","page":"ImplicitBVH.jl Documentation","title":"ImplicitBVH.jl Documentation","text":"","category":"page"}]
}
