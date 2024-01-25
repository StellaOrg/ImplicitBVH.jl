# File   : ImplicitBVH.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 02.06.2022


module ImplicitBVH

# Functionality exported by this package by default
export BVH, BVHTraversal, traverse, default_start_level
export ImplicitTree, memory_index, level_indices, isvirtual


# Internal dependencies
using LinearAlgebra
using DocStringExtensions


# Include code from other files
include("utils.jl")
include("morton.jl")
include("implicit_tree.jl")
include("bounding_volumes.jl")
include("build.jl")
include("traverse/traverse.jl")

end     # module ImplicitBVH
