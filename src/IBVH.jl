# File   : OIBVH.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 02.06.2022


module IBVH

# Functionality exported by this package by default
export BVH, BVHTraversal, traverse
export ImplicitTree, memory_index, level_indices, isvirtual


# Internal dependencies
using LinearAlgebra
using Parameters
using StaticArrays
using DocStringExtensions


# Include code from other files
include("utils.jl")
include("morton.jl")
include("implicit_tree.jl")
include("bounding_volumes.jl")
include("ibvh_build.jl")
include("ibvh_traverse.jl")

end     # module IBVH
