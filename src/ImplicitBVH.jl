# File   : ImplicitBVH.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 02.06.2022


module ImplicitBVH

# Functionality exported by this package by default
export BVH, BVHTraversal, BVHOptions, traverse,  traverse_rays, default_start_level
export ImplicitTree, memory_index, level_indices, isvirtual


# Internal dependencies
using LinearAlgebra
using DocStringExtensions

using ArgCheck
using KernelAbstractions
using Atomix: @atomic
using GPUArraysCore: AbstractGPUVector, @allowscalar

import AcceleratedKernels as AK


include("utils.jl")
include("morton.jl")
include("implicit_tree.jl")
include("bounding_volumes/bounding_volumes.jl")
include("build.jl")
include("traverse/traverse.jl")
include("raytrace/raytrace.jl")
include("utils_post.jl")

end     # module ImplicitBVH
