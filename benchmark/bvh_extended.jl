using ImplicitBVH
using ImplicitBVH: BSphere, BBox

using MeshIO
using FileIO

using BenchmarkTools
using Profile
using PProf


# Types used
const LeafType = BSphere{Float32}
const NodeType = BBox{Float32}
options = BVHOptions(index_exemplar=Int32(0), morton=ExtendedMortonAlgorithm(UInt32(0)))


# Load mesh and compute bounding spheres for each triangle. Can download mesh from:
# https://github.com/alecjacobson/common-3d-test-models/blob/master/data/xyzrgb_dragon.obj
mesh = load(joinpath(@__DIR__, "xyzrgb_dragon.obj"))
display(mesh)

bounding_spheres = [LeafType(tri) for tri in mesh]

# Pre-compile BVH build
bvh = BVH(bounding_spheres, NodeType, options=options)
