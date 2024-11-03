[![ImplicitBVH](https://github.com/StellaOrg/ImplicitBVH.jl/blob/main/docs/src/assets/logo.png?raw=true)](https://stellaorg.github.io/ImplicitBVH.jl/)
[![ImplicitBVH](https://github.com/StellaOrg/ImplicitBVH.jl/blob/main/docs/src/assets/bunny.png?raw=true)](https://stellaorg.github.io/ImplicitBVH.jl/)


[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://stellaorg.github.io/ImplicitBVH.jl/stable)
[![Docs Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://stellaorg.github.io/ImplicitBVH.jl/dev)
[![Build Status](https://github.com/StellaOrg/ImplicitBVH.jl/workflows/CI/badge.svg)](https://github.com/StellaOrg/ImplicitBVH.jl/actions/workflows/ci.yml)


# ImplicitBVH.jl
*High-Performance Cross-Architecture Bounding Volume Hierarchy for Collision Detection and Ray Tracing*

**New in v0.5.0: Ray Tracing and GPU acceleration via [AcceleratedKernels.jl](https://github.com/anicusan/AcceleratedKernels.jl)/[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) targeting all JuliaGPU backends, i.e. Nvidia CUDA, AMD ROCm, Intel oneAPI, Apple Metal.**

It uses an implicit bounding volume hierarchy constructed from an iterable of some geometric
primitives' (e.g. triangles in a mesh) bounding volumes forming the `ImplicitTree` leaves. The leaves
and merged nodes above them can have different types - e.g. `BSphere{Float64}` leaves merged into
larger `BBox{Float64}`.

The initial geometric primitives are sorted according to their Morton-encoded coordinates; the
unsigned integer type used for the Morton encoding can be chosen between `UInt16`, `UInt32` and `UInt64`.

Finally, the tree can be incompletely-built up to a given `built_level` and later start contact
detection downwards from this level.


## Examples

Simple usage with bounding spheres and default 64-bit types:

```julia
using ImplicitBVH
using ImplicitBVH: BBox, BSphere

# Generate some simple bounding spheres
bounding_spheres = [
    BSphere([0., 0., 0.], 0.5),
    BSphere([0., 0., 1.], 0.6),
    BSphere([0., 0., 2.], 0.5),
    BSphere([0., 0., 3.], 0.4),
    BSphere([0., 0., 4.], 0.6),
]

# Build BVH
bvh = BVH(bounding_spheres)

# Traverse BVH for contact detection
traversal = traverse(bvh)
@show traversal.contacts

# output
traversal.contacts = [(1, 2), (2, 3), (4, 5)]
```

Using `Float32` bounding spheres for leaves, `Float32` bounding boxes for nodes above, and `UInt32`
Morton codes:

```julia
using ImplicitBVH
using ImplicitBVH: BBox, BSphere

# Generate some simple bounding spheres
bounding_spheres = [
    BSphere{Float32}([0., 0., 0.], 0.5),
    BSphere{Float32}([0., 0., 1.], 0.6),
    BSphere{Float32}([0., 0., 2.], 0.5),
    BSphere{Float32}([0., 0., 3.], 0.4),
    BSphere{Float32}([0., 0., 4.], 0.6),
]

# Build BVH
bvh = BVH(bounding_spheres, BBox{Float32}, UInt32)

# Traverse BVH for contact detection
traversal = traverse(bvh)
@show traversal.contacts

# output
traversal.contacts = [(1, 2), (2, 3), (4, 5)]
```

Build BVH up to level 2 and start traversing down from level 3, reusing the previous traversal
cache:

```julia
bvh = BVH(bounding_spheres, BBox{Float32}, UInt32, 2)
traversal = traverse(bvh, 3, traversal)
```

Update previous BVH bounding volumes' positions and rebuild BVH *reusing previous memory*:

```julia
new_positions = rand(3, 5)
bvh_rebuilt = BVH(bvh, new_positions)
```

Compute contacts between two different BVH trees (e.g. two different robotic parts):

```julia
using ImplicitBVH
using ImplicitBVH: BBox, BSphere

# Generate some simple bounding spheres (will be BVH leaves)
bounding_spheres1 = [
    BSphere{Float32}([0., 0., 0.], 0.5),
    BSphere{Float32}([0., 0., 3.], 0.4),
]

bounding_spheres2 = [
    BSphere{Float32}([0., 0., 1.], 0.6),
    BSphere{Float32}([0., 0., 2.], 0.5),
    BSphere{Float32}([0., 0., 4.], 0.6),
]

# Build BVHs using bounding boxes for nodes
bvh1 = BVH(bounding_spheres1, BBox{Float32}, UInt32)
bvh2 = BVH(bounding_spheres2, BBox{Float32}, UInt32)

# Traverse BVH for contact detection
traversal = traverse(
    bvh1,
    bvh2,
    default_start_level(bvh1),
    default_start_level(bvh2),
    # previous_traversal_cache,
    # options=BVHOptions(),
)
```

Check out the `benchmark` folder for an example traversing an STL model.


# GPU Bounding Volume Hierarchy Building and Traversal

Simply use a GPU array for the bounding volumes; the interface remains the same, and all operations - Morton encoding, sorting, BVH building and traversal for contact finding - will run on the right backend:

```julia
# Works with CUDA.jl/CuArray, AMDGPU.jl/ROCArray, oneAPI.jl/oneArray, Metal.jl/MtlArray
using AMDGPU

using ImplicitBVH
using ImplicitBVH: BBox, BSphere

# Generate some simple bounding spheres; save them in a GPU array
bounding_spheres = ROCArray([
    BSphere{Float32}([0., 0., 0.], 0.5),
    BSphere{Float32}([0., 0., 1.], 0.6),
    BSphere{Float32}([0., 0., 2.], 0.5),
    BSphere{Float32}([0., 0., 3.], 0.4),
    BSphere{Float32}([0., 0., 4.], 0.6),
])

# Build BVH
bvh = BVH(bounding_spheres, BBox{Float32}, UInt32)

# Traverse BVH for contact detection
traversal = traverse(bvh)
```


# Implicit Bounding Volume Hierarchy

The main idea behind the ImplicitBVH is the use of an implicit perfect binary tree constructed from some
bounding volumes. If we had, say, 5 objects to construct the BVH from, it would form an incomplete
binary tree as below:

```
Implicit tree from 5 bounding volumes - i.e. the real leaves:

Tree Level          Nodes & Leaves               Build Up    Traverse Down
    1                     1                         Ʌ              |
    2             2               3                 |              |
    3         4       5       6        7v           |              |
    4       8   9   10 11   12 13v  14v  15v        |              V
            -------Real------- ---Virtual---
```

We do not need to store the "virtual" nodes in memory; rather, we can compute the number of virtual
nodes we need to skip to get to a given node index, following the fantastic ideas from [1].


# Performance

As contact detection is one of the most computationally-intensive parts of physical simulation and computer
vision applications, ~~we spent a stupid amount of time optimising~~ this implementation has been optimised for maximum performance and scalability:

- Computing bounding volumes is optimised for triangles, e.g. constructing 249,882 `BSphere{Float64}` on a single thread takes 4.47 ms on my Mac M1. The construction itself has zero allocations; all computation can be done in parallel in user code.
- Building a complete bounding volume hierarchy from the 249,882 triangles of [`xyzrgb_dragon.obj`](https://github.com/alecjacobson/common-3d-test-models/blob/master/data/xyzrgb_dragon.obj) takes 11.83 ms single-threaded. The sorting step is the bottleneck, so multi-threading the Morton encoding and BVH up-building does not significantly improve the runtime; waiting on a multi-threaded sorter.
- Traversing the same 249,882 `BSphere{Float64}` for the triangles (aggregated into `BBox{Float64}` parents) takes 136.38 ms single-threaded and 43.16 ms with 4 threads, at 79% strong scaling.

Only fundamental Julia types are used - e.g. `struct`, `Tuple`, `UInt`, `Float64` - which can be straightforwardly inlined, unrolled and fused by the compiler. These types are also straightforward to transpile to accelerators via [`KernelAbstractions.jl`](https://github.com/JuliaGPU/KernelAbstractions.jl) such as `CUDA`, `AMDGPU`, `oneAPI`, `Apple Metal`.


# Roadmap

- Raytracing
- Atomics on Metal and oneAPI - waiting on Atomix.jl ([Metal Issue](https://github.com/JuliaGPU/Metal.jl/issues/218), [oneAPI Issue](https://github.com/JuliaGPU/oneAPI.jl/issues/336)).
- Coherent types for indices on GPUs (e.g. when using Int32 in core kernels, only use 32-bit objects)
- Avoiding / exposing memory allocations (temps, minmax reduce, morton order, etc.)
- GPU CI


# References

The implicit tree formulation (genius idea!) which forms the core of the BVH structure originally appeared in the following paper:

> [1] Chitalu FM, Dubach C, Komura T. Binary Ostensibly‐Implicit Trees for Fast Collision Detection. InComputer Graphics Forum 2020 May (Vol. 39, No. 2, pp. 509-521).


# License
`ImplicitBVH.jl` is MIT-licensed. Enjoy.
