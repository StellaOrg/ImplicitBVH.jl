[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://stellaorg.github.io/IBVH.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://stellaorg.github.io/IBVH.jl/dev)

# IBVH.jl
*Robust Multithreaded Bounding Volume Hierarchy for Collision Detection in Dynamic Scenes*

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
using IBVH
using IBVH: BBox, BSphere
using StaticArrays

# Generate some simple bounding spheres
bounding_spheres = [
    BSphere(SA[0., 0., 0.], 0.5),
    BSphere(SA[0., 0., 1.], 0.6),
    BSphere(SA[0., 0., 2.], 0.5),
    BSphere(SA[0., 0., 3.], 0.4),
    BSphere(SA[0., 0., 4.], 0.6),
]

# Build BVH
bvh = BVH(bounding_spheres)

# Traverse BVH for contact detection
traversal = traverse(bvh)
@show traversal.contacts

# output
traversal.contacts = [(4, 5), (1, 2), (2, 3)]
```

Using `Float32` bounding spheres for leaves, `Float32` bounding boxes for nodes above, and `UInt32`
Morton codes:

```julia
using IBVH
using IBVH: BBox, BSphere
using StaticArrays

# Generate some simple bounding spheres
bounding_spheres = [
    BSphere{Float32}(SA[0., 0., 0.], 0.5),
    BSphere{Float32}(SA[0., 0., 1.], 0.6),
    BSphere{Float32}(SA[0., 0., 2.], 0.5),
    BSphere{Float32}(SA[0., 0., 3.], 0.4),
    BSphere{Float32}(SA[0., 0., 4.], 0.6),
]

# Build BVH
bvh = BVH(bounding_spheres, BBox{Float32}, UInt32)

# Traverse BVH for contact detection
traversal = traverse(bvh)
@show traversal.contacts

# output
traversal.contacts = [(4, 5), (1, 2), (2, 3)]
```

Build BVH up to level 2 and start traversing down from level 3, reusing the previous traversal
cache:

```julia
bvh = BVH(bounding_spheres, BBox{Float32}, UInt32, 2)
traversal = traverse(bvh, 3, traversal)
```

Check out the `benchmark` folder for an example traversing an STL model.


# Implicit Bounding Volume Hierarchy

The main idea behind the IBVH is the use of an implicit perfect binary tree constructed from some
bounding volumes. If we had, say, 5 objects to construct the BVH from, it would form an incomplete
binary tree as below:

```
Implicit tree from 5 bounding volumes - i.e. the real leaves:

Tree Level          Nodes & Leaves               Build Up    Traverse Down
    1                     1                         Ʌ              |
    2             2               3                 |              |
    3         4       5       6        7v           |              |
    4       8   9   10 11   12 13v  14v  15v        |              V
            -----Real----   -----Virtual----
```

We do not need to store the "virtual" nodes in memory; rather, we can compute the number of virtual
nodes we need to skip to get to a given node index, following the fantastic ideas from [1].


# Roadmap

- Use `KernelAbstractions.jl` kernels to build and traverse the BVH; I think we just need a performant KA `sort!` function, the rest is straightforward.


# References

The implicit tree formulation (genius idea!) which forms the core of the BVH structure originally appeared in the following paper:

> [1] Chitalu FM, Dubach C, Komura T. Binary Ostensibly‐Implicit Trees for Fast Collision Detection. InComputer Graphics Forum 2020 May (Vol. 39, No. 2, pp. 509-521).


# License
`IBVH.jl` is MIT-licensed. Enjoy.
