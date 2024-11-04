"""
    iscontact(a::BSphere, b::BSphere)
    iscontact(a::BBox, b::BBox)
    iscontact(a::BSphere, b::BBox)
    iscontact(a::BBox, b::BSphere)

Check if two bounding volumes are touching or inter-penetrating.
"""
function iscontact end


"""
    isintersection(b::BBox, p::AbstractVector, d::AbstractVector)
    isintersection(s::BSphere, p::AbstractVector, d::AbstractVector)

Check if a forward ray, defined by a point `p` and a direction `d` intersects a bounding volume;
`p` and `d` can be any iterables with 3 numbers (e.g. `Vector{Float64}`).

# Examples
Simple ray bounding box intersection example:

```jldoctest
using ImplicitBVH
using ImplicitBVH: BSphere, BBox, isintersection

# Generate a simple bounding box
bounding_box = BBox((0., 0., 0.), (1., 1., 1.))

# Generate a ray passing up and through the bottom face of the bounding box
point = [.5, .5, -10]
direction = [0, 0, 1]
isintersection(bounding_box, point, direction)

# output
true
```

Simple ray bounding sphere intersection example:
```jldoctest
using ImplicitBVH
using ImplicitBVH: BSphere, BBox, isintersection

# Generate a simple bounding sphere
bounding_sphere = BSphere((0., 0., 0.), 0.5)

# Generate a ray passing up and through the bounding sphere
point = [0, 0, -10]
direction = [0, 0, 1]
isintersection(bounding_sphere, point, direction)

# output
true
```
"""
function isintersection end


"""
    center(b::BSphere)
    center(b::BBox{T}) where T

Get the coordinates of a bounding volume's centre, as a NTuple{3, T}.
"""
function center end


"""
    translate(b::BSphere{T}, dx) where T
    translate(b::BBox{T}, dx) where T

Get a new bounding volume translated by dx; dx can be any iterable with 3 elements.
"""
function translate end


# Sub-includes
include("bsphere.jl")
include("bbox.jl")
include("merge.jl")
include("iscontact.jl")
include("isintersection.jl")
