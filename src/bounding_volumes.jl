"""
    iscontact(a::BSphere, b::BSphere)
    iscontact(a::BBox, b::BBox)
    iscontact(a::BSphere, b::BBox)
    iscontact(a::BBox, b::BSphere)

Check if two bounding volumes are touching or inter-penetrating.
"""
function iscontact end

"""
    isintersection(b::BBox, p::Type{3, T}, d::Type{3, T})
    isintersection(s::BSphere, p::Type{3, T}, d::Type{3, T})

Return True if ray intersects a sphere or box
"""

# will go into bounding volumes
function isintersection end

"""
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


"""
    $(TYPEDEF)

Bounding sphere, highly optimised for computing bounding volumes for triangles and merging into
larger bounding volumes.

# Methods
    # Convenience constructors
    BSphere(x::NTuple{3, T}, r)
    BSphere{T}(x::AbstractVector, r) where T
    BSphere(x::AbstractVector, r)

    # Construct from triangle vertices
    BSphere{T}(p1, p2, p3) where T
    BSphere(p1, p2, p3)
    BSphere{T}(vertices::AbstractMatrix) where T
    BSphere(vertices::AbstractMatrix)
    BSphere{T}(triangle) where T
    BSphere(triangle)

    # Merging bounding volumes
    BSphere{T}(a::BSphere, b::BSphere) where T
    BSphere(a::BSphere{T}, b::BSphere{T}) where T
    Base.:+(a::BSphere, b::BSphere)
"""
struct BSphere{T}
    x::NTuple{3, T}
    r::T
end

Base.eltype(::BSphere{T}) where T = T
Base.eltype(::Type{BSphere{T}}) where T = T


# Convenience constructors, with and without type parameter
BSphere{T}(x::AbstractVector, r) where T = BSphere(NTuple{3, T}(x), T(r))
BSphere(x::AbstractVector, r) = BSphere{eltype(x)}(x, r)


# Constructors from triangles
function BSphere{T}(p1, p2, p3) where T

    # Adapted from https://realtimecollisiondetection.net/blog/?p=20
    a = (T(p1[1]), T(p1[2]), T(p1[3]))
    b = (T(p2[1]), T(p2[2]), T(p2[3]))
    c = (T(p3[1]), T(p3[2]), T(p3[3]))

    # Unrolled dot(b - a, b - a)
    abab = (b[1] - a[1]) * (b[1] - a[1]) +
           (b[2] - a[2]) * (b[2] - a[2]) +
           (b[3] - a[3]) * (b[3] - a[3])

    # Unrolled dot(b - a, c - a)
    abac = (b[1] - a[1]) * (c[1] - a[1]) +
           (b[2] - a[2]) * (c[2] - a[2]) +
           (b[3] - a[3]) * (c[3] - a[3])

    # Unrolled dot(c - a, c - a)
    acac = (c[1] - a[1]) * (c[1] - a[1]) +
           (c[2] - a[2]) * (c[2] - a[2]) +
           (c[3] - a[3]) * (c[3] - a[3])

    d = T(2.) * (abab * acac - abac * abac)

    if abs(d) <= eps(T)
        # a, b, c lie on a line. Find line centre and radius
        lower = (minimum3(a[1], b[1], c[1]),
                 minimum3(a[2], b[2], c[2]),
                 minimum3(a[3], b[3], c[3]))

        upper = (maximum3(a[1], b[1], c[1]),
                 maximum3(a[2], b[2], c[2]),
                 maximum3(a[3], b[3], c[3]))

        centre = (T(0.5) * (lower[1] + upper[1]),
                  T(0.5) * (lower[2] + upper[2]),
                  T(0.5) * (lower[3] + upper[3]))
        radius = dist3(centre, upper)
    else
        s = (abab * acac - acac * abac) / d
        t = (acac * abab - abab * abac) / d

        if s <= zero(T)
            centre = (T(0.5) * (a[1] + c[1]),
                      T(0.5) * (a[2] + c[2]),
                      T(0.5) * (a[3] + c[3]))
            radius = dist3(centre, a)
        elseif t <= zero(T)
            centre = (T(0.5) * (a[1] + b[1]),
                      T(0.5) * (a[2] + b[2]),
                      T(0.5) * (a[3] + b[3]))
            radius = dist3(centre, a)
        elseif s + t >= one(T)
            centre = (T(0.5) * (b[1] + c[1]),
                      T(0.5) * (b[2] + c[2]),
                      T(0.5) * (b[3] + c[3]))
            radius = dist3(centre, b)
        else
            centre = (a[1] + s * (b[1] - a[1]) + t * (c[1] - a[1]),
                      a[2] + s * (b[2] - a[2]) + t * (c[2] - a[2]),
                      a[3] + s * (b[3] - a[3]) + t * (c[3] - a[3]))
            radius = dist3(centre, a)
        end
    end

    BSphere(centre, radius)
end


# Convenience constructors, with and without explicit type parameter
function BSphere(p1, p2, p3)
    BSphere{eltype(p1)}(p1, p2, p3)
end

function BSphere{T}(triangle) where T
    # Decompose triangle into its 3 vertices.
    # Works transparently with GeometryBasics.Triangle, Vector{SVector{3, T}}, etc.
    p1, p2, p3 = triangle
    BSphere{T}(p1, p2, p3)
end

function BSphere(triangle)
    p1, p2, p3 = triangle
    BSphere{eltype(p1)}(p1, p2, p3)
end

function BSphere{T}(vertices::AbstractMatrix) where T
    BSphere{T}(@view(vertices[:, 1]), @view(vertices[:, 2]), @view(vertices[:, 3]))
end

function BSphere(vertices::AbstractMatrix)
    BSphere{eltype(vertices)}(@view(vertices[:, 1]), @view(vertices[:, 2]), @view(vertices[:, 3]))
end


# Overloaded center function
center(b::BSphere) = b.x


# Overloaded translate function
function translate(b::BSphere{T}, dx) where T
    new_center = (b.x[1] + T(dx[1]),
                  b.x[2] + T(dx[2]),
                  b.x[3] + T(dx[3]))
    BSphere{T}(new_center, b.r)
end


# Merge two bounding spheres
function BSphere{T}(a::BSphere, b::BSphere) where T
    length = dist3(a.x, b.x)

    # a is enclosed within b
    if length + a.r <= b.r
        return BSphere{T}(b.x, b.r)

    # b is enclosed within a
    elseif length + b.r <= a.r
        return BSphere{T}(a.x, a.r)

    # Bounding spheres are not enclosed
    else
        frac = T(0.5) * ((b.r - a.r) / length + T(1))
        centre = (a.x[1] + frac * (b.x[1] - a.x[1]),
                  a.x[2] + frac * (b.x[2] - a.x[2]),
                  a.x[3] + frac * (b.x[3] - a.x[3]))
        radius = T(0.5) * (length + a.r + b.r)
        return BSphere{T}(centre, radius)
    end
end


BSphere(a::BSphere{T}, b::BSphere{T}) where T = BSphere{T}(a, b)
Base.:+(a::BSphere, b::BSphere) = BSphere(a, b)


# Contact detection
function iscontact(a::BSphere, b::BSphere)
    dist3sq(a.x, b.x) <= (a.r + b.r) * (a.r + b.r)
end




"""
    $(TYPEDEF)

Axis-aligned bounding box, highly optimised for computing bounding volumes for triangles and
merging into larger bounding volumes.

Can also be constructed from two spheres to e.g. allow merging [`BSphere`](@ref) leaves into
[`BBox`](@ref) nodes.

# Methods
    # Convenience constructors
    BBox(lo::NTuple{3, T}, up::NTuple{3, T}) where T
    BBox{T}(lo::AbstractVector, up::AbstractVector) where T
    BBox(lo::AbstractVector, up::AbstractVector)

    # Construct from triangle vertices
    BBox{T}(p1, p2, p3) where T
    BBox(p1, p2, p3)
    BBox{T}(vertices::AbstractMatrix) where T
    BBox(vertices::AbstractMatrix)
    BBox{T}(triangle) where T
    BBox(triangle)

    # Merging bounding boxes
    BBox{T}(a::BBox, b::BBox) where T
    BBox(a::BBox{T}, b::BBox{T}) where T
    Base.:+(a::BBox, b::BBox)

    # Merging bounding spheres
    BBox{T}(a::BSphere{T}) where T
    BBox(a::BSphere{T}) where T
    BBox{T}(a::BSphere{T}, b::BSphere{T}) where T
    BBox(a::BSphere{T}, b::BSphere{T}) where T
"""
struct BBox{T}
    lo::NTuple{3, T}
    up::NTuple{3, T}
end

Base.eltype(::BBox{T}) where T = T
Base.eltype(::Type{BBox{T}}) where T = T



# Convenience constructors, with and without type parameter
function BBox{T}(lo::AbstractVector, up::AbstractVector) where T
    BBox(NTuple{3, eltype(lo)}(lo), NTuple{3, eltype(up)}(up))
end

function BBox(lo::AbstractVector, up::AbstractVector)
    BBox{eltype(lo)}(lo, up)
end



# Constructors from triangles
function BBox{T}(p1, p2, p3) where T

    lower = (minimum3(p1[1], p2[1], p3[1]),
             minimum3(p1[2], p2[2], p3[2]),
             minimum3(p1[3], p2[3], p3[3]))

    upper = (maximum3(p1[1], p2[1], p3[1]),
             maximum3(p1[2], p2[2], p3[2]),
             maximum3(p1[3], p2[3], p3[3]))
   
    BBox{T}(lower, upper)
end


# Convenience constructors, with and without explicit type parameter
function BBox(p1, p2, p3)
    BBox{eltype(p1)}(p1, p2, p3)
end

function BBox{T}(triangle) where T
    # Decompose triangle into its 3 vertices.
    # Works transparently with GeometryBasics.Triangle, Vector{SVector{3, T}}, etc.
    p1, p2, p3 = triangle
    BBox{T}(p1, p2, p3)
end

function BBox(triangle)
    p1, p2, p3 = triangle
    BBox{eltype(p1)}(p1, p2, p3)
end

function BBox{T}(vertices::AbstractMatrix) where T
    BBox{T}(@view(vertices[:, 1]), @view(vertices[:, 2]), @view(vertices[:, 3]))
end

function BBox(vertices::AbstractMatrix)
    BBox{eltype(vertices)}(@view(vertices[:, 1]), @view(vertices[:, 2]), @view(vertices[:, 3]))
end


# Overloaded center function
center(b::BBox{T}) where T = (T(0.5) * (b.lo[1] + b.up[1]),
                              T(0.5) * (b.lo[2] + b.up[2]),
                              T(0.5) * (b.lo[3] + b.up[3]))


# Overloaded translate function
function translate(b::BBox{T}, dx) where T
    dx1, dx2, dx3 = T(dx[1]), T(dx[2]), T(dx[3])
    new_lo = (b.lo[1] + dx1,
              b.lo[2] + dx2,
              b.lo[3] + dx3)
    new_up = (b.up[1] + dx1,
              b.up[2] + dx2,
              b.up[3] + dx3)
    BBox{T}(new_lo, new_up)
end


# Merge two bounding boxes
function BBox{T}(a::BBox, b::BBox) where T
    lower = (minimum2(a.lo[1], b.lo[1]),
             minimum2(a.lo[2], b.lo[2]),
             minimum2(a.lo[3], b.lo[3]))

    upper = (maximum2(a.up[1], b.up[1]),
             maximum2(a.up[2], b.up[2]),
             maximum2(a.up[3], b.up[3]))

    BBox{T}(lower, upper)
end

BBox(a::BBox{T}, b::BBox{T}) where T = BBox{T}(a, b)
Base.:+(a::BBox, b::BBox) = BBox(a, b)


# Convert BSphere to BBox
function BBox{T}(a::BSphere{T}) where T
    lower = (a.x[1] - a.r, a.x[2] - a.r, a.x[3] - a.r)
    upper = (a.x[1] + a.r, a.x[2] + a.r, a.x[3] + a.r)
    BBox(lower, upper)
end

function BBox(a::BSphere{T}) where T
    BBox{T}(a)
end

# Merge two BSphere into enclosing BBox
function BBox{T}(a::BSphere{T}, b::BSphere{T}) where T
    length = dist3(a.x, b.x)

    # a is enclosed within b
    if length + a.r <= b.r
        return BBox(b)

    # b is enclosed within a
    elseif length + b.r <= a.r
        return BBox(a)

    # Bounding spheres are not enclosed
    else
        lower = (minimum2(a.x[1] - a.r, b.x[1] - b.r),
                 minimum2(a.x[2] - a.r, b.x[2] - b.r),
                 minimum2(a.x[3] - a.r, b.x[3] - b.r))

        upper = (maximum2(a.x[1] + a.r, b.x[1] + b.r),
                 maximum2(a.x[2] + a.r, b.x[2] + b.r),
                 maximum2(a.x[3] + a.r, b.x[3] + b.r))

        return BBox(lower, upper)
    end
end

function BBox(a::BSphere{T}, b::BSphere{T}) where T
    BBox{T}(a, b)
end


# Contact detection
function iscontact(a::BBox, b::BBox)
    (a.up[1] >= b.lo[1] && a.lo[1] <= b.up[1]) &&
    (a.up[2] >= b.lo[2] && a.lo[2] <= b.up[2]) &&
    (a.up[3] >= b.lo[3] && a.lo[3] <= b.up[3])
end


# Contact detection between heterogeneous BVs - only needed when one BVH has exactly one leaf
function iscontact(a::BSphere, b::BBox)
    # This is an edge case, used for broad-phase collision detection, so we simply take the
    # sphere's bounding box, as a full sphere-box contact detection is computationally heavy
    ab = BBox(
        (a.x[1] - a.r, a.x[2] - a.r, a.x[3] - a.r),
        (a.x[1] + a.r, a.x[2] + a.r, a.x[3] + a.r),
    )
    iscontact(ab, b)
end


function iscontact(a::BBox, b::BSphere)
    iscontact(b, a)
end


@inline function isintersection(b::BBox, p::AbstractVector, d::AbstractVector)

    @boundscheck begin
        @assert length(p) == 3
        @assert length(d) == 3
    end

    T = eltype(d)

    @inbounds begin
        inv_d = (one(T) / d[1], one(T) / d[2], one(T) / d[3])

        # Set x bounds
        t_bound_x1 = (b.lo[1] - p[1]) * inv_d[1]
        t_bound_x2 = (b.up[1] - p[1]) * inv_d[1]

        tmin = minimum2(t_bound_x1, t_bound_x2)
        tmax = maximum2(t_bound_x1, t_bound_x2)

        # Set y bounds
        t_bound_y1 = (b.lo[2] - p[2]) * inv_d[2]
        t_bound_y2 = (b.up[2] - p[2]) * inv_d[2]

        tmin = maximum2(tmin, minimum2(t_bound_y1, t_bound_y2))
        tmax = minimum2(tmax, maximum2(t_bound_y1, t_bound_y2))

        # Set z bounds
        t_bound_z1 = (b.lo[3] - p[3]) * inv_d[3]
        t_bound_z2 = (b.up[3] - p[3]) * inv_d[3]

        tmin = maximum2(tmin, minimum2(t_bound_z1, t_bound_z2))
        tmax = minimum2(tmax, maximum2(t_bound_z1, t_bound_z2))
    end
        
    # If condition satisfied ray intersects box. tmax >= 0 
    # ensure only forwards intersections are counted
    (tmin <= tmax) && (tmax >= 0)
end


@inline function isintersection(s::BSphere, p::AbstractVector, d::AbstractVector)

    @boundscheck begin
        @assert length(p) == 3
        @assert length(d) == 3
    end

    @inbounds begin
        a = dot3(d, d)
        b = 2 * (
            (p[1] - s.x[1]) * d[1] +
            (p[2] - s.x[2]) * d[2] +
            (p[3] - s.x[3]) * d[3]
        )
        c = (
            (p[1] - s.x[1]) * (p[1] - s.x[1]) +
            (p[2] - s.x[2]) * (p[2] - s.x[2]) +
            (p[3] - s.x[3]) * (p[3] - s.x[3])
        ) - s.r * s.r
    end

    discriminant = b * b - 4 * a * c

    if discriminant >= 0
        # Ensure only forwards intersections are counted
        return 0 >= b * c
    else
        return false
    end    
end