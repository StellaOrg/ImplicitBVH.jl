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
