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
