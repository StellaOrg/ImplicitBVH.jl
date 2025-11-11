"""
    iscontact(a::BSphere, b::BSphere)
    iscontact(a::BBox, b::BBox)
    iscontact(a::BSphere, b::BBox)
    iscontact(a::BBox, b::BSphere)

Check if two bounding volumes are touching or inter-penetrating.
"""
function iscontact end


"""
    isintersection(
        b::BBox{T},
        p::Union{AbstractVector{T}, NTuple{3, T}},
        d::Union{AbstractVector{T}, NTuple{3, T}},
    ) where T

    isintersection(
        s::BSphere{T},
        p::Union{AbstractVector{T}, NTuple{3, T}},
        d::Union{AbstractVector{T}, NTuple{3, T}},
    ) where T

Check if a forward ray, defined by a point `p` and a direction `d` intersects a bounding volume;
`p` and `d` can be any iterables with 3 numbers (e.g. `Vector{Float64}`).
"""
function isintersection end


"""
    center(b::BSphere)
    center(b::BBox{T}) where T

Get the coordinates of a bounding volume's centre, as a NTuple{3, T}.
"""
function center end


"""
    $(TYPEDEF)

Bounding volume wrapper, containing a bounding volume of type `V`, an index of type `I`,
and a computed Morton code of type `M`.

The index will be the one reported in case of contact during traversal; it can be anything
(user-defined) to identify the bounding volume later in e.g. a simulation.

# Fields
- `volume::V`: the bounding volume, e.g. `BSphere` or `BBox`.
- `index::I`: the user-defined index associated with this bounding volume.
- `morton::M`: the Morton code for this bounding volume computed during BVH construction.

"""
struct BoundingVolume{V, I, M}
    volume::V
    index::I
    morton::M
end

Base.eltype(b::BoundingVolume) = eltype(b.volume)
Base.eltype(::Type{BoundingVolume{V, I, M}}) where {V, I, M} = eltype(V)


function same_leaf_node(
    leaves::AbstractVector{<:BoundingVolume{V, I, M}},
    nodes::AbstractVector{N}    
) where {V, I, M, N}
    V === N
end


# Sub-includes
include("bsphere.jl")
include("bbox.jl")
include("merge.jl")
include("iscontact.jl")
include("isintersection.jl")
