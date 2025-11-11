"""
    $(TYPEDEF)

Morton encoding algorithms; a new `alg` needs to implement:

```julia
morton_encode!(bounding_volumes, alg, options)
eltype(alg) -> Type
```

At the moment, only the canonical [`DefaultMortonAlgorithm`](@ref) is exported; we are prototyping
extended Morton encoding algorithms as well to ideally improve BVH quality and reduce the number of
contact checks during traversal.
"""
abstract type MortonAlgorithm end


"""
    morton_encode!(
        bounding_volumes::AbstractVector{<:BoundingVolume},
        options=BVHOptions(),
    )

Encode each each bounding volume (given as [`BoundingVolume`](@ref)) into Morton codes following
the algorithm set in `options.alg`; the Morton codes are updated inline in the `morton` field of
each bounding volume.
"""
function morton_encode!(
    bounding_volumes::AbstractVector{<:BoundingVolume},
    options::BVHOptions=BVHOptions(),
)
    # Forward call to morton encoding algorithm from options
    check_morton_type(eltype(bounding_volumes), eltype(options.morton))
    morton_encode!(bounding_volumes, options.morton, options)
end


function check_morton_type(::Type{<:BoundingVolume{L, I, M}}, ::Type{N}) where {L, I, M, N}
    if M != N
        throw(ArgumentError("Bounding volume Morton type $M does not match options Morton type $N"))
    end
end


# Sub-includes
include("utils.jl")
include("default.jl")
# include("extended.jl")
