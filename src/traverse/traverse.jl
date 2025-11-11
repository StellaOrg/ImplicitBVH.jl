"""
    const IndexPair{I} = Tuple{I, I}

Alias for a tuple of two indices representing e.g. a contacting pair.
"""
const IndexPair{I} = Tuple{I, I}


"""
    abstract type TraversalAlgorithm end
    struct BFSTraversal <: TraversalAlgorithm end
    struct LVTTraversal <: TraversalAlgorithm end

The algorithm used to traverse one / two BVHs for contact detection and ray-tracing.

In general, use `LVTTraversal` (the default) unless you use few CPU threads and memory usage is
not a bottleneck.

## `BFSTraversal`
Simultaneous breadth-first search traversal - nodes are paired level-by-level:
- Theoretical minimum number of contact checks.
- Much higher memory usage (spiking at around 10-20x the final number of contacts found).
- Faster on CPUs with few threads, but slightly worse scalability.
- More kernel launches (thread launches or enqueues - one per level).

## `LVTTraversal`
Leaf-vs-tree traversal - independent traversal of leaves in one BVH against the entire other BVH:
- More contact checks (around 10x more than `BFSTraversal`).
- Much lower memory usage (only the final number of contacts found, plus 4/8 bytes per CPU/GPU thread).
- Excellent cache locality - can yield ~17 processor cycles per contact check.
- Much faster on GPUs (about 4x faster than `BFSTraversal`), with better scalability due to improved
  memory and work divergence.
- Fewer kernel launches (thread launches or enqueues = 2 passes + `accumulate`).
- About 2-3x slower on CPUs when single-threaded, but ideal scalability with threads.
"""
abstract type TraversalAlgorithm end


"""
    $(TYPEDEF)

Collected BVH traversal `contacts` vector, some stats, plus the two buffers `cache1` and `cache2`
which can be reused for future traversals to minimise memory allocations.

# Fields
- `start_level1::Int`: the level at which the single/pair-tree traversal started for the first BVH.
- `start_level2::Int`: the level at which the pair-tree traversal started for the second BVH.
- `num_checks::Int`: the number of contact checks performed during traversal, not always computed.
- `num_contacts::Int`: the number of contacts found.
- `contacts::view(cache_contacts, 1:num_contacts)`: the contacting pairs found, as a view into `cache1`.
- `cache1::C1{IndexPair} <: AbstractVector`: cache of all contacts found (may have greater size; use num_contacts).
- `cache2::C2{IndexPair} <: AbstractVector`: second cache used, depending on the traversal algorithm.
"""
struct BVHTraversal{C1 <: AbstractVector, C2 <: AbstractVector}
    # Stats
    start_level1::Int
    start_level2::Int
    num_checks::Int

    # Data
    num_contacts::Int
    cache1::C1
    cache2::C2
end


# Constructor in the case of single-tree traversal (e.g. traverse(bvh)), when we only have a
# single start_level
function BVHTraversal(
    start_level::Int,
    num_checks::Int,
    num_contacts::Int,
    cache1::AbstractVector,
    cache2::AbstractVector,
)
    BVHTraversal(start_level, 0, num_checks, num_contacts, cache1, cache2)
end


# Custom pretty-printing
function Base.show(io::IO, t::BVHTraversal{C1, C2}) where {C1, C2}
    print(
        io,
        """
        BVHTraversal
          start_level1:     $(typeof(t.start_level1)) $(t.start_level1)
          start_level2:     $(typeof(t.start_level2)) $(t.start_level2)
          num_checks:       $(typeof(t.num_checks)) $(t.num_checks)
          num_contacts:     $(typeof(t.num_contacts)) $(t.num_contacts)
          contacts:         $(Base.typename(typeof(t.contacts)).wrapper){IndexPair}($(size(t.contacts)))
          cache1:           $C1($(size(t.cache1)))
          cache2:           $C2($(size(t.cache2)))
        """
    )
end


function Base.getproperty(bt::BVHTraversal, sym::Symbol)
   if sym === :contacts
       return @view bt.cache1[1:bt.num_contacts]
   else
       return getfield(bt, sym)
   end
end

Base.propertynames(::BVHTraversal) = (:start_level1, :start_level2, :contacts,
                                      :num_contacts, :cache1, :cache2)


"""
    default_start_level(bvh::BVH, alg::TraversalAlgorithm)::Int

Compute the default start level for BVH traversal given the BVH and the traversal algorithm.
"""
function default_start_level(bvh::BVH, alg)::Int
    throw(ArgumentError("default_start_level not implemented for: $alg"))
end


"""
    traverse(
        bvh::BVH, alg::TraversalAlgorithm=LVTTraversal();
        start_level::Int=default_start_level(bvh, alg),
        narrow=(bv1, bv2) -> true,
        cache::Union{Nothing, BVHTraversal}=nothing,
        options=BVHOptions(),
    )::BVHTraversal

    traverse(
        bvh1::BVH, bvh2::BVH, alg::TraversalAlgorithm=LVTTraversal();
        start_level1::Int=default_start_level(bvh1, alg),
        start_level2::Int=default_start_level(bvh2, alg),
        narrow=(bv1, bv2) -> true,
        cache::Union{Nothing, BVHTraversal}=nothing,
        options=BVHOptions(),
    )::BVHTraversal

Traverse `bvh` downwards from `start_level`, returning all contacting bounding volume leaves, using
the traversal algorithm `alg` ([`TraversalAlgorithm`](@ref)). The returned [`BVHTraversal`](@ref)
also contains two buffers that can be reused on future traversals as cache to avoid reallocations.

The optional `narrow` function can be used to perform a custom narrow-phase test between two
bounding volumes `bv1` and `bv2` before registering a contact. By default, all bounding volume
pairs reaching the narrow-phase are considered contacting.

# Examples

```jldoctest
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
bvh = BVH(bounding_spheres, BBox{Float32})

# Traverse BVH for contact detection
traversal = traverse(bvh)

# Reuse traversal buffers for future contact detection - possibly with different BVHs
traversal = traverse(bvh, cache=traversal)
@show traversal.contacts;
;

# output
traversal.contacts = Tuple{Int32, Int32}[(1, 2), (2, 3), (4, 5)]
```

Traverse `bvh1` and `bvh2` downwards from `start_level1` and `start_level2`, returning all
contacting bounding volume leaves:
```jldoctest
using ImplicitBVH
using ImplicitBVH: BBox, BSphere

# Generate some simple bounding spheres
bounding_spheres1 = [
    BSphere{Float32}([0., 0., 0.], 0.5),
    BSphere{Float32}([0., 0., 3.], 0.4),
]

bounding_spheres2 = [
    BSphere{Float32}([0., 0., 1.], 0.6),
    BSphere{Float32}([0., 0., 2.], 0.5),
    BSphere{Float32}([0., 0., 4.], 0.6),
]

# Build BVHs
bvh1 = BVH(bounding_spheres1, BBox{Float32})
bvh2 = BVH(bounding_spheres2, BBox{Float32})

# Traverse BVH for contact detection
traversal = traverse(bvh1, bvh2, start_level1=2, start_level2=3)

# Reuse traversal buffers for future contact detection - possibly with different BVHs
traversal = traverse(bvh1, bvh2, cache=traversal)
@show traversal.contacts;
;

# output
traversal.contacts = Tuple{Int32, Int32}[(1, 1), (2, 3)]
```
"""
function traverse(
    bvh::BVH, alg::TraversalAlgorithm=LVTTraversal();
    start_level::Int=default_start_level(bvh, alg),
    narrow=(bv1, bv2) -> true,
    cache::Union{Nothing, BVHTraversal}=nothing,
    options=BVHOptions(),
)
    throw(ArgumentError("Traversal algorithm not implemented: $alg"))
end


function traverse(
    bvh1::BVH, bvh2::BVH, alg::TraversalAlgorithm=LVTTraversal();
    start_level1::Int=default_start_level(bvh1, alg),
    start_level2::Int=default_start_level(bvh2, alg),
    narrow=(bv1, bv2) -> true,
    cache::Union{Nothing, BVHTraversal}=nothing,
    options=BVHOptions(),
)
    throw(ArgumentError("Traversal algorithm not implemented: $alg"))
end


# Old interfaces
function traverse(
    bvh::BVH,
    start_level::Int,
    cache::Union{Nothing, BVHTraversal}=nothing;
    options=BVHOptions(),
)
    traverse(bvh, BFSTraversal(); start_level=start_level, cache=cache, options=options)
end


function traverse(
    bvh1::BVH, bvh2::BVH,
    start_level1,
    start_level2::Int=default_start_level(bvh2, BFSTraversal()),
    cache::Union{Nothing, BVHTraversal}=nothing;
    options=BVHOptions(),
)
    traverse(
        bvh1, bvh2, BFSTraversal();
        start_level1=start_level1, start_level2=start_level2,
        cache=cache, options=options,
    )
end


# Sub-includes
include("breadth_first/breadth_first.jl")
# include("depth_first/depth_first.jl")
include("leaf_vs_tree/leaf_vs_tree.jl")
