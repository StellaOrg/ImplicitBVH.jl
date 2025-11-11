using ImplicitBVH
using ImplicitBVH: BBox, BSphere
import AcceleratedKernels as AK

using Profile
using PProf

# using Metal
# using AtomixMetal

using Random
Random.seed!(0)


function get_interacting_pairs(particle_centers::AbstractMatrix, cutoff::AbstractFloat)

    # Construct bounding sphere around each particle of `cutoff` radius
    num_particles = size(particle_centers, 2)
    bounding_volumes = similar(particle_centers, BSphere{Float32}, num_particles)
    AK.foreachindex(bounding_volumes) do i
        bounding_volumes[i] = BSphere{Float32}(
            (particle_centers[1, i], particle_centers[2, i], particle_centers[3, i]),
            cutoff,
        )
    end

    # Construct BVH, merging BSpheres into BBoxes, and using 32-bit Morton indices
    bvh = BVH(bounding_volumes, BBox{Float32}, UInt32, default_start_level(num_particles))

    # Traverse BVH - this returns a BVHTraversal
    contacting_pairs = traverse(bvh)

    # Return Vector{Tuple{Int32, Int32}} of particle index pairs
    contacting_pairs.contacts
end


particle_centers = rand(Float32, 3, 12_166)
# particle_centers = Float32[
#     0. 1 2 3 4
#     0. 0 0 0 0
#     0. 0 0 0 0
# ]


# particle_centers = MtlArray(particle_centers)
interacting_pairs = get_interacting_pairs(particle_centers, 0.01f0)




# # Collect an allocation profile
# Profile.Allocs.clear()
# Profile.Allocs.@profile get_interacting_pairs(particle_centers, 0.0312f0)
# PProf.Allocs.pprof()



# Example output: 
#   julia> @show interacting_pairs
#   interacting_pairs = Tuple{Int32, Int32}[(369, 667), (427, 974), ...]


# using BenchmarkTools
# @benchmark get_interacting_pairs(particle_centers, 0.0312f0)



