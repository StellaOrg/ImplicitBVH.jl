
using Random

using BenchmarkTools

using ImplicitBVH
using ImplicitBVH: BSphere, BBox

using Profile
using PProf


Random.seed!(0)

num_bvs = 100_000
bvs = map(BSphere{Float32}, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_bvs])

options = BVHOptions(block_size=128, num_threads=8)
bvh = BVH(bvs, BBox{Float32}, UInt32, 1, options=options)


num_rays = 100_000
points = 100 * rand(3, num_rays) .+ rand(Float32, 3, num_rays)
directions = rand(Float32, 3, num_rays)


bvtt1, bvtt2, num_bvtt = ImplicitBVH.initial_bvtt(
    bvh,
    points,
    directions,
    2,
    nothing,
    options,
)

bvt = traverse_rays(bvh, points, directions)

for (ibv, iray) in bvt.contacts
    @assert ImplicitBVH.isintersection(bvs[ibv], points[:, iray], directions[:, iray])
end


# TODO add example to README
# TODO maybe a pretty image / render?
# TODO benchmark against a standard raytracer? Check Chitalu's paper; the Stanford bunny example

