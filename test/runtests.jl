using ImplicitBVH
using ImplicitBVH: BBox, BSphere

using Test
using Random
using StaticArrays


@testset "test_implicit_tree" begin

    # Perfect, filled tree
    #          1
    #     2         3
    #   4   5     6   7
    #   ------Real-----
    tree = ImplicitTree(4)
    @test tree.levels == 3
    @test tree.real_leaves == 4
    @test tree.virtual_leaves == 0
    @test tree.real_nodes == 7
    @test tree.virtual_nodes == 0
    @test memory_index(tree, 1) == 1
    @test memory_index(tree, 7) == 7
    @test level_indices(tree, 1) == (1, 1)
    @test level_indices(tree, 2) == (2, 3)
    @test level_indices(tree, 3) == (4, 7)
    @test isvirtual(tree, 1) == false
    @test isvirtual(tree, 7) == false

    # Incomplete tree with virtual nodes
    #                                             1
    #                        2                                          3
    #            4                     5                     6                       7v
    #      8           9         10         11         12         13          14v          15v
    #   16   17     18   19    20  21     22  23     24  25     26  27v    28v   29v    30v   31v
    #   -------------------------Real-----------------------------  -----------Virtual-----------
    tree = ImplicitTree(11)
    @test tree.levels == 5
    @test tree.real_leaves == 11
    @test tree.virtual_leaves == 5
    @test tree.real_nodes == 23
    @test tree.virtual_nodes == 8
    @test memory_index(tree, 1) == 1
    @test memory_index(tree, 8) == 7
    @test memory_index(tree, 16) == 13
    @test level_indices(tree, 1) == (1, 1)
    @test level_indices(tree, 3) == (4, 6)
    @test level_indices(tree, 5) == (13, 23)
    @test isvirtual(tree, 6) == false
    @test isvirtual(tree, 7) == true
    @test isvirtual(tree, 26) == false
    @test isvirtual(tree, 27) == true
    @test isvirtual(tree, 31) == true

    # Trees with different integer types
    @test ImplicitTree{Int32}(11).real_nodes isa Int32
    @test ImplicitTree{UInt32}(11).real_nodes isa UInt32

end




@testset "test_bsphere" begin

    # Planar equilateral triangle
    p1 = SVector{3}((0., 0., 0.))
    p2 = SVector{3}((1., 0., 0.))
    p3 = SVector{3}((cosd(60), sind(60), 0.))

    bs = BSphere{Float64}(p1, p2, p3)
    @test bs.x ≈ (p1 + p2 + p3) / 3.
    @test bs.r ≈ 1. / sqrt(3.)

    # Planar right triangle
    p1 = SVector{3}((0., 0., 0.))
    p2 = SVector{3}((0., 1., 0.))
    p3 = SVector{3}((0., 1., 1.))

    bs = BSphere{Float64}(p1, p2, p3)
    @test bs.x ≈ SVector{3}((0., 0.5, 0.5))
    @test bs.r ≈ 1. / sqrt(2.)

    # Points in straight line
    p1 = SVector{3}((0., 0., 0.))
    p2 = SVector{3}((1., 0., 0.))
    p3 = SVector{3}((2., 0., 0.))

    bs = BSphere{Float64}(p1, p2, p3)
    @test bs.x ≈ SVector{3}((1., 0., 0.))
    @test bs.r ≈ 1.

    # Other constructors
    BSphere{Float32}(p1, p2, p3)
    BSphere(p1, p2, p3)
    BSphere{Float32}([p1, p2, p3])
    BSphere([p1, p2, p3])
    BSphere(hcat(p1, p2, p3))

    # Merging two touching spheres
    a = BSphere(SVector{3}((0., 0., 0.)), 0.5)
    b = BSphere(SVector{3}((1., 0., 0.)), 0.5)
    c = a + b
    @test c.x ≈ SVector{3}((0.5, 0., 0.))
    @test c.r ≈ 1.

    # Merging when a is inside b
    a = BSphere(SVector{3}((0.1, 0., 0.)), 0.1)
    b = BSphere(SVector{3}((0., 0., 0.)), 0.5)
    c = a + b
    @test c.x ≈ b.x
    @test c.r ≈ b.r

    # Merging when b is inside a
    a = BSphere(SVector{3}((0., 0., 0.)), 0.5)
    b = BSphere(SVector{3}((0.1, 0., 0.)), 0.1)
    c = a + b
    @test c.x ≈ a.x
    @test c.r ≈ a.r

    # Merging for completely overlapping spheres
    a = BSphere(SVector{3}((0., 0., 0.)), 0.5)
    c = a + a
    @test c.x ≈ a.x
    @test c.r ≈ a.r

    a = BSphere(SVector{3}((1e25, 1e25, 1e25)), 0.5)
    c = a + a
    @test c.x ≈ a.x
    @test c.r ≈ a.r

end




@testset "test_bbox" begin

    # Cubically-placed points
    p1 = SVector{3}((0., 0., 0.))
    p2 = SVector{3}((1., 1., 0.))
    p3 = SVector{3}((1., 1., 1.))

    bb = BBox{Float64}(p1, p2, p3)
    @test bb.lo ≈ SVector{3}((0., 0., 0.))
    @test bb.up ≈ SVector{3}((1., 1., 1.))

    # Points in straight line
    p1 = SVector{3}((0., 0., 0.))
    p2 = SVector{3}((1., 0., 0.))
    p3 = SVector{3}((2., 0., 0.))

    bb = BBox{Float64}(p1, p2, p3)
    @test bb.lo ≈ SVector{3}((0., 0., 0.))
    @test bb.up ≈ SVector{3}((2., 0., 0.))

    # Other constructors
    BBox{Float32}(p1, p2, p3)
    BBox(p1, p2, p3)
    BBox{Float32}([p1, p2, p3])
    BBox([p1, p2, p3])
    BBox(hcat(p1, p2, p3))

    # Merging two touching boxes
    a = BBox(SVector{3}((0., 0., 0.)), SVector{3}((1., 1., 1.)))
    b = BBox(SVector{3}((1., 0., 0.)), SVector{3}((2., 1., 1.)))
    c = a + b
    @test c.lo ≈ SVector{3}((0., 0., 0.))
    @test c.up ≈ SVector{3}((2., 1., 1.))

    # Merging when a is inside b
    a = BBox(SVector{3}((0.1, 0.1, 0.1)), SVector{3}((0.2, 0.2, 0.2)))
    b = BBox(SVector{3}((0., 0., 0.)), SVector{3}((1., 1., 1.)))
    c = a + b
    @test c.lo ≈ b.lo
    @test c.up ≈ b.up

    # Merging when b is inside a
    a = BBox(SVector{3}((0., 0., 0.)), SVector{3}((1., 1., 1.)))
    b = BBox(SVector{3}((0.1, 0.1, 0.1)), SVector{3}((0.2, 0.2, 0.2)))
    c = a + b
    @test c.lo ≈ a.lo
    @test c.up ≈ a.up

    # Merging for completely overlapping boxes
    a = BBox(SVector{3}((0., 0., 0.)), SVector{3}((1., 1., 1.)))
    c = a + a
    @test c.lo ≈ a.lo
    @test c.up ≈ a.up

    a = BBox(SVector{3}((1e-25, 1e-25, 1e-25)), SVector{3}((1e25, 1e25, 1e25)))
    c = a + a
    @test c.lo ≈ a.lo
    @test c.up ≈ a.up

end




@testset "test_morton" begin

    # Single numbers
    x = UInt16(0b111)
    m = ImplicitBVH.morton_split3(x)
    @test m == 0b1001001

    x = UInt32(0b111)
    m = ImplicitBVH.morton_split3(x)
    @test m == 0b1001001

    x = UInt64(0b111)
    m = ImplicitBVH.morton_split3(x)
    @test m == 0b1001001

    # Random bounding volumes
    Random.seed!(42)

    # Extrema computed at different precisions
    #
    # TODO: BSphere{Float16} returns NaNs
    # bv = map(BSphere{Float16}, [100 .* rand(3, 3) for _ in 1:100])
    # display(bv)
    # mins, maxs = ImplicitBVH.bounding_volumes_extrema(bv)
    # @test all([ImplicitBVH.center(b)[1] > mins[1] for b in bv])
    # @test all([ImplicitBVH.center(b)[2] > mins[2] for b in bv])
    # @test all([ImplicitBVH.center(b)[3] > mins[3] for b in bv])
    # @test all([ImplicitBVH.center(b)[1] < maxs[1] for b in bv])
    # @test all([ImplicitBVH.center(b)[2] < maxs[2] for b in bv])
    # @test all([ImplicitBVH.center(b)[3] < maxs[3] for b in bv])

    bv = map(BSphere{Float32}, [1000 .* rand(3, 3) for _ in 1:100])
    mins, maxs = ImplicitBVH.bounding_volumes_extrema(bv)
    @test all([ImplicitBVH.center(b)[1] > mins[1] for b in bv])
    @test all([ImplicitBVH.center(b)[2] > mins[2] for b in bv])
    @test all([ImplicitBVH.center(b)[3] > mins[3] for b in bv])
    @test all([ImplicitBVH.center(b)[1] < maxs[1] for b in bv])
    @test all([ImplicitBVH.center(b)[2] < maxs[2] for b in bv])
    @test all([ImplicitBVH.center(b)[3] < maxs[3] for b in bv])

    bv = map(BSphere{Float64}, [1000 .* rand(3, 3) for _ in 1:100])
    mins, maxs = ImplicitBVH.bounding_volumes_extrema(bv)
    @test all([ImplicitBVH.center(b)[1] > mins[1] for b in bv])
    @test all([ImplicitBVH.center(b)[2] > mins[2] for b in bv])
    @test all([ImplicitBVH.center(b)[3] > mins[3] for b in bv])
    @test all([ImplicitBVH.center(b)[1] < maxs[1] for b in bv])
    @test all([ImplicitBVH.center(b)[2] < maxs[2] for b in bv])
    @test all([ImplicitBVH.center(b)[3] < maxs[3] for b in bv])

    # Extrema computed for degenerate inputs
    bv = [BSphere(SA[0., 0, 0], 1.)]
    mins, maxs = ImplicitBVH.bounding_volumes_extrema(bv)
    @test all([ImplicitBVH.center(b)[1] > mins[1] for b in bv])
    @test all([ImplicitBVH.center(b)[2] > mins[2] for b in bv])
    @test all([ImplicitBVH.center(b)[3] > mins[3] for b in bv])
    @test all([ImplicitBVH.center(b)[1] < maxs[1] for b in bv])
    @test all([ImplicitBVH.center(b)[2] < maxs[2] for b in bv])
    @test all([ImplicitBVH.center(b)[3] < maxs[3] for b in bv])

    bv = [BSphere(SA[1000., 0, 0], 1.), BSphere(SA[1000., 0, 0], 1.)]
    mins, maxs = ImplicitBVH.bounding_volumes_extrema(bv)
    @test all([ImplicitBVH.center(b)[1] > mins[1] for b in bv])
    @test all([ImplicitBVH.center(b)[2] > mins[2] for b in bv])
    @test all([ImplicitBVH.center(b)[3] > mins[3] for b in bv])
    @test all([ImplicitBVH.center(b)[1] < maxs[1] for b in bv])
    @test all([ImplicitBVH.center(b)[2] < maxs[2] for b in bv])
    @test all([ImplicitBVH.center(b)[3] < maxs[3] for b in bv])

    # Different morton code sizes
    bv = map(BSphere, [rand(3, 3) for _ in 1:10])
    ImplicitBVH.morton_encode(bv, UInt16)
    ImplicitBVH.morton_encode(bv, UInt32)
    ImplicitBVH.morton_encode(bv, UInt64)
    ImplicitBVH.morton_encode(bv)

    bv = map(BBox, [rand(3, 3) for _ in 1:10])
    ImplicitBVH.morton_encode(bv, UInt16)
    ImplicitBVH.morton_encode(bv, UInt32)
    ImplicitBVH.morton_encode(bv, UInt64)
    ImplicitBVH.morton_encode(bv)

    bv = map(BSphere{Float16}, [rand(3, 3) for _ in 1:10])
    ImplicitBVH.morton_encode(bv, UInt16)
    ImplicitBVH.morton_encode(bv, UInt32)
    ImplicitBVH.morton_encode(bv, UInt64)
    ImplicitBVH.morton_encode(bv)

    bv = map(BBox{Float16}, [rand(3, 3) for _ in 1:10])
    ImplicitBVH.morton_encode(bv, UInt16)
    ImplicitBVH.morton_encode(bv, UInt32)
    ImplicitBVH.morton_encode(bv, UInt64)
    ImplicitBVH.morton_encode(bv)

    bv = map(BSphere{Float32}, [rand(3, 3) for _ in 1:10])
    ImplicitBVH.morton_encode(bv, UInt16)
    ImplicitBVH.morton_encode(bv, UInt32)
    ImplicitBVH.morton_encode(bv, UInt64)
    ImplicitBVH.morton_encode(bv)

    bv = map(BBox{Float32}, [rand(3, 3) for _ in 1:10])
    ImplicitBVH.morton_encode(bv, UInt16)
    ImplicitBVH.morton_encode(bv, UInt32)
    ImplicitBVH.morton_encode(bv, UInt64)
    ImplicitBVH.morton_encode(bv)

    # Degenerate inputs
    a = BSphere(SVector{3}((0., 0., 0.)), 0.5)
    b = BSphere(SVector{3}((1., 0., 0.)), 0.1)
    ImplicitBVH.morton_encode([a, b], UInt32)
    ImplicitBVH.morton_encode([a, a], UInt32)
    ImplicitBVH.morton_encode([a], UInt32)
end




@testset "test_bvh" begin

    # Simple, ordered bounding spheres traversal test
    bvs = [
        BSphere(SA[0., 0, 0], 0.5),
        BSphere(SA[0., 0, 1], 0.6),
        BSphere(SA[0., 0, 2], 0.5),
        BSphere(SA[0., 0, 3], 0.4),
        BSphere(SA[0., 0, 4], 0.6),
    ]

    # Build the following ImplicitBVH from 5 bounding volumes:
    #
    #         Nodes & Leaves                Tree Level
    #               1                       1
    #       2               3               2
    #   4       5       6        7v         3
    # 8   9   10 11   12 13v  14v  15v      4
    bvh = BVH(bvs)
    @test length(bvh.nodes) == 6

    # Level 3
    @test bvh.nodes[4].x ≈ (bvs[1] + bvs[2]).x      # First two BVs are paired
    @test bvh.nodes[5].x ≈ (bvs[3] + bvs[4]).x      # Next two BVs are paired
    @test bvh.nodes[6].x ≈ bvs[5].x                 # Last BV has no pair

    # Level 2
    @test bvh.nodes[2].x ≈ ((bvs[1] + bvs[2]) + (bvs[3] + bvs[4])).x
    @test bvh.nodes[3].x ≈ bvs[5].x

    # Root
    @test bvh.nodes[1].x ≈ ((bvs[1] + bvs[2]) + (bvs[3] + bvs[4]) + bvs[5]).x

    # Find contacting pairs
    traversal = traverse(bvh)

    @test length(traversal.contacts) == 3
    @test (4, 5) in traversal.contacts
    @test (1, 2) in traversal.contacts
    @test (2, 3) in traversal.contacts

    # Bounding spheres traversal test with unordered spheres
    bvs = [
        BSphere(SA[0., 0, 1], 0.6),
        BSphere(SA[0., 0, 2], 0.5),
        BSphere(SA[0., 0, 0], 0.5),
        BSphere(SA[0., 0, 4], 0.6),
        BSphere(SA[0., 0, 3], 0.4),
    ]

    # Build the following ImplicitBVH from 5 bounding volumes:
    #
    #         Nodes & Leaves                Tree Level
    #               1                       1
    #       2               3               2
    #   4       5       6        7v         3
    # 8   9   10 11   12 13v  14v  15v      4
    bvh = BVH(bvs)
    @test length(bvh.nodes) == 6

    # Level 3
    @test bvh.nodes[4].x ≈ (bvs[3] + bvs[1]).x      # First two BVs are paired
    @test bvh.nodes[5].x ≈ (bvs[2] + bvs[5]).x      # Next two BVs are paired
    @test bvh.nodes[6].x ≈ bvs[4].x                 # Last BV has no pair

    # Level 2
    @test bvh.nodes[2].x ≈ ((bvs[3] + bvs[1]) + (bvs[2] + bvs[5])).x
    @test bvh.nodes[3].x ≈ bvs[4].x

    # Root
    @test bvh.nodes[1].x ≈ ((bvs[3] + bvs[1]) + (bvs[2] + bvs[5]) + bvs[4]).x

    # Find contacting pairs
    traversal = traverse(bvh)

    @test length(traversal.contacts) == 3
    @test (4, 5) in traversal.contacts
    @test (1, 3) in traversal.contacts
    @test (1, 2) in traversal.contacts

    # Random bounding volumes; fairly dense, about 45 out of 100 are in contact
    Random.seed!(42)
    bvs = map(BSphere, [6 * rand(3) .+ rand(3, 3) for _ in 1:100])

    # Brute force contact detection
    brute_contacts = ImplicitBVH.IndexPair[]
    for i in 1:length(bvs)
        for j in i + 1:length(bvs)
            if ImplicitBVH.iscontact(bvs[i], bvs[j])
                push!(brute_contacts, (i, j))
            end
        end
    end

    # ImplicitBVH-based contact detection
    bvh = BVH(bvs, BBox{Float64})
    traversal = traverse(bvh)
    bvh_contacts = traversal.contacts

    # Ensure ImplicitBVH finds same contacts as checking all possible pairs
    @test length(brute_contacts) == length(bvh_contacts)
    for brute_contact in brute_contacts
        @test brute_contact in bvh_contacts
    end

    # Testing different settings
    BVH(bvs, BSphere{Float64})
    BVH(bvs, BBox{Float64})
    BVH(bvs, BBox{Float64}, UInt32)
    BVH(bvs, BBox{Float64}, UInt32, 3)
    BVH(bvs, BBox{Float64}, UInt32, 0.0)
    BVH(bvs, BBox{Float64}, UInt32, 0.5)
    BVH(bvs, BBox{Float64}, UInt32, 1.0)

    traverse(bvh, 3)
    traverse(bvh, 3, traversal)
end
