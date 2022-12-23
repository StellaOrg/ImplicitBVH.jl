using IBVH

using Test
using Random
using StaticArrays


function test_implicit_tree()

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

test_implicit_tree()




function test_bsphere()

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

test_bsphere()




function test_bbox()

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

test_bbox()




function test_morton()

    # Single numbers
    x = UInt32(0x111)
    m = IBVH.morton_split3(x)
    @test m == 0x1001001

    x = UInt64(0x111)
    m = IBVH.morton_split3(x)
    @test m == 0x1001001

    # Random bounding volumes
    Random.seed!(42)

    bv = map(BSphere, [rand(3, 3) for _ in 1:10])
    IBVH.morton_encode(bv, UInt32)
    IBVH.morton_encode(bv, UInt64)
    IBVH.morton_encode(bv)

    bv = map(BBox, [rand(3, 3) for _ in 1:10])
    IBVH.morton_encode(bv, UInt32)
    IBVH.morton_encode(bv, UInt64)
    IBVH.morton_encode(bv)

    bv = map(BSphere{Float32}, [rand(3, 3) for _ in 1:10])
    IBVH.morton_encode(bv, UInt32)
    IBVH.morton_encode(bv, UInt64)
    IBVH.morton_encode(bv)

    bv = map(BBox{Float32}, [rand(3, 3) for _ in 1:10])
    IBVH.morton_encode(bv, UInt32)
    IBVH.morton_encode(bv, UInt64)
    IBVH.morton_encode(bv)

    # Degenerate inputs
    a = BSphere(SVector{3}((0., 0., 0.)), 0.5)
    b = BSphere(SVector{3}((1., 0., 0.)), 0.1)
    IBVH.morton_encode([a, b], UInt32)
    IBVH.morton_encode([a, a], UInt32)
    IBVH.morton_encode([a], UInt32)

end

test_morton()




function test_bvh()

    # Simple bounding spheres traversal test
    bvs = [
        BSphere(SA[0., 0, 0], 0.5),
        BSphere(SA[0., 0, 1], 0.6),
        BSphere(SA[0., 0, 2], 0.5),
        BSphere(SA[0., 0, 3], 0.4),
        BSphere(SA[0., 0, 4], 0.6),
    ]

    bvh = BVH(bvs)
    traversal = traverse(bvh)

    @test length(traversal.contacts) == 3
    @test (4, 5) in traversal.contacts
    @test (1, 2) in traversal.contacts
    @test (2, 3) in traversal.contacts

    # Testing different settings
    BVH(bvs, BBox{Float64})
    BVH(bvs, BBox{Float64}, UInt32)
    BVH(bvs, BBox{Float64}, UInt32, 3)

    traverse(bvh, 3)
    traverse(bvh, 3, traversal)
end

test_bvh()
