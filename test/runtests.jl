using ImplicitBVH
using ImplicitBVH: BBox, BSphere

using Test
using Random
using LinearAlgebra


@testset "test_utilities" begin

    using ImplicitBVH: minimum2, minimum3, maximum2, maximum3, dot3, dist3sq, dist3

    Random.seed!(42)
    for _ in 1:20
        a, b, c = rand(), rand(), rand()
        @test minimum2(a, b) == minimum([a, b])
        @test maximum2(a, b) == maximum([a, b])

        @test minimum3(a, b, c) == minimum([a, b, c])
        @test maximum3(a, b, c) == maximum([a, b, c])
    end

    for _ in 1:20
        x = (rand(), rand(), rand())
        y = (rand(), rand(), rand())

        @test dot3(x, y) ≈ dot(x, y)
        @test dist3sq(x, y) ≈ dot(x .- y, x .- y)
        @test dist3(x, y) ≈ sqrt(dot(x .- y, x .- y))
    end
end


@testset "test_implicit_tree" begin

    # Perfect, filled tree
    #
    # Level      Nodes
    # 1          1
    # 2     2         3
    # 3   4   5     6   7
    #     ------Real-----
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

    # Incomplete tree with virtual (v) nodes
    #
    # Level                                         Nodes
    # 1                                             1
    # 2                        2                                          3
    # 3            4                     5                     6                       7v
    # 4      8           9         10         11         12         13          14v          15v
    # 5   16   17     18   19    20  21     22  23     24  25     26  27v    28v   29v    30v   31v
    #     --------------------------Real----------------------------- -----------Virtual-----------
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

    Base.isapprox(a::NTuple{3, T}, b) where T = all(isapprox.(a, b))

    # Planar equilateral triangle
    p1 = (0., 0., 0.)
    p2 = (1., 0., 0.)
    p3 = (cosd(60), sind(60), 0.)

    bs = BSphere{Float64}(p1, p2, p3)
    @test bs.x ≈ (p1 .+ p2 .+ p3) ./ 3.
    @test bs.r ≈ 1. / sqrt(3.)

    # Planar right triangle
    p1 = [0., 0., 0.]
    p2 = [0., 1., 0.]
    p3 = [0., 1., 1.]

    bs = BSphere{Float64}(p1, p2, p3)
    @test bs.x ≈ (0., 0.5, 0.5)
    @test bs.r ≈ 1. / sqrt(2.)

    # Points in straight line
    p1 = (0., 0., 0.)
    p2 = (1., 0., 0.)
    p3 = (2., 0., 0.)

    bs = BSphere{Float64}(p1, p2, p3)
    @test bs.x ≈ (1., 0., 0.)
    @test bs.r ≈ 1.

    # Other constructors
    BSphere{Float32}(p1, p2, p3)
    BSphere(p1, p2, p3)
    BSphere{Float32}([p1, p2, p3])
    BSphere([p1, p2, p3])
    BSphere(reshape([p1..., p2..., p3...], 3, 3))

    # Merging two touching spheres
    a = BSphere((0., 0., 0.), 0.5)
    b = BSphere((1., 0., 0.), 0.5)
    c = a + b
    @test c.x ≈ (0.5, 0., 0.)
    @test c.r ≈ 1.

    # Merging when a is inside b
    a = BSphere((0.1, 0., 0.), 0.1)
    b = BSphere((0., 0., 0.), 0.5)
    c = a + b
    @test c.x ≈ b.x
    @test c.r ≈ b.r

    # Merging when b is inside a
    a = BSphere((0., 0., 0.), 0.5)
    b = BSphere((0.1, 0., 0.), 0.1)
    c = a + b
    @test c.x ≈ a.x
    @test c.r ≈ a.r

    # Merging for completely overlapping spheres
    a = BSphere((0., 0., 0.), 0.5)
    c = a + a
    @test c.x ≈ a.x
    @test c.r ≈ a.r

    a = BSphere((1e25, 1e25, 1e25), 0.5)
    c = a + a
    @test c.x ≈ a.x
    @test c.r ≈ a.r

    # Translating
    a = BSphere((0., 0., 0.), 0.5)
    dx = (1., 1., 1.)
    b = ImplicitBVH.translate(a, dx)
    @test b.x ≈ dx
    @test b.r == a.r
end




@testset "test_bbox" begin

    # Cubically-placed points
    p1 = (0., 0., 0.)
    p2 = (1., 1., 0.)
    p3 = (1., 1., 1.)

    bb = BBox{Float64}(p1, p2, p3)
    @test bb.lo ≈ (0., 0., 0.)
    @test bb.up ≈ (1., 1., 1.)

    # Points in straight line
    p1 = [0., 0., 0.]
    p2 = [1., 0., 0.]
    p3 = [2., 0., 0.]

    bb = BBox{Float64}(p1, p2, p3)
    @test bb.lo ≈ (0., 0., 0.)
    @test bb.up ≈ (2., 0., 0.)

    # Other constructors
    BBox{Float32}(p1, p2, p3)
    BBox(p1, p2, p3)
    BBox{Float32}([p1, p2, p3])
    BBox([p1, p2, p3])
    BBox(reshape([p1..., p2..., p3...], 3, 3))

    # Merging two touching boxes
    a = BBox((0., 0., 0.), (1., 1., 1.))
    b = BBox((1., 0., 0.), (2., 1., 1.))
    c = a + b
    @test c.lo ≈ (0., 0., 0.)
    @test c.up ≈ (2., 1., 1.)

    # Merging when a is inside b
    a = BBox((0.1, 0.1, 0.1), (0.2, 0.2, 0.2))
    b = BBox((0., 0., 0.), (1., 1., 1.))
    c = a + b
    @test c.lo ≈ b.lo
    @test c.up ≈ b.up

    # Merging when b is inside a
    a = BBox((0., 0., 0.), (1., 1., 1.))
    b = BBox((0.1, 0.1, 0.1), (0.2, 0.2, 0.2))
    c = a + b
    @test c.lo ≈ a.lo
    @test c.up ≈ a.up

    # Merging for completely overlapping boxes
    a = BBox((0., 0., 0.), (1., 1., 1.))
    c = a + a
    @test c.lo ≈ a.lo
    @test c.up ≈ a.up

    a = BBox((1e-25, 1e-25, 1e-25), (1e25, 1e25, 1e25))
    c = a + a
    @test c.lo ≈ a.lo
    @test c.up ≈ a.up

    # Translating
    a = BBox((0., 0., 0.), (1., 1., 1.))
    dx = (1., 1., 1.)
    b = ImplicitBVH.translate(a, dx)
    @test b.lo ≈ dx
    @test b.up ≈ a.up .+ dx
end


@testset "ray-box isintersection" begin

    using ImplicitBVH: isintersection

    # Below box and ray going through corner
    box = BBox((0., 0., 0.), (1., 1., 1.))
    point = [-1., -1., -1.]
    direction = [1., 1., 1.]
    @test isintersection(box, point, direction) == true

    # Below box and ray going through corner ray direction flipped case
    box = BBox((0., 0., 0.), (1., 1., 1.))
    point = [-1., -1., -1.]
    direction = [-1., -1., -1.]
    @test isintersection(box, point, direction) == false

    # Below box ray going up and through face
    box = BBox((0., 0., 0.), (1., 1., 1.))
    point = [-1., -.5, 0.]
    direction = [5., 3., 1.5]
    @test isintersection(box, point, direction) == true

    # Below box ray going up and through face
    box = BBox((0., 0., 0.), (1., 1., 1.))
    point = [0.5, -0.5, 0.5]
    direction = [0., 1., 0.]
    @test isintersection(box, point, direction) == true

    # Below box ray going up and through face ray direction flipped case
    box = BBox((0., 0., 0.), (1., 1., 1.))
    point = [-1., -.5, 0.]
    direction = [-5., -3., -1.5]
    @test isintersection(box, point, direction) == false

    # Inside box going through upper corner case 
    box = BBox((0., 0., 0.), (1., 1., 1.))
    point = [.5, .5, .5]
    direction = [1., 1., 1.]
    @test isintersection(box, point, direction) == true

    # Inside box going through bottom corner (direction flipped case)
    box = BBox((0., 0., 0.), (1., 1., 1.))
    point = [.5, .5, .5]
    direction = [-1., -1., -1.]
    @test isintersection(box, point, direction) == true

    # Inside box going along face surface
    box = BBox((0., 0., 0.), (1., 1., 1.))
    point = [1e-8, 0, 0.5]
    direction = [0, 1., 0]
    @test isintersection(box, point, direction) == true

    # Outside box going along edge
    box = BBox((0., 0., 0.), (1., 1., 1.))
    point = [1e-8, -1., 1e-8]
    direction = [0, 1., 0]
    @test isintersection(box, point, direction) == true
end


@testset "ray-sphere isintersection" begin

    # ray above sphere passing down and through
    sphere = BSphere((0., 0., 0.), 0.5)
    point = [.5, .5, .5]
    direction = [-1., -1., -1.]
    @test isintersection(sphere, point, direction) == true

    # ray above sphere passing up and not intersecting direction flipped
    sphere = BSphere((0., 0., 0.), 0.5)
    point = [.5, .5, .5]
    direction = [1., 1., 1.]
    @test isintersection(sphere, point, direction) == false

    # ray below sphere passing up and intersecting
    sphere = BSphere((0., 0., 0.), 0.5)
    point = [0., 0., -1.]
    direction = [0., 0., 1.]
    @test isintersection(sphere, point, direction) == true

    # ray below sphere passing and not intersecting
    sphere = BSphere((0., 0., 0.), 0.5)
    point = [0., 0., -1.]
    direction = [0., 0., -1.]
    @test isintersection(sphere, point, direction) == false

    # ray below sphere passing up and tangent to sphere
    sphere = BSphere((0., 0., 0.), 0.5)
    point = [0., 0.5, -1.]
    direction = [0., 0., 1.]
    @test isintersection(sphere, point, direction) == true

    # ray to the side of sphere and passing tangent to sphere
    sphere = BSphere((0., 0., 0.), 0.5)
    point = [0., -1, 0.5]
    direction = [0., 1., 0.]
    @test isintersection(sphere, point, direction) == true

    # ray inside sphere passing up and out
    sphere = BSphere((0., 0., 0.), 0.5)
    point = [0., 0., 0.]
    direction = [0., 0., 1.]
    @test isintersection(sphere, point, direction) == true

    # ray inside sphere passing down and out flipped direction
    sphere = BSphere((0., 0., 0.), 0.5)
    point = [0., 0., 0.]
    direction = [0., 0., -1.]
    @test isintersection(sphere, point, direction) == true

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
    bv = map(BSphere{Float16}, [10 .* rand(3, 3) for _ in 1:100])
    mins, maxs = ImplicitBVH.bounding_volumes_extrema(bv)
    @test all([ImplicitBVH.center(b)[1] > mins[1] for b in bv])
    @test all([ImplicitBVH.center(b)[2] > mins[2] for b in bv])
    @test all([ImplicitBVH.center(b)[3] > mins[3] for b in bv])
    @test all([ImplicitBVH.center(b)[1] < maxs[1] for b in bv])
    @test all([ImplicitBVH.center(b)[2] < maxs[2] for b in bv])
    @test all([ImplicitBVH.center(b)[3] < maxs[3] for b in bv])

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
    bv = [BSphere((0., 0., 0.), 1.)]
    mins, maxs = ImplicitBVH.bounding_volumes_extrema(bv)
    @test all([ImplicitBVH.center(b)[1] > mins[1] for b in bv])
    @test all([ImplicitBVH.center(b)[2] > mins[2] for b in bv])
    @test all([ImplicitBVH.center(b)[3] > mins[3] for b in bv])
    @test all([ImplicitBVH.center(b)[1] < maxs[1] for b in bv])
    @test all([ImplicitBVH.center(b)[2] < maxs[2] for b in bv])
    @test all([ImplicitBVH.center(b)[3] < maxs[3] for b in bv])

    bv = [BSphere((1000., 0., 0.), 1.), BSphere((1000., 0., 0.), 1.)]
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
    # ImplicitBVH.morton_encode(bv, UInt64)     # Range of UInt64 is too high compared to Float16
    # ImplicitBVH.morton_encode(bv)

    bv = map(BBox{Float16}, [rand(3, 3) for _ in 1:10])
    ImplicitBVH.morton_encode(bv, UInt16)
    ImplicitBVH.morton_encode(bv, UInt32)
    # ImplicitBVH.morton_encode(bv, UInt64)     # Range of UInt64 is too high compared to Float16
    # ImplicitBVH.morton_encode(bv)

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
    a = BSphere((0., 0., 0.), 0.5)
    b = BSphere((1., 0., 0.), 0.1)
    ImplicitBVH.morton_encode([a, b], UInt32)
    ImplicitBVH.morton_encode([a, a], UInt32)
    ImplicitBVH.morton_encode([a], UInt32)
end




@testset "bvh_single_bsphere_small_ordered" begin

    using ImplicitBVH: center

    # Simple, ordered bounding spheres traversal test
    bvs = [
        BSphere([0., 0, 0], 0.5),
        BSphere([0., 0, 1], 0.6),
        BSphere([0., 0, 2], 0.5),
        BSphere([0., 0, 3], 0.4),
        BSphere([0., 0, 4], 0.6),
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

    # Test the default start levels
    @test default_start_level(bvh) == default_start_level(length(bvs))

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

    # Build the same BVH with BBox nodes
    leaf = BBox{Float64}
    bvh = BVH(bvs, leaf)
    @test length(bvh.nodes) == 6

    # Level 3
    @test center(bvh.nodes[4]) ≈ center(leaf(bvs[1], bvs[2]))      # First two BVs are paired
    @test center(bvh.nodes[5]) ≈ center(leaf(bvs[3], bvs[4]))      # Next two BVs are paired
    @test center(bvh.nodes[6]) ≈ center(bvs[5])                    # Last BV has no pair

    # Level 2
    @test center(bvh.nodes[2]) ≈ center(leaf(bvs[1], bvs[2]) + leaf(bvs[3], bvs[4]))
    @test center(bvh.nodes[3]) ≈ center(bvs[5])

    # Root
    @test center(bvh.nodes[1]) ≈ center(leaf(bvs[1], bvs[2]) + leaf(bvs[3], bvs[4]) + leaf(bvs[5]))

    # Find contacting pairs
    traversal = traverse(bvh)

    @test length(traversal.contacts) == 3
    @test (4, 5) in traversal.contacts
    @test (1, 2) in traversal.contacts
    @test (2, 3) in traversal.contacts
end




@testset "bvh_single_bbox_small_ordered" begin

    # Simple, ordered bounding box traversal test
    bvs = [
        BBox(BSphere([0., 0, 0], 0.5)),
        BBox(BSphere([0., 0, 1], 0.6)),
        BBox(BSphere([0., 0, 2], 0.5)),
        BBox(BSphere([0., 0, 3], 0.4)),
        BBox(BSphere([0., 0, 4], 0.6)),
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

    # Test the default start levels
    @test default_start_level(bvh) == default_start_level(length(bvs))

    # Level 3
    center = ImplicitBVH.center
    @test center(bvh.nodes[4]) ≈ center(bvs[1] + bvs[2])      # First two BVs are paired
    @test center(bvh.nodes[5]) ≈ center(bvs[3] + bvs[4])      # Next two BVs are paired
    @test center(bvh.nodes[6]) ≈ center(bvs[5])               # Last BV has no pair

    # Level 2
    @test center(bvh.nodes[2]) ≈ center((bvs[1] + bvs[2]) + (bvs[3] + bvs[4]))
    @test center(bvh.nodes[3]) ≈ center(bvs[5])

    # Root
    @test center(bvh.nodes[1]) ≈ center((bvs[1] + bvs[2]) + (bvs[3] + bvs[4]) + bvs[5])

    # Find contacting pairs
    traversal = traverse(bvh)

    @test length(traversal.contacts) == 3
    @test (4, 5) in traversal.contacts
    @test (1, 2) in traversal.contacts
    @test (2, 3) in traversal.contacts
end




@testset "bvh_single_bsphere_small_unordered" begin
    # Bounding spheres traversal test with unordered spheres
    bvs = [
        BSphere([0., 0, 1], 0.6),
        BSphere([0., 0, 2], 0.5),
        BSphere([0., 0, 0], 0.5),
        BSphere([0., 0, 4], 0.6),
        BSphere([0., 0, 3], 0.4),
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

    # Test the default start levels
    @test default_start_level(bvh) == default_start_level(length(bvs))

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

    # Build the same BVH with BBox nodes
    leaf = BBox{Float64}
    bvh = BVH(bvs, leaf)
    @test length(bvh.nodes) == 6

    # Level 3
    center = ImplicitBVH.center
    @test center(bvh.nodes[4]) ≈ center(leaf(bvs[3], bvs[1]))      # First two BVs are paired
    @test center(bvh.nodes[5]) ≈ center(leaf(bvs[2], bvs[5]))      # Next two BVs are paired
    @test center(bvh.nodes[6]) ≈ center(bvs[4])                    # Last BV has no pair

    # Level 2
    @test center(bvh.nodes[2]) ≈ center(leaf(bvs[3], bvs[1]) + leaf(bvs[2], bvs[5]))
    @test center(bvh.nodes[3]) ≈ center(bvs[4])

    # Root
    @test center(bvh.nodes[1]) ≈ center(leaf(bvs[3], bvs[1]) + leaf(bvs[2], bvs[5]) + leaf(bvs[4]))

    # Find contacting pairs
    traversal = traverse(bvh)

    @test length(traversal.contacts) == 3
    @test (4, 5) in traversal.contacts
    @test (1, 3) in traversal.contacts
    @test (1, 2) in traversal.contacts
end




@testset "bvh_single_bbox_small_unordered" begin
    # Bounding spheres traversal test with unordered spheres
    bvs = [
        BBox(BSphere([0., 0, 1], 0.6)),
        BBox(BSphere([0., 0, 2], 0.5)),
        BBox(BSphere([0., 0, 0], 0.5)),
        BBox(BSphere([0., 0, 4], 0.6)),
        BBox(BSphere([0., 0, 3], 0.4)),
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

    # Test the default start levels
    @test default_start_level(bvh) == default_start_level(length(bvs))

    # Level 3
    center = ImplicitBVH.center
    @test center(bvh.nodes[4]) ≈ center(bvs[3] + bvs[1])      # First two BVs are paired
    @test center(bvh.nodes[5]) ≈ center(bvs[2] + bvs[5])      # Next two BVs are paired
    @test center(bvh.nodes[6]) ≈ center(bvs[4])               # Last BV has no pair

    # Level 2
    @test center(bvh.nodes[2]) ≈ center((bvs[3] + bvs[1]) + (bvs[2] + bvs[5]))
    @test center(bvh.nodes[3]) ≈ center(bvs[4])

    # Root
    @test center(bvh.nodes[1]) ≈ center((bvs[3] + bvs[1]) + (bvs[2] + bvs[5]) + bvs[4])

    # Find contacting pairs
    traversal = traverse(bvh)

    @test length(traversal.contacts) == 3
    @test (4, 5) in traversal.contacts
    @test (1, 3) in traversal.contacts
    @test (1, 2) in traversal.contacts
end




@testset "bvh_translate" begin

    using ImplicitBVH: center

    # Simple, ordered bounding spheres traversal test
    bvs = [
        BSphere([0., 0, 0], 0.5),
        BSphere([0., 0, 1], 0.6),
        BSphere([0., 0, 2], 0.5),
        BSphere([0., 0, 3], 0.4),
        BSphere([0., 0, 4], 0.6),
    ]

    # Translate BVH made of BSphere
    bvh = BVH(bvs)
    new_positions = [
        1. 0 0
        1. 0 2
        1. 0 4
        1. 0 6
        1. 0 8
    ]'
    translated = BVH(bvh, new_positions)
    @test center(translated.leaves[1]) ≈ (1, 0, 0)
    @test center(translated.leaves[2]) ≈ (1, 0, 2)
    @test center(translated.leaves[3]) ≈ (1, 0, 4)
    @test center(translated.leaves[4]) ≈ (1, 0, 6)
    @test center(translated.leaves[5]) ≈ (1, 0, 8)

    # Simple, ordered bounding box traversal test
    bvs = [
        BBox(BSphere([0., 0, 0], 0.5)),
        BBox(BSphere([0., 0, 1], 0.6)),
        BBox(BSphere([0., 0, 2], 0.5)),
        BBox(BSphere([0., 0, 3], 0.4)),
        BBox(BSphere([0., 0, 4], 0.6)),
    ]

    # Translate BVH made of BBox
    bvh = BVH(bvs)
    new_positions = [
        1. 0 0
        1. 0 2
        1. 0 4
        1. 0 6
        1. 0 8
    ]'
    translated = BVH(bvh, new_positions)
    @test center(translated.leaves[1]) ≈ (1, 0, 0)
    @test center(translated.leaves[2]) ≈ (1, 0, 2)
    @test center(translated.leaves[3]) ≈ (1, 0, 4)
    @test center(translated.leaves[4]) ≈ (1, 0, 6)
    @test center(translated.leaves[5]) ≈ (1, 0, 8)
end




@testset "bvh_single_randomised" begin
    # Random bounding volumes of different densities; BSphere leaves, BSphere nodes
    Random.seed!(42)

    for num_entities in 1:11:200

        # Test different starting levels
        tree = ImplicitTree(num_entities)
        for start_level in 1:tree.levels
            bvs = map(BSphere, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities])

            # Brute force contact detection
            brute_contacts = ImplicitBVH.IndexPair{Int}[]
            for i in 1:length(bvs)
                for j in i + 1:length(bvs)
                    if ImplicitBVH.iscontact(bvs[i], bvs[j])
                        push!(brute_contacts, (i, j))
                    end
                end
            end

            # ImplicitBVH-based contact detection
            bvh = BVH(bvs)
            traversal = traverse(bvh, start_level)
            bvh_contacts = traversal.contacts

            # Test the default start levels
            @test default_start_level(bvh) == default_start_level(length(bvs))

            # Ensure ImplicitBVH finds same contacts as checking all possible pairs
            @test length(brute_contacts) == length(bvh_contacts)
            @test all(brute_contact in bvh_contacts for brute_contact in brute_contacts)
        end
    end

    # Random bounding volumes of different densities; BSphere leaves, BBox nodes
    Random.seed!(42)
    for num_entities in 1:11:200
        # Test different starting levels
        tree = ImplicitTree(num_entities)
        for start_level in 1:tree.levels
            bvs = map(BSphere, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities])

            # Brute force contact detection
            brute_contacts = ImplicitBVH.IndexPair{Int}[]
            for i in 1:length(bvs)
                for j in i + 1:length(bvs)
                    if ImplicitBVH.iscontact(bvs[i], bvs[j])
                        push!(brute_contacts, (i, j))
                    end
                end
            end

            # ImplicitBVH-based contact detection
            bvh = BVH(bvs, BBox{Float64})
            traversal = traverse(bvh, start_level)
            bvh_contacts = traversal.contacts

            # Test the default start levels
            @test default_start_level(bvh) == default_start_level(length(bvs))

            # Ensure ImplicitBVH finds same contacts as checking all possible pairs
            @test length(brute_contacts) == length(bvh_contacts)
            @test all(brute_contact in bvh_contacts for brute_contact in brute_contacts)
        end
    end

    # Testing different settings
    Random.seed!(42)
    bvs = map(BSphere, [6 * rand(3) .+ rand(3, 3) for _ in 1:100])
    bvh = BVH(bvs)
    traversal = traverse(bvh)

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




@testset "bvh_pair_equivalent_randomised" begin
    # Random bounding volumes of different densities; BSphere leaves, BSphere nodes
    Random.seed!(42)
    for num_entities in 1:11:200

        # Test different starting levels
        tree = ImplicitTree(num_entities)
        for start_level1 in 1:tree.levels, start_level2 in 1:tree.levels
            bvs = map(BSphere, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities])
            bvh = BVH(bvs)

            # Test the default start levels
            @test default_start_level(bvh) == default_start_level(length(bvs))

            # First traverse the BVH normally, then as if we had two different BVHs
            contacts1 = traverse(bvh, start_level1).contacts
            contacts2 = traverse(bvh, bvh, start_level1, start_level2).contacts

            # The second one should have the same contacts as contacts1, plus contacts between the
            # same BVs and reverse order; e.g. if contacts1=[(1, 2), (2, 3)], then
            # contacts2=[(1, 1), (2, 2), (3, 3), (1, 2), (2, 1), (2, 3), (3, 2)]. Check this.
            @test all((i, i) in contacts2 for i in 1:num_entities)
            contacts2 = [(i, j) for (i, j) in contacts2 if i != j]

            @test all((j, i) in contacts2 for (i, j) in contacts2)
            contacts2 = [(i, j) for (i, j) in contacts2 if i < j]

            sort!(contacts1)
            sort!(contacts2)
            @test contacts1 == contacts2
        end
    end

    # Random bounding volumes of different densities; BSphere leaves, BBox nodes
    Random.seed!(42)
    for num_entities in 1:11:200

        # Test different starting levels
        tree = ImplicitTree(num_entities)
        for start_level1 in 1:tree.levels, start_level2 in 1:tree.levels
            bvs = map(BSphere, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities])
            bvh = BVH(bvs, BBox{Float64})

            # Test the default start levels
            @test default_start_level(bvh) == default_start_level(length(bvs))

            # First traverse the BVH normally, then as if we had two different BVHs
            contacts1 = traverse(bvh, start_level1).contacts
            contacts2 = traverse(bvh, bvh, start_level1, start_level2).contacts

            # The second one should have the same contacts as contacts1, plus contacts between the
            # same BVs and reverse order; e.g. if contacts1=[(1, 2), (2, 3)], then
            # contacts2=[(1, 1), (2, 2), (3, 3), (1, 2), (2, 1), (2, 3), (3, 2)]. Check this.
            @test all((i, i) in contacts2 for i in 1:num_entities)
            contacts2 = [(i, j) for (i, j) in contacts2 if i != j]

            @test all((j, i) in contacts2 for (i, j) in contacts2)
            contacts2 = [(i, j) for (i, j) in contacts2 if i < j]

            sort!(contacts1)
            sort!(contacts2)
            @test contacts1 == contacts2
        end
    end
end




@testset "bvh_pair_randomised" begin
    # Random bounding volumes of different densities; BSphere leaves, BSphere nodes
    Random.seed!(42)

    for num_entities1 in 1:21:200, num_entities2 in 1:21:200

        # Test different starting levels
        tree1 = ImplicitTree(num_entities1)
        tree2 = ImplicitTree(num_entities2)

        for start_level1 in 1:tree1.levels, start_level2 in 1:tree2.levels
            bvs1 = map(BSphere, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities1])
            bvs2 = map(BSphere, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities2])

            # Brute force contact detection
            brute_contacts = ImplicitBVH.IndexPair{Int}[]
            for i in 1:length(bvs1)
                for j in 1:length(bvs2)
                    if ImplicitBVH.iscontact(bvs1[i], bvs2[j])
                        push!(brute_contacts, (i, j))
                    end
                end
            end

            # ImplicitBVH-based contact detection
            bvh1 = BVH(bvs1)
            bvh2 = BVH(bvs2)
            traversal = traverse(bvh1, bvh2, start_level1, start_level2)
            bvh_contacts = traversal.contacts

            # Ensure ImplicitBVH finds same contacts as checking all possible pairs
            @test length(brute_contacts) == length(bvh_contacts)
            @test all(brute_contact in bvh_contacts for brute_contact in brute_contacts)
        end
    end

    # Random bounding volumes of different densities; BSphere leaves, BBox nodes
    Random.seed!(42)
    for num_entities1 in 1:21:200, num_entities2 in 1:21:200

        # Test different starting levels
        tree1 = ImplicitTree(num_entities1)
        tree2 = ImplicitTree(num_entities2)
        min_levels = tree1.levels < tree2.levels ? tree1.levels : tree2.levels

        for start_level in 1:min_levels - 1
            bvs1 = map(BSphere, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities1])
            bvs2 = map(BSphere, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities2])

            # Brute force contact detection
            brute_contacts = ImplicitBVH.IndexPair{Int}[]
            for i in 1:length(bvs1)
                for j in 1:length(bvs2)
                    if ImplicitBVH.iscontact(bvs1[i], bvs2[j])
                        push!(brute_contacts, (i, j))
                    end
                end
            end

            # ImplicitBVH-based contact detection
            bvh1 = BVH(bvs1, BBox{Float64})
            bvh2 = BVH(bvs2, BBox{Float64})
            traversal = traverse(bvh1, bvh2, start_level)
            bvh_contacts = traversal.contacts

            # Ensure ImplicitBVH finds same contacts as checking all possible pairs
            @test length(brute_contacts) == length(bvh_contacts)
            @test all(brute_contact in bvh_contacts for brute_contact in brute_contacts)
        end
    end
end


# GPU tests
# Pass command-line argument to test suite to install the right backend, e.g.
#   julia> import Pkg
#   julia> Pkg.test(test_args=["--oneAPI"])
import Pkg

if "--CUDA" in ARGS
    Pkg.add("CUDA")
    using CUDA
    const backend = CUDABackend()
    include(joinpath(@__DIR__, "gputests.jl"))

elseif "--oneAPI" in ARGS
    Pkg.add("oneAPI")
    using oneAPI
    const backend = oneAPIBackend()
    include(joinpath(@__DIR__, "gputests.jl"))

elseif "--AMDGPU" in ARGS
    Pkg.add("AMDGPU")
    using AMDGPU
    const backend = ROCBackend()
    include(joinpath(@__DIR__, "gputests.jl"))

elseif "--Metal" in ARGS
    Pkg.add("Metal")
    using Metal
    const backend = MetalBackend()
    include(joinpath(@__DIR__, "gputests.jl"))
end

