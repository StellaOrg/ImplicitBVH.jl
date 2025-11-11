# The CPU tests have "first-principles" correctness tests, against known correct results. If these
# work, then it is sufficient for the GPU implementations to be checked against the CPU results.
#
# Called from runtests.jl which sets the following, e.g. for CUDA:
# Pkg.add("CUDA")
# using CUDA
# const backend = CUDABackend()
println("Running GPU tests on backend $backend")
using KernelAbstractions


function array_from_host(h_arr::AbstractArray)
    exemplar = KernelAbstractions.zeros(backend, UInt32, 1)
    d_arr = similar(exemplar, eltype(h_arr), size(h_arr))
    copyto!(d_arr, h_arr)
    d_arr
end


@testset "wrap_bounding_volumes_$(backend)" begin
    bvs = map(BSphere{Float32}, [rand(3, 3) for _ in 1:10])
    bvs_gpu = array_from_host(bvs)
    options = BVHOptions(morton=DefaultMortonAlgorithm(UInt32))
    wbvs = ImplicitBVH.wrap_bounding_volumes(bvs, options)
    @test length(wbvs) == length(bvs)

    wbvs_host = Array(wbvs)
    @test all([b.index == i for (i, b) in enumerate(wbvs_host)])
    @test all([b.volume == bvs[i] for (i, b) in enumerate(wbvs_host)])
    @test all([b.morton isa UInt32 for b in wbvs_host])
end


@testset "mortons_gpu_$(backend)" begin
    # This tests BV bounds computation too
    Random.seed!(42)
    for num_entities in 1:200
        bvs = map(BSphere{Float32}, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities])
        options = BVHOptions(morton=DefaultMortonAlgorithm(UInt32))
        wbvs = ImplicitBVH.wrap_bounding_volumes(bvs, options)
        wbvs_gpu = array_from_host(wbvs)

        mortons = ImplicitBVH.morton_encode!(wbvs, options)
        mortons_gpu = ImplicitBVH.morton_encode!(wbvs_gpu, options)

        @test all(mortons .== Array(mortons_gpu))
    end
end


for alg in (BFSTraversal(), LVTTraversal())
    @testset "bvh_gpu_$(backend)_single_$(alg)_randomised" begin
        # Random bounding volumes of different densities; BSphere leaves, BSphere nodes
        Random.seed!(42)

        for num_entities in 1:11:200

            # Test different starting levels
            tree = ImplicitTree(num_entities)
            for start_level in 1:tree.levels
                bvs = map(BSphere{Float32}, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities])
                bvs_gpu = array_from_host(bvs)

                # ImplicitBVH-based contact detection
                bvh = BVH(bvs)
                bvh_gpu = BVH(bvs_gpu)

                traversal = traverse(bvh, alg, start_level=start_level)
                traversal_gpu = traverse(bvh_gpu, alg, start_level=start_level)

                # Compare sorted contact pair indices on the host
                bvh_contacts = traversal.contacts
                sort!(bvh_contacts)

                bvh_contacts_gpu = Array(traversal_gpu.contacts)
                sort!(bvh_contacts_gpu)

                @test all(bvh_contacts .== bvh_contacts_gpu)
            end
        end

        # Random bounding volumes of different densities; BSphere leaves, BBox nodes
        Random.seed!(42)
        for num_entities in 1:11:200
            # Test different starting levels
            tree = ImplicitTree(num_entities)
            for start_level in 1:tree.levels
                bvs = map(BSphere{Float32}, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities])
                bvs_gpu = array_from_host(bvs)

                # ImplicitBVH-based contact detection
                bvh = BVH(bvs, BBox{Float32})
                bvh_gpu = BVH(bvs_gpu, BBox{Float32})

                traversal = traverse(bvh, alg, start_level=start_level)
                traversal_gpu = traverse(bvh_gpu, alg, start_level=start_level)

                # Compare sorted contact pair indices on the host
                bvh_contacts = traversal.contacts
                sort!(bvh_contacts)

                bvh_contacts_gpu = Array(traversal_gpu.contacts)
                sort!(bvh_contacts_gpu)

                @test all(bvh_contacts .== bvh_contacts_gpu)
            end
        end

        # Testing different settings
        Random.seed!(42)
        bvs = array_from_host(map(BSphere{Float32}, [6 * rand(3) .+ rand(3, 3) for _ in 1:100]))
        bvh = BVH(bvs)
        traversal = traverse(bvh, alg)

        BVH(bvs, BSphere{Float32})
        BVH(bvs, BBox{Float32})
        BVH(bvs, BBox{Float32}, options=BVHOptions(morton=DefaultMortonAlgorithm(UInt32)))
        BVH(bvs, BBox{Float32}, options=BVHOptions(morton=DefaultMortonAlgorithm(UInt32)), built_level=3)
        BVH(bvs, BBox{Float32}, options=BVHOptions(morton=DefaultMortonAlgorithm(UInt32)), built_level=0.0)
        BVH(bvs, BBox{Float32}, options=BVHOptions(morton=DefaultMortonAlgorithm(UInt32)), built_level=0.5)
        BVH(bvs, BBox{Float32}, options=BVHOptions(morton=DefaultMortonAlgorithm(UInt32)), built_level=1.0)
        BVH(bvs, BBox{Float32}, options=BVHOptions(morton=DefaultMortonAlgorithm(UInt32)), built_level=3, cache=bvh)

        traverse(bvh, alg, start_level=3)
        traverse(bvh, alg, start_level=3, cache=traversal)
    end
end




for alg in (BFSTraversal(), LVTTraversal())
    @testset "bvh_gpu_$(backend)_pair_$(alg)_randomised" begin
        # Random bounding volumes of different densities; BSphere leaves, BSphere nodes
        Random.seed!(42)

        for num_entities1 in 1:41:200, num_entities2 in 1:41:200

            # Test different starting levels
            tree1 = ImplicitTree(num_entities1)
            tree2 = ImplicitTree(num_entities2)

            for start_level1 in 1:tree1.levels, start_level2 in 1:tree2.levels
                bvs1 = map(BSphere{Float32}, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities1])
                bvs2 = map(BSphere{Float32}, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities2])

                bvs_gpu1 = array_from_host(bvs1)
                bvs_gpu2 = array_from_host(bvs2)

                # ImplicitBVH-based contact detection
                bvh1 = BVH(bvs1)
                bvh2 = BVH(bvs2)

                bvh_gpu1 = BVH(bvs_gpu1)
                bvh_gpu2 = BVH(bvs_gpu2)

                traversal = traverse(bvh1, bvh2, alg; start_level1, start_level2)
                traversal_gpu = traverse(bvh_gpu1, bvh_gpu2, alg; start_level1, start_level2)

                # Compare sorted contact pair indices on the host
                bvh_contacts = traversal.contacts
                sort!(bvh_contacts)

                bvh_contacts_gpu = Array(traversal_gpu.contacts)
                sort!(bvh_contacts_gpu)

                @test all(bvh_contacts .== bvh_contacts_gpu)
            end
        end

        # Random bounding volumes of different densities; BSphere leaves, BBox nodes
        Random.seed!(42)
        for num_entities1 in 1:41:200, num_entities2 in 1:41:200

            # Test different starting levels
            tree1 = ImplicitTree(num_entities1)
            tree2 = ImplicitTree(num_entities2)
            min_levels = tree1.levels < tree2.levels ? tree1.levels : tree2.levels

            for start_level in 1:min_levels - 1
                bvs1 = map(BSphere{Float32}, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities1])
                bvs2 = map(BSphere{Float32}, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities2])

                bvs_gpu1 = array_from_host(bvs1)
                bvs_gpu2 = array_from_host(bvs2)

                # ImplicitBVH-based contact detection
                bvh1 = BVH(bvs1, BBox{Float32})
                bvh2 = BVH(bvs2, BBox{Float32})

                bvh_gpu1 = BVH(bvs_gpu1, BBox{Float32})
                bvh_gpu2 = BVH(bvs_gpu2, BBox{Float32})

                traversal = traverse(bvh1, bvh2, alg, start_level1=start_level, start_level2=start_level)
                traversal_gpu = traverse(bvh_gpu1, bvh_gpu2, alg; start_level1=start_level, start_level2=start_level)

                # Compare sorted contact pair indices on the host
                bvh_contacts = traversal.contacts
                sort!(bvh_contacts)

                bvh_contacts_gpu = Array(traversal_gpu.contacts)
                sort!(bvh_contacts_gpu)

                @test all(bvh_contacts .== bvh_contacts_gpu)
            end
        end
    end
end


for alg in (BFSTraversal(), LVTTraversal())
    @testset "bvh_gpu_$(backend)_ray_$(alg)_traversal" begin
        Random.seed!(42)

        # Generate some random rays
        for num_rays in 0:50:200
            points = 10f0 * rand(Float32, 3, num_rays)
            theta = Float32(π) * rand(Float32, num_rays)
            phi = Float32(2 * π) * rand(Float32, num_rays)
            directions = permutedims(stack((
                sin.(theta) .* cos.(phi),
                sin.(theta) .* sin.(phi),
                cos.(theta),
            )))

            points_gpu = array_from_host(points)
            directions_gpu = array_from_host(directions)

            # Random bounding volumes of different densities; BSphere leaves, BBox nodes
            bvs = map(BSphere{Float32}, [6 * rand(3) .+ rand(3, 3) for _ in 1:100])
            bvs_gpu = array_from_host(bvs)
            bvh = BVH(bvs, BBox{Float32})
            bvh_gpu = BVH(bvs_gpu, BBox{Float32})

            traversal = traverse_rays(bvh, points, directions, alg)
            traversal_gpu = traverse_rays(bvh_gpu, points_gpu, directions_gpu, alg)

            # Compare sorted contact pair indices on the host
            bvh_contacts = traversal.contacts
            sort!(bvh_contacts)

            bvh_contacts_gpu = Array(traversal_gpu.contacts)
            sort!(bvh_contacts_gpu)

            @test all(bvh_contacts .== bvh_contacts_gpu)
        end
    end
end


@testset "bvh_gpu_$(backend)_bfs_vs_lvt_narrowing" begin

    # Check the narrowing function is applied correctly in both BFS and LVT traversals
    Random.seed!(42)

    num_entities = 100
    bvs = map(BSphere{Float32}, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities])
    bvs_gpu = array_from_host(bvs)
    bvh_gpu = BVH(bvs_gpu, BBox{Float32})
    bfs_traversal = traverse(bvh_gpu, BFSTraversal(); narrow=(bv1, bv2) -> bv1.morton > bv2.morton)
    lvt_traversal = traverse(bvh_gpu, LVTTraversal(); narrow=(bv1, bv2) -> bv1.morton > bv2.morton)

    bfs_contacts = Array(bfs_traversal.contacts) |> sort!
    lvt_contacts = Array(lvt_traversal.contacts) |> sort!
    @test bfs_contacts == lvt_contacts

    # Same but for BVH-BVH pair traversal
    num_entities1 = 121
    num_entities2 = 91
    bvs1 = map(BSphere{Float32}, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities1])
    bvs2 = map(BSphere{Float32}, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities2])
    bvs_gpu1 = array_from_host(bvs1)
    bvs_gpu2 = array_from_host(bvs2)
    bvh_gpu1 = BVH(bvs_gpu1, BBox{Float32})
    bvh_gpu2 = BVH(bvs_gpu2, BBox{Float32})
    bfs_traversal = traverse(
        bvh_gpu1, bvh_gpu2, BFSTraversal();
        narrow=(bv1, bv2) -> bv1.morton > bv2.morton,
    )
    lvt_traversal = traverse(
        bvh_gpu1, bvh_gpu2, LVTTraversal();
        narrow=(bv1, bv2) -> bv1.morton > bv2.morton,
    )

    bfs_contacts = Array(bfs_traversal.contacts) |> sort!
    lvt_contacts = Array(lvt_traversal.contacts) |> sort!
    @test bfs_contacts == lvt_contacts
end
