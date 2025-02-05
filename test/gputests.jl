# The CPU tests have "first-principles" correctness tests, against known correct results. If these
# work, then it is sufficient for the GPU implementations to be checked against the CPU results.
#
# Called from runtests.jl which sets the following, e.g. for CUDA:
# Pkg.add("CUDA")
# using CUDA
# const backend = CUDABackend()
println("Running GPU tests on backend $backend")
using KernelAbstractions


# Defining here to work with KA.zeros
Base.zero(::Type{BSphere{T}}) where T = BSphere{T}((T(0), T(0), T(0)), T(0))
Base.zero(::Type{BBox{T}}) where T = BBox{T}((T(0), T(0), T(0)), (T(0), T(0), T(0)))


function array_from_host(h_arr::AbstractArray, dtype=nothing)
    d_arr = KernelAbstractions.zeros(backend, isnothing(dtype) ? eltype(h_arr) : dtype, size(h_arr))
    copyto!(d_arr, h_arr isa Array ? h_arr : Array(h_arr))      # Allow unmaterialised types, e.g. ranges
    d_arr
end


@testset "mortons_gpu_$(backend)" begin
    # This tests BV bounds computation too
    Random.seed!(42)
    for num_entities in 1:200
        bvs = map(BSphere{Float32}, [6 * rand(3) .+ rand(3, 3) for _ in 1:num_entities])
        bvs_gpu = array_from_host(bvs)

        mortons = ImplicitBVH.morton_encode(bvs)
        mortons_gpu = ImplicitBVH.morton_encode(bvs_gpu)

        @test all(mortons .== Array(mortons_gpu))
    end
end


@testset "bvh_gpu_$(backend)_single_randomised" begin
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

            traversal = traverse(bvh, start_level)
            traversal_gpu = traverse(bvh_gpu, start_level)

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

            traversal = traverse(bvh, start_level)
            traversal_gpu = traverse(bvh_gpu, start_level)

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
    traversal = traverse(bvh)

    BVH(bvs, BSphere{Float32})
    BVH(bvs, BBox{Float32})
    BVH(bvs, BBox{Float32}, UInt32)
    BVH(bvs, BBox{Float32}, UInt32, 3)
    BVH(bvs, BBox{Float32}, UInt32, 0.0)
    BVH(bvs, BBox{Float32}, UInt32, 0.5)
    BVH(bvs, BBox{Float32}, UInt32, 1.0)

    traverse(bvh, 3)
    traverse(bvh, 3, traversal)
end




@testset "bvh_gpu_$(backend)_pair_randomised" begin
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

            traversal = traverse(bvh1, bvh2, start_level1, start_level2)
            traversal_gpu = traverse(bvh_gpu1, bvh_gpu2, start_level1, start_level2)

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

            traversal = traverse(bvh1, bvh2, start_level)
            traversal_gpu = traverse(bvh_gpu1, bvh_gpu2, start_level)

            # Compare sorted contact pair indices on the host
            bvh_contacts = traversal.contacts
            sort!(bvh_contacts)

            bvh_contacts_gpu = Array(traversal_gpu.contacts)
            sort!(bvh_contacts_gpu)

            @test all(bvh_contacts .== bvh_contacts_gpu)
        end
    end
end

