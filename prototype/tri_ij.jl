using ImplicitBVH
using ImplicitBVH: BBox, BSphere
import AcceleratedKernels as AK

using BenchmarkTools

using Profile
using PProf

# using Metal
# using AtomixMetal

using Random
Random.seed!(0)


# Get i_lin-th (i, j) for level_nodes if we have num_real nodes at the given level
# First add n_lin node-node pair checks, then another n self-checks
# i_lin between 1 and n_lin + n
@inline function _initial_level_pair(level_nodes::Integer, num_real::Integer, i_lin::Integer)
    n_lin = num_real * (num_real - 0x1) ÷ 0x2
    if i_lin > n_lin
        i = i_lin - n_lin - 0x1 + level_nodes
        return (i, i)
    else
        i, j = _tri_ij(num_real, i_lin - 0x1)
        i += level_nodes
        j += level_nodes
        return (i, j)
    end
end


# """
# Get the k-th (i,j) pair with 0 <= i < j < n in the lexicographic order:
# 
#     (0,1), (0,2), ..., (0,n-1),
#     (1,2), (1,3), ..., (1,n-1),
#     ...
#     (n-2,n-1)
# 
# Condition: 0 <= k < n*(n-1)÷2
# """
# @inline function _k2ij_exclusive(n::I, k::I) where I <: Integer
#     a = Float32(-I(8) * k + I(4) * n * (n - I(1)) - I(7))
#     b = unsafe_trunc(I, sqrt(a) / 2.0f0 - 0.5f0)
#     i = n - I(2) - b
#     j = k + i + I(1) - n * (n - I(1)) ÷ I(2) + (n - i) * ((n - i) - I(1)) ÷ I(2)
#     (i, j)
# end




# """
# Get the k-th (i,j) pair with 0 <= i <= j < n in the lexicographic order:
# 
#     (0,0), (0,1), ..., (0,n-1),
#     (1,1), (1,2), ..., (1,n-1),
#     ...
#     (n-1,n-1)
# 
# Condition: 0 <= k < n*(n+1)÷2
# """
# @inline function _k2ij_inclusive(n::I, k::I) where I <: Integer
#     a = Float32(I(4) * n * n + I(4) * n - I(8) * k + I(1))
#     i = unsafe_trunc(I, 0.5f0 * (I(2) * n + I(1) - sqrt(a)))
#     j = unsafe_trunc(I, k - (i * n - 0.5f0 * i * i - 0.5f0 * i))
#     (i, j)
# end






function test1_exclusive(level_nodes, num_real)
    @assert level_nodes >= num_real
    kmax = num_real * (num_real - 0x1) ÷ 0x2
    v = Vector{Tuple{Int, Int}}(undef, kmax)
    for k in 1:kmax
        i, j = _k2ij_exclusive(num_real, k - 1)
        v[k] = (i + level_nodes, j + level_nodes)
    end
    return v
end


function test1_inclusive(level_nodes, num_real)
    @assert level_nodes >= num_real
    kmax = num_real * (num_real + 0x1) ÷ 0x2
    v = Vector{Tuple{Int, Int}}(undef, kmax)
    for k in 1:kmax
        i, j = _k2ij_inclusive(num_real, k - 1)
        v[k] = (i + level_nodes, j + level_nodes)
    end
    return v
end


# Testing the functions
function test_inclusive(n)
    kmax = n * (n + 0x1) ÷ 0x2
    v = Vector{Tuple{Int, Int}}(undef, kmax)
    for k in 0x1:kmax
        v[k] = _k2ij_inclusive(n, k - 0x1)
    end

    # Test that pairs are generated in the correct order
    iv = 1
    for i in 0x0:n - 0x1
        for j in i:n - 0x1
            if v[iv] != (i, j)
                @error "Expected $((i, j)) but got $(v[iv]) at index $iv for n = $n"
            end
            @assert v[iv] == (i, j)
            iv += 1
        end
    end
end


function test_exclusive(n)
    kmax = n * (n - 0x1) ÷ 0x2
    v = Vector{Tuple{Int, Int}}(undef, kmax)
    for k in 0x1:kmax
        v[k] = _k2ij_exclusive(n, k - 0x1)
    end

    # Test that pairs are generated in the correct order
    iv = 1
    for i in 0x0:n - 0x2
        for j in i + 0x1:n - 0x1
            if v[iv] != (i, j)
                @error "Expected $((i, j)) but got $(v[iv]) at index $iv for n = $n"
            end
            @assert v[iv] == (i, j)
            iv += 1
        end
    end
end


# Run tests
for n in 1:100:10000
    test_inclusive(n)
    test_exclusive(n)
end

for n in Int32(1):Int32(100):Int32(10000)
    test_inclusive(n)
    test_exclusive(n)
end

display(@benchmark test1_inclusive(512, 500))
display(@benchmark test1_exclusive(512, 500))
