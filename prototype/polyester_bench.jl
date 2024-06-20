using Polyester: @batch
using Base.Threads: @threads
using LinearAlgebra
using BenchmarkTools
import AcceleratedKernels as AK

# pinning threads for good measure
using ThreadPinning
pinthreads(:cores)

# Single threaded.
function axpy_serial!(y, a, x)
    for i in eachindex(y,x)
        @inbounds y[i] = a * x[i] + y[i]
    end
end

# Multithreaded with @batch
function axpy_batch!(y, a, x)
    @batch for i in eachindex(y,x)
        @inbounds y[i] = a * x[i] + y[i]
    end
end

# Multithreaded with @threads (default scheduling)
function axpy_atthreads!(y, a, x)
    @threads for i in eachindex(y,x)
        @inbounds y[i] = a * x[i] + y[i]
    end
end

# Multithreaded with @threads :static
function axpy_atthreads_static!(y, a, x)
    @threads :static for i in eachindex(y,x)
        @inbounds y[i] = a * x[i] + y[i]
    end
end

# Multithreaded with AcceleratedKernels.jl
function axpy_ak!(y, a, x)
    AK.foreachindex(y, scheduler=:threads) do i
        @inbounds y[i] = a * x[i] + y[i]
    end
end


y = rand(10_000);
x = rand(10_000);
# @benchmark axpy_serial!($y, eps(), $x)
# @benchmark axpy_batch!($y, eps(), $x)
# @benchmark axpy_ak!($y, eps(), $x)
# @benchmark axpy_atthreads!($y, eps(), $x)
# @benchmark axpy_atthreads_static!($y, eps(), $x)
# @benchmark axpy!(eps(), $x, $y) # BLAS built-in axpy
# VERSION
