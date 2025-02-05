using KernelAbstractions
using Atomix: @atomic
using Metal


@kernel cpu=false inbounds=true function _traverse_nodes_gpu!(
    dst, level,
)
    @atomic dst[level] += 1
    
    ithread = @index(Local, Linear)

    if typeof(ithread) === Int32
        dst[1] = 1
    elseif typeof(ithread) === Int64
        dst[1] = 2
    elseif typeof(ithread) === UInt32
        dst[1] = 3
    else
        dst[1] = sizeof(ithread)
    end

    @synchronize()
end


dst = MtlArray{UInt32}([0,0,0])
level = 2

kernel = _traverse_nodes_gpu!(get_backend(dst), 128)
kernel(dst, level, ndrange=length(dst))
KernelAbstractions.synchronize(get_backend(dst))


