"""
    $(TYPEDEF)

Partitioning `num_elems` elements / jobs over maximum `max_tasks` tasks with minimum `min_elems`
elements per task.

# Methods
    TaskPartitioner(num_elems, max_tasks=Threads.nthreads(), min_elems=1)

# Fields
    $(TYPEDFIELDS)

# Examples

```jldoctest
using ImplicitBVH: TaskPartitioner

# Divide 10 elements between 4 tasks
tp = TaskPartitioner(10, 4)
for i in 1:tp.num_tasks
    @show tp[i]
end

# output
tp[i] = (1, 3)
tp[i] = (4, 6)
tp[i] = (7, 9)
tp[i] = (10, 10)
```

```jldoctest
using ImplicitBVH: TaskPartitioner

# Divide 20 elements between 6 tasks with minimum 5 elements per task.
# Not all tasks will be required
tp = TaskPartitioner(20, 6, 5)
for i in 1:tp.num_tasks
    @show tp[i]
end

# output
tp[i] = (1, 5)
tp[i] = (6, 10)
tp[i] = (11, 15)
tp[i] = (16, 20)
```

"""
struct TaskPartitioner
    num_elems::Int
    max_tasks::Int
    min_elems::Int
    num_tasks::Int      # computed
end


function TaskPartitioner(num_elems, max_tasks=Threads.nthreads(), min_elems=1)
    # Number of tasks needed to have at least `min_nodes` per task
    num_tasks = num_elems ÷ max_tasks >= min_elems ? max_tasks : num_elems ÷ min_elems
    if num_tasks < 1
        num_tasks = 1
    end

    TaskPartitioner(num_elems, max_tasks, min_elems, num_tasks)
end


function Base.getindex(tp::TaskPartitioner, itask::Integer)

    @boundscheck 1 <= itask <= tp.num_tasks || throw(BoundsError(tp, itask))

    # Compute element indices handled by this task
    per_task = (tp.num_elems + tp.num_tasks - 1) ÷ tp.num_tasks

    task_istart = (itask - 1) * per_task + 1
    task_istop = min(itask * per_task, tp.num_elems)

    task_istart, task_istop
end


Base.firstindex(tp::TaskPartitioner) = 1
Base.lastindex(tp::TaskPartitioner) = tp.num_tasks
Base.length(tp::TaskPartitioner) = tp.num_tasks




# Fast ilog2 adapted from https://github.com/jlapeyre/ILog2.jl - thank you!
# Included here directly to minimise dependencies and possible errors surface area
const IntBits = Union{Int8, Int16, Int32, Int64, Int128,
                      UInt8, UInt16, UInt32, UInt64, UInt128}

ilog2(x, ::typeof(RoundUp)) = ispow2(x) ? ilog2(x) : ilog2(x) + 1
ilog2(x, ::typeof(RoundDown)) = ilog2(x)

@generated function msbindex(::Type{T}) where {T <: Integer}
    sizeof(T) * 8 - 1
end

@inline function ilog2(n::T) where {T <: IntBits}
    @boundscheck n > zero(T) || throw(DomainError(n))
    msbindex(T) - leading_zeros(n)
end




# Specialised maths functions
function dot3(x, y)
    x[1] * y[1] + x[2] * y[2] + x[3] * y[3]
end


function dist23(x, y)
    (x[1] - y[1]) * (x[1] - y[1]) +
    (x[2] - y[2]) * (x[2] - y[2]) +
    (x[3] - y[3]) * (x[3] - y[3])
end


dist3(x, y) = sqrt(dist23(x, y))

minimum2(a, b) = a < b ? a : b
minimum3(a, b, c) = a < b ? minimum2(a, c) : minimum2(b, c)

maximum2(a, b) = a > b ? a : b
maximum3(a, b, c) = a > b ? maximum2(a, c) : maximum2(b, c)
