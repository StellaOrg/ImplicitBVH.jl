"""
    $(TYPEDEF)

Options for building and traversing bounding volume hierarchies, including parallel strategy
settings.

An exemplar of an index (e.g. `Int32(0)`) is used to deduce the types of indices used in the BVH
building (`ImplicitTree`(@ref), order) and traversal (`IndexPair`(@ref)).

The CPU scheduler can be `:threads` (for base Julia threads) or `:polyester` (for Polyester.jl
threads).

If `compute_extrema=false` and `mins` / `maxs` are defined, they will not be computed from the
distribution of bounding volumes; useful if you have a fixed simulation box, for example.

# Methods
    BVHOptions(;

        # Example index from which to deduce type
        index_exemplar::I               = Int32(0),

        # CPU threading
        scheduler::Symbol               = :threads,
        num_threads::Int                = Threads.nthreads(),
        min_mortons_per_thread::Int     = 1000,
        min_boundings_per_thread::Int   = 1000,
        min_traversals_per_thread::Int  = 1000,

        # GPU scheduling
        block_size::Int                 = 256,

        # Minima / maxima
        compute_extrema::Bool           = true,
        mins::NTuple{3, T}              = (NaN32, NaN32, NaN32),
        maxs::NTuple{3, T}              = (NaN32, NaN32, NaN32),
    ) where {I <: Integer, T}

# Fields
    $(TYPEDFIELDS)

"""
struct BVHOptions{I <: Integer, T}

    # Example index from which to deduce type
    index_exemplar::I

    # CPU threading
    scheduler::Symbol
    num_threads::Int
    min_mortons_per_thread::Int
    min_boundings_per_thread::Int
    min_traversals_per_thread::Int

    # GPU scheduling
    block_size::Int

    # Minima / maxima
    compute_extrema::Bool
    mins::NTuple{3, T}
    maxs::NTuple{3, T}
end


function BVHOptions(;

    # Example index from which to deduce type
    index_exemplar::I               = Int32(0),

    # CPU threading
    scheduler::Symbol               = :threads,
    num_threads::Int                = Threads.nthreads(),
    min_mortons_per_thread::Int     = 1000,
    min_boundings_per_thread::Int   = 1000,
    min_traversals_per_thread::Int  = 1000,

    # GPU scheduling
    block_size::Int                 = 256,

    # Minima / maxima
    compute_extrema::Bool           = true,
    mins::NTuple{3, T}              = (NaN32, NaN32, NaN32),
    maxs::NTuple{3, T}              = (NaN32, NaN32, NaN32),
) where {I <: Integer, T}

    # Correctness checks
    @argcheck num_threads > 0
    @argcheck min_mortons_per_thread > 0
    @argcheck min_boundings_per_thread > 0
    @argcheck min_traversals_per_thread > 0
    @argcheck block_size > 0

    # If we want to avoid computing extrema, make sure `mins` and `maxs` were defined
    if !compute_extrema
        @argcheck all(isfinite, mins)
        @argcheck all(isfinite, maxs)
    end

    BVHOptions(
        index_exemplar,
        scheduler,
        num_threads,
        min_mortons_per_thread,
        min_boundings_per_thread,
        min_traversals_per_thread,
        block_size,
        compute_extrema,
        mins,
        maxs,
    )
end




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

@generated function msbindex(::Type{T}) where {T <: Integer}
    sizeof(T) * 8 - 1
end

ilog2(x, ::typeof(RoundUp)) = ispow2(x) ? ilog2(x) : ilog2(x) + 1
ilog2(x, ::typeof(RoundDown)) = ilog2(x)

@inline function ilog2(n::T) where {T <: IntBits}
    @boundscheck n > zero(T) || throw(DomainError(n))
    msbindex(T) - leading_zeros(n)
end

unsafe_ilog2(x, ::typeof(RoundUp)) = ispow2(x) ? unsafe_ilog2(x) : unsafe_ilog2(x) + 1
unsafe_ilog2(x, ::typeof(RoundDown)) = unsafe_ilog2(x)

@inline function unsafe_ilog2(n::T) where {T <: IntBits}
    msbindex(T) - leading_zeros(n)
end




# Specialised maths functions
pow2(n::I) where I <: Integer = one(I) << n


function dot3(x, y)
    x[1] * y[1] + x[2] * y[2] + x[3] * y[3]
end


function dist3sq(x, y)
    (x[1] - y[1]) * (x[1] - y[1]) +
    (x[2] - y[2]) * (x[2] - y[2]) +
    (x[3] - y[3]) * (x[3] - y[3])
end


dist3(x, y) = sqrt(dist3sq(x, y))

minimum2(a, b) = a < b ? a : b
minimum3(a, b, c) = a < b ? minimum2(a, c) : minimum2(b, c)

maximum2(a, b) = a > b ? a : b
maximum3(a, b, c) = a > b ? maximum2(a, c) : maximum2(b, c)
