# Bounding sphere
struct BSphere{T <: Real}
    x::SVector{3, T}
    r::T
end


# Adapted from https://realtimecollisiondetection.net/blog/?p=20
# TODO: for Float16 we currently get NaNs
function BSphere{T}(p1, p2, p3) where {T <: Real}

    # Convert to SVectors of the requested type; code will be unrolled and heavily vectorized
    a = SVector{3, T}(p1)
    b = SVector{3, T}(p2)
    c = SVector{3, T}(p3)

    abab = T(dot(b - a, b - a))
    abac = T(dot(b - a, c - a))
    acac = T(dot(c - a, c - a))

    d = T(2.) * (abab * acac - abac * abac)

    # Declaring bounding sphere centre and radius
    centre = SVector{3, T}(a)
    radius = zero(T)

    if d^2 <= T(1e-5) * sum(dot(centre, centre))
        # a, b, c lie on a line. Find line centre and radius
        lower = MVector{3}(a)
        upper = MVector{3}(a)

        # This will all be unrolled and vectorized
        for current in (b, c)
            for i in 1:3
                if current[i] < lower[i]
                    lower[i] = current[i]
                end

                if current[i] > upper[i]
                    upper[i] = current[i]
                end
            end
        end

        centre = SVector{3}(T(0.5) * (lower + upper))
        radius = norm(upper - centre)
    else
        s = (abab * acac - acac * abac) / d
        t = (acac * abab - abab * abac) / d

        if s <= zero(T)
            centre = T(0.5) * (a + c)
            radius = norm(centre - a)
        elseif t <= zero(T)
            centre = T(0.5) * (a + b)
            radius = norm(centre - a)
        elseif s + t >= one(T)
            centre = T(0.5) * (b + c)
            radius = norm(centre - b)
        else
            centre = a + s * (b - a) + t * (c - a)
            radius = norm(centre - a)
        end
    end

    BSphere(SVector{3, T}(centre), T(radius))
end


# Convenience constructors
function BSphere(p1, p2, p3)
    BSphere{eltype(p1)}(p1, p2, p3)
end

function BSphere{T}(triangle) where {T <: Real}
    # Decompose triangle into its 3 vertices.
    # Works transparently with GeometryBasics.Triangle, Vector{SVector{3, T}}, etc.
    p1, p2, p3 = triangle
    BSphere{T}(p1, p2, p3)
end

function BSphere(triangle)
    p1, p2, p3 = triangle
    BSphere{eltype(p1)}(p1, p2, p3)
end

function BSphere{T}(vertices::AbstractMatrix) where {T <: Real}
    BSphere{T}(@view(vertices[:, 1]), @view(vertices[:, 2]), @view(vertices[:, 3]))
end

function BSphere(vertices::AbstractMatrix)
    BSphere{eltype(vertices)}(@view(vertices[:, 1]), @view(vertices[:, 2]), @view(vertices[:, 3]))
end


# Ensure consistent interfaces
Base.eltype(::BSphere{T}) where {T <: Real} = T
Base.eltype(::Type{BSphere{T}}) where {T <: Real} = T

center(b::BSphere) = b.x
radius(b::BSphere) = b.r
lower(b::BSphere) = b.x .- b.r
upper(b::BSphere) = b.x .+ b.r


# Merge two bounding spheres
function Base.:+(a::BSphere, b::BSphere)
    relative = b.x - a.x
    length = norm(relative)
    direction = relative / length

    # a is enclosed within b
    if length + a.r <= b.r
        centre = b.x
        radius = b.r

    # b is enclosed within a
    elseif length + b.r <= a.r
        centre = a.x
        radius = a.r

    # Bounding spheres are not enclosed
    else
        lower = a.x - a.r * direction
        upper = b.x + b.r * direction

        centre = lower + (upper - lower) / 2
        radius = (a.r + length + b.r) / 2
    end

    BSphere(centre, radius)
end


# Contact detection
function iscontact(a::BSphere, b::BSphere)
    (b.x[1] - a.x[1]) * (b.x[1] - a.x[1]) +
    (b.x[2] - a.x[2]) * (b.x[2] - a.x[2]) +
    (b.x[3] - a.x[3]) * (b.x[3] - a.x[3]) <=
    (a.r + b.r) * (a.r + b.r)
end




# Bounding box
struct BBox{T <: Real}
    lo::SVector{3, T}
    up::SVector{3, T}
end


function BBox{T}(p1, p2, p3) where {T <: Real}

    # Convert to SVectors of the requested type; code will be unrolled and heavily vectorized
    lower = MVector{3, T}(p1)
    upper = MVector{3, T}(p1)

    b = SVector{3, T}(p2)
    c = SVector{3, T}(p3)

    # This will all be unrolled and vectorized
    for current in (b, c)
        for i in 1:3
            if current[i] < lower[i]
                lower[i] = current[i]
            end

            if current[i] > upper[i]
                upper[i] = current[i]
            end
        end
    end
   
    BBox{T}(SVector{3, T}(lower), SVector{3, T}(upper))
end


# Convenience constructors
function BBox(p1, p2, p3)
    BBox{eltype(p1)}(p1, p2, p3)
end

function BBox{T}(triangle) where {T <: Real}
    # Decompose triangle into its 3 vertices.
    # Works transparently with GeometryBasics.Triangle, Vector{SVector{3, T}}, etc.
    p1, p2, p3 = triangle
    BBox{T}(p1, p2, p3)
end

function BBox(triangle)
    p1, p2, p3 = triangle
    BBox{eltype(p1)}(p1, p2, p3)
end

function BBox{T}(vertices::AbstractMatrix) where {T <: Real}
    BBox{T}(@view(vertices[:, 1]), @view(vertices[:, 2]), @view(vertices[:, 3]))
end

function BBox(vertices::AbstractMatrix)
    BBox{eltype(vertices)}(@view(vertices[:, 1]), @view(vertices[:, 2]), @view(vertices[:, 3]))
end


# Construct BBox from BSphere
function BBox{T}(a::BSphere{T}) where {T <: Real}
    BBox(a.x .- a.r, a.x .+ a.r)
end

function BBox(a::BSphere{T}) where {T <: Real}
    BBox{T}(a)
end

function BBox{T}(a::BSphere{T}, b::BSphere{T}) where {T <: Real}
    relative = b.x - a.x
    length = norm(relative)
    direction = relative / length

    # a is enclosed within b
    if length + a.r <= b.r
        lo = b.x .- b.r
        up = b.x .+ b.r

    # b is enclosed within a
    elseif length + b.r <= a.r
        lo = a.x .- a.r
        up = a.x .+ a.r

    # Bounding spheres are not enclosed
    else
        lo = SVector{3, T}(
            min(a.x[1] - a.r, b.x[1] - b.r),
            min(a.x[2] - a.r, b.x[2] - b.r),
            min(a.x[3] - a.r, b.x[3] - b.r),
        )

        up = SVector{3, T}(
            max(a.x[1] + a.r, b.x[1] + b.r),
            max(a.x[2] + a.r, b.x[2] + b.r),
            max(a.x[3] + a.r, b.x[3] + b.r),
        )
    end

    BBox(lo, up)
end

function BBox(a::BSphere{T}, b::BSphere{T}) where {T <: Real}
    BBox{T}(a, b)
end


# Ensure consistent interfaces
Base.eltype(::BBox{T}) where {T <: Real} = T
Base.eltype(::Type{BBox{T}}) where {T <: Real} = T

center(b::BBox) = (b.lo + b.up) / 2
radius(b::BBox) = (b.up - b.lo) / 2
lower(b::BBox) = b.lo
upper(b::BBox) = b.up


# Merge two bounding boxes
function Base.:+(a::BBox, b::BBox)

    lower = MVector{3}(a.lo)
    upper = MVector{3}(b.up)

    # This will be unrolled and vectorized
    for current in (a, b)
        for i in 1:3
            if current.lo[i] < lower[i]
                lower[i] = current.lo[i]
            end

            if current.up[i] > upper[i]
                upper[i] = current.up[i]
            end
        end
    end

    BBox(SVector{3}(lower), SVector{3}(upper))
end


# Contact detection
function iscontact(a::BBox, b::BBox)
    (a.up[1] >= b.lo[1] && a.lo[1] <= b.up[1]) &&
    (a.up[2] >= b.lo[2] && a.lo[2] <= b.up[2]) &&
    (a.up[3] >= b.lo[3] && a.lo[3] <= b.up[3])
end
