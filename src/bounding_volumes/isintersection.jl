@inline function isintersection(
    b::BBox{T},
    p::Union{AbstractVector{T}, NTuple{3, T}},
    d::Union{AbstractVector{T}, NTuple{3, T}},
) where T

    inv_d = (one(T) / d[1], one(T) / d[2], one(T) / d[3])

    # Set x bounds
    t_bound_x1 = (b.lo[1] - p[1]) * inv_d[1]
    t_bound_x2 = (b.up[1] - p[1]) * inv_d[1]

    tmin = minimum2(t_bound_x1, t_bound_x2)
    tmax = maximum2(t_bound_x1, t_bound_x2)

    # Set y bounds
    t_bound_y1 = (b.lo[2] - p[2]) * inv_d[2]
    t_bound_y2 = (b.up[2] - p[2]) * inv_d[2]

    tmin = maximum2(tmin, minimum2(t_bound_y1, t_bound_y2))
    tmax = minimum2(tmax, maximum2(t_bound_y1, t_bound_y2))

    # Set z bounds
    t_bound_z1 = (b.lo[3] - p[3]) * inv_d[3]
    t_bound_z2 = (b.up[3] - p[3]) * inv_d[3]

    tmin = maximum2(tmin, minimum2(t_bound_z1, t_bound_z2))
    tmax = minimum2(tmax, maximum2(t_bound_z1, t_bound_z2))
        
    # If condition satisfied ray intersects box. tmax >= 0 
    # ensure only forwards intersections are counted
    (tmin <= tmax) && (tmax >= 0)
end


@inline function isintersection(
    s::BSphere{T},
    p::Union{AbstractVector{T}, NTuple{3, T}},
    d::Union{AbstractVector{T}, NTuple{3, T}},
) where T

    a = dot3(d, d)
    b = T(2) * (
        (p[1] - s.x[1]) * d[1] +
        (p[2] - s.x[2]) * d[2] +
        (p[3] - s.x[3]) * d[3]
    )
    c = (
        (p[1] - s.x[1]) * (p[1] - s.x[1]) +
        (p[2] - s.x[2]) * (p[2] - s.x[2]) +
        (p[3] - s.x[3]) * (p[3] - s.x[3])
    ) - s.r * s.r

    discriminant = b * b - T(4) * a * c

    if discriminant >= T(0)
        if b <= T(0)
            return true
        else
            return T(0) >= c
        end
    else
        return false
    end
end
