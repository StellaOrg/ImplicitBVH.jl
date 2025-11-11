# Merge two bounding spheres
function BSphere{T}(a::BSphere, b::BSphere) where T
    length = dist3(a.x, b.x)

    # a is enclosed within b
    if length + a.r <= b.r
        return BSphere{T}(b.x, b.r)

    # b is enclosed within a
    elseif length + b.r <= a.r
        return BSphere{T}(a.x, a.r)

    # Bounding spheres are not enclosed
    else
        frac = T(0.5) * ((b.r - a.r) / length + T(1))
        centre = (a.x[1] + frac * (b.x[1] - a.x[1]),
                  a.x[2] + frac * (b.x[2] - a.x[2]),
                  a.x[3] + frac * (b.x[3] - a.x[3]))
        radius = T(0.5) * (length + a.r + b.r)
        return BSphere{T}(centre, radius)
    end
end


BSphere(a::BSphere{T}, b::BSphere{T}) where T = BSphere{T}(a, b)
Base.:+(a::BSphere, b::BSphere) = BSphere(a, b)


# Merge two bounding boxes
function BBox{T}(a::BBox, b::BBox) where T
    lower = (minimum2(a.lo[1], b.lo[1]),
             minimum2(a.lo[2], b.lo[2]),
             minimum2(a.lo[3], b.lo[3]))

    upper = (maximum2(a.up[1], b.up[1]),
             maximum2(a.up[2], b.up[2]),
             maximum2(a.up[3], b.up[3]))

    BBox{T}(lower, upper)
end

BBox(a::BBox{T}, b::BBox{T}) where T = BBox{T}(a, b)
Base.:+(a::BBox, b::BBox) = BBox(a, b)


# Convert BSphere to BBox
function BBox{T}(a::BSphere) where T
    lower = (a.x[1] - a.r, a.x[2] - a.r, a.x[3] - a.r)
    upper = (a.x[1] + a.r, a.x[2] + a.r, a.x[3] + a.r)
    BBox{T}(lower, upper)
end

function BBox(a::BSphere{T}) where T
    BBox{T}(a)
end

# Merge two BSphere into enclosing BBox
function BBox{T}(a::BSphere, b::BSphere) where T
    length = dist3(a.x, b.x)

    # a is enclosed within b
    if length + a.r <= b.r
        return BBox{T}(b)

    # b is enclosed within a
    elseif length + b.r <= a.r
        return BBox{T}(a)

    # Bounding spheres are not enclosed
    else
        lower = (minimum2(a.x[1] - a.r, b.x[1] - b.r),
                 minimum2(a.x[2] - a.r, b.x[2] - b.r),
                 minimum2(a.x[3] - a.r, b.x[3] - b.r))

        upper = (maximum2(a.x[1] + a.r, b.x[1] + b.r),
                 maximum2(a.x[2] + a.r, b.x[2] + b.r),
                 maximum2(a.x[3] + a.r, b.x[3] + b.r))

        return BBox{T}(lower, upper)
    end
end

function BBox(a::BSphere{T}, b::BSphere{T}) where T
    BBox{T}(a, b)
end
