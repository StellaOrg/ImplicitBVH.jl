# Contact detection
function iscontact(a::BSphere, b::BSphere)
    dist3sq(a.x, b.x) <= (a.r + b.r) * (a.r + b.r)
end


function iscontact(a::BBox, b::BBox)
    (a.up[1] >= b.lo[1] && a.lo[1] <= b.up[1]) &&
    (a.up[2] >= b.lo[2] && a.lo[2] <= b.up[2]) &&
    (a.up[3] >= b.lo[3] && a.lo[3] <= b.up[3])
end


# Contact detection between heterogeneous BVs - only needed when one BVH has exactly one leaf
function iscontact(a::BSphere, b::BBox)
    # This is an edge case, used for broad-phase collision detection, so we simply take the
    # sphere's bounding box, as a full sphere-box contact detection is computationally heavy
    ab = BBox(
        (a.x[1] - a.r, a.x[2] - a.r, a.x[3] - a.r),
        (a.x[1] + a.r, a.x[2] + a.r, a.x[3] + a.r),
    )
    iscontact(ab, b)
end


function iscontact(a::BBox, b::BSphere)
    iscontact(b, a)
end
