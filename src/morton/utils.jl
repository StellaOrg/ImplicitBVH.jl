function _compute_extrema(bounding_volumes::AbstractVector, options)

    function min_centers(a, b)
        # a and b are NTuple{3, Float}
        (
            a[1] < b[1] ? a[1] : b[1],
            a[2] < b[2] ? a[2] : b[2],
            a[3] < b[3] ? a[3] : b[3],
        )
    end

    function max_centers(a, b)
        # a and b are NTuple{3, Float}
        (
            a[1] > b[1] ? a[1] : b[1],
            a[2] > b[2] ? a[2] : b[2],
            a[3] > b[3] ? a[3] : b[3],
        )
    end

    # Get numeric type of the inner bounding volume
    T = eltype(eltype(bounding_volumes))

    xyzmin = AK.mapreduce(
        bv -> center(bv.volume),    # Take the centre of each bounding volume
        min_centers,                # Reduce to the 3D minimum
        bounding_volumes,
        init=(floatmax(T), floatmax(T), floatmax(T)),
        neutral=(floatmax(T), floatmax(T), floatmax(T)),
        max_tasks=options.num_threads,
        min_elems=options.min_mortons_per_thread,
        block_size=options.block_size,
    )

    xyzmax = AK.mapreduce(
        bv -> center(bv.volume),
        max_centers,
        bounding_volumes,
        init=(floatmin(T), floatmin(T), floatmin(T)),
        neutral=(floatmin(T), floatmin(T), floatmin(T)),
        max_tasks=options.num_threads,
        min_elems=options.min_mortons_per_thread,
        block_size=options.block_size,
    )

    xyzmin, xyzmax
end


"""
    bounding_volumes_extrema(bounding_volumes)

Compute exclusive lower and upper bounds in iterable of bounding volumes, e.g. Vector{BBox}.
"""
function bounding_volumes_extrema(bounding_volumes::AbstractVector, options=BVHOptions())

    # Compute exact extrema
    (xmin, ymin, zmin), (xmax, ymax, zmax) = _compute_extrema(bounding_volumes, options)

    # Expand extrema by float precision to ensure morton codes are exclusively bounded by them
    T = typeof(xmin)

    xmin = xmin - relative_precision(T) * abs(xmin) - floatmin(T)
    ymin = ymin - relative_precision(T) * abs(ymin) - floatmin(T)
    zmin = zmin - relative_precision(T) * abs(zmin) - floatmin(T)

    xmax = xmax + relative_precision(T) * abs(xmax) + floatmin(T)
    ymax = ymax + relative_precision(T) * abs(ymax) + floatmin(T)
    zmax = zmax + relative_precision(T) * abs(zmax) + floatmin(T)

    (xmin, ymin, zmin), (xmax, ymax, zmax)
end
