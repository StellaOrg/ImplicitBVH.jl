# Running Benchmarks / Examples

First, download the [`xyzrgb_dragon.obj`](https://github.com/alecjacobson/common-3d-test-models/blob/master/data/xyzrgb_dragon.obj) into this directory.

You can run the following benchmarks, which then generate [PProf](https://github.com/JuliaPerf/PProf.jl) profiling archives:

```bash
bvh_build.jl        => bvh_build.pb.gz
bvh_contact.jl      => bvh_contact.pb.gz
bvh_volumes.jl      => bvh_volumes.pb.gz
morton.jl           => morton.pb.gz
```

You can open each profile using the `view_profile.jl` script.
Finally, you can plot the bounding boxes around a given mesh via [GLMakie](https://docs.makie.org/stable/) using the `plotting.jl` script.
