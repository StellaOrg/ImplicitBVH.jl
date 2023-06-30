using IBVH
using Documenter

makedocs(
    modules = [IBVH],
    sitename = "IBVH.jl",
    format = Documenter.HTML(
        # Only create web pretty-URLs on the CI
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
)
deploydocs(repo = "github.com/StellaOrg/IBVH.jl.git")
