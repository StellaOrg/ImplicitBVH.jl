using ImplicitBVH
using Documenter

makedocs(
    modules = [ImplicitBVH],
    sitename = "ImplicitBVH.jl",
    format = Documenter.HTML(
        # Only create web pretty-URLs on the CI
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
)
deploydocs(repo = "github.com/StellaOrg/ImplicitBVH.jl.git")
