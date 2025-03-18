# compgrad_Rhapsodie.jl

To install:
```julia
using Pkg
Pkg.add(url="https://github.com/LaurenceDenneulin/RhapsodieDirect.git")
Pkg.add(url="https://github.com/andferrari/compgrad_Rhapsodie.jl.git")
# init
using compgrad_Rhapsodie
path = replace(pathof(compgrad_Rhapsodie), "src/compgrad_Rhapsodie.jl" => "data")
D = init_rhapsodie(path)
# to compute the gradient
x = randn(128,128,3)
g, chi2 = comp_grad(x, D)
```
