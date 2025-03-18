# compgrad_Rhapsodie.jl

First install the julia dependencies:
```julia
using Pkg
pkg"registry add General"
pkg"registry add https://github.com/emmt/EmmtRegistry"
pkg"add EasyFITS"
Pkg.add(url="https://github.com/LaurenceDenneulin/RhapsodieDirect.git")
Pkg.add(url="https://github.com/andferrari/compgrad_Rhapsodie.jl.git")
#
# to test first init:
using compgrad_Rhapsodie
D = init_rhapsodie()
#
# and then compute the gradient:
x = randn(128,128,3)
g, chi2 = comp_grad(x, D)
```

To call from python:
```python
from juliacall import Main as jl
import numpy as np
jl.seval("using compgrad_Rhapsodie")
jl.seval("D = init_rhapsodie()")
x = np.random.uniform(low=0,high=1,size=(128,128,3))
g, chi2 = jl.comp_grad(x, jl.D)
# g is by default an Array. Convert to numpy array
g = np.array(g)
```
