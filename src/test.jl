using compgrad_Rhapsodie
using HDF5

D = init_rhapsodie()

path = replace(pathof(compgrad_Rhapsodie), "src/compgrad_Rhapsodie.jl" => "data")
path_disk = path*"/sample_for_rhapsodie_128x128.h5"
dset = h5open(path_disk, "r")
θ = read(dset["disk_theta"])
close(dset)

function recon(D, μ, niter, θ)
    X = zeros(128,128,3)
    #X[:,:,3] = θ

    for k in 1:niter
        X = X - μ*comp_grad(X, D)[1]
    end
    return X
end