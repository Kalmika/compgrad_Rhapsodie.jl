using compgrad_Rhapsodie
D = init_rhapsodie()

function recon(D)
    X = zeros(128,128,3)
    for k in 1:10
        X = X - 0.1*comp_grad(X, D)[1]
    end
    return X
end