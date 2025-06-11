module compgrad_Rhapsodie

using RhapsodieDirect
using DelimitedFiles
using EasyFITS
using InterpolationKernels
using HDF5
using LinearAlgebra

# Remove warnings
ENV["JULIA_WARN_OVERWRITE"] = "0"
import Logging
Logging.disable_logging(Logging.Warn)

export comp_grad, comp_grad2, init_rhapsodie, generate_star, init_rhapsodie2, comp_residual, comp_grad_x, comp_grad_f, comp_joint_evaluation

function init_rhapsodie2(;alpha = 1e-2, write_files=false, data_folder = "default")
    
    (data_folder ==  "default") && (data_folder = replace(pathof(compgrad_Rhapsodie), "src/compgrad_Rhapsodie.jl" => "data"))
    path_disk = data_folder*"/sample_for_rhapsodie_128x128.h5"
    path_star = data_folder*"/star_cropped.fits"
    ker = CatmullRomSpline(Float64, Flat)

    # load disk
    
    object_params=ObjectParameters((128,128),(64.,64.))  # check with file!

    dset = h5open(path_disk, "r")
    I = read(dset["disk_data"])[:,:,1]
    Ip = read(dset["disk_data"])[:,:,2]
    θ = read(dset["disk_theta"])
    close(dset)

    STAR = readfits(path_star)
    
    coef = alpha*maximum(STAR)/maximum(I)

    S = PolarimetricMap("intensities", coef*(I - Ip) + STAR, coef*Ip, θ) 

    # create model
    
    data_params=DatasetParameters((128,256), 4, 1, 1, (64.,64.))
    indices=get_indices_table(data_params)
    polar_params=set_default_polarisation_coefficients(indices)
    ker=CatmullRomSpline(Float64, Flat)
    field_params=FieldTransformParameters[]

    for i=1:data_params.frames_total
        push!(field_params, FieldTransformParameters(ker,
                                                    0.,
                                                    (0.,0.),
                                                    (-10.7365 , 1.39344),
                                                    polar_params[i][1],
                                                    polar_params[i][2]))
    end

    field_transforms=load_field_transforms(object_params,
                                           data_params,
                                           field_params)
   
    psf_center=readdlm(data_folder*"/PSF_centers_Airy.txt");
    psf=readfits(data_folder*"/PSF_parametered_Airy.fits");
    blur=set_fft_operator(object_params,(psf[1:end÷2,:]'), psf_center[1:2])[1];

    H = DirectModel(size(S), (128,256,4),S.parameter_type,field_transforms,blur)
    BadPixMap=Float64.(rand(0.0:1e-16:1.0,data_params.size).< 0.9);
    # BadPixMap=ones(data_params.size)

    # compute measurements

    data, weight = data_simulator(BadPixMap, field_transforms, S; A=blur);
    S_convolved = PolarimetricMap("stokes", cat(blur*S.I, blur*S.Q, blur*S.U, dims=3)) 

    if write_files == true
        if prod(readdir() .!= "test_results")     
            mkdir("test_results")
        end

        writefits("test_results/DATA_$(data_params.size[1]).fits",
            ["TYPE" => "data"],
            mapslices(transpose,data,dims=[1,2]), overwrite=true)

        writefits("test_results/WEIGHT_$(data_params.size[1]).fits", 
            ["TYPE" => "weights"],
            mapslices(transpose,weight,dims=[1,2]), overwrite=true)

        write(S, "test_results/TRUE_$(data_params.size[1]).fits")

        write(S_convolved, "test_results/TRUE_convolved_$(data_params.size[1]).fits")
    end
    
    # DataSet for gradient computation
    D = Dataset(data, weight , H)

    Ip = zeros(object_params.size);
    θ = zeros(object_params.size);
    STAR_p = PolarimetricMap("intensities", STAR, Ip, θ)

    nAS = sum(D.weights .* abs2.(D.direct_model*STAR_p))

    return D, STAR_p.I, nAS
end


function init_rhapsodie(;alpha = 1.0, write_files=false, path_disk = "default")
    
    path = replace(pathof(compgrad_Rhapsodie), "src/compgrad_Rhapsodie.jl" => "data")
    (path_disk ==  "default") && (path_disk = path*"/sample_for_rhapsodie_128x128.h5")
    ker = CatmullRomSpline(Float64, Flat)

    # load disk
    
    object_params=ObjectParameters((128,128),(64.,64.))  # check with file!

    dset = h5open(path_disk, "r")
    I = read(dset["disk_data"])[:,:,1]
    Ip = read(dset["disk_data"])[:,:,2]
    θ = read(dset["disk_theta"])
    close(dset)

    STAR = generate_star(object_params, alpha)
    
    S = PolarimetricMap("intensities", I - Ip, Ip, θ) + STAR


    # create model
    
    data_params=DatasetParameters((128,256), 4, 1, 1, (64.,64.))
    indices=get_indices_table(data_params)
    polar_params=set_default_polarisation_coefficients(indices)
    ker=CatmullRomSpline(Float64, Flat)
    field_params=FieldTransformParameters[]

    for i=1:data_params.frames_total
        push!(field_params, FieldTransformParameters(ker,
                                                    0.,
                                                    (0.,0.),
                                                    (-10.7365 , 1.39344),
                                                    polar_params[i][1],
                                                    polar_params[i][2]))
    end

    field_transforms=load_field_transforms(object_params,
                                           data_params,
                                           field_params)
   
    psf_center=readdlm(path*"/PSF_centers_Airy.txt");
    psf=readfits(path*"/PSF_parametered_Airy.fits");
    blur=set_fft_operator(object_params,(psf[1:end÷2,:]'), psf_center[1:2])[1];

    H = DirectModel(size(S), (128,256,4),S.parameter_type,field_transforms,blur)
    BadPixMap=Float64.(rand(0.0:1e-16:1.0,data_params.size).< 0.9);
    # BadPixMap=ones(data_params.size)

    # compute measurements

    data, weight = data_simulator(BadPixMap, field_transforms, S; A=blur);
    S_convolved = PolarimetricMap("stokes", cat(blur*S.I, blur*S.Q, blur*S.U, dims=3)) 

    if write_files == true
        if prod(readdir() .!= "test_results")     
            mkdir("test_results")
        end

        writefits("test_results/DATA_$(data_params.size[1]).fits",
            ["TYPE" => "data"],
            mapslices(transpose,data,dims=[1,2]), overwrite=true)

        writefits("test_results/WEIGHT_$(data_params.size[1]).fits", 
            ["TYPE" => "weights"],
            mapslices(transpose,weight,dims=[1,2]), overwrite=true)

        write(S, "test_results/TRUE_$(data_params.size[1]).fits")

        write(S_convolved, "test_results/TRUE_convolved_$(data_params.size[1]).fits")
    end
    
    # DataSet for gradient computation
    D = Dataset(data, weight , H)
    nAS = sum(D.weights .* abs2.(D.direct_model*STAR))

    return D, STAR.I, nAS
end

function comp_grad(x::AbstractArray{T,3}, D) where {T<:AbstractFloat}  
    S = PolarimetricMap("intensities", x[:, :, 1] - x[:, :, 2], x[:, :, 2], x[:, :, 3])
    g = copy(S)
    res = D.direct_model*S - D.data
    wres = D.weights .* res
    apply!(g, D.direct_model', wres)
    chi2 = dot(res,wres)

    ga = transform_polarimetric_to_array(g, S, x)

    return ga, chi2
end

function comp_grad2(x::AbstractArray{T,3}, D) where {T<:AbstractFloat}  
    S = PolarimetricMap("intensities", x[:, :, 1] - x[:, :, 2], x[:, :, 2], x[:, :, 3])
    g = copy(S)
    res = D.direct_model*S - D.data
    wres = D.weights .* res
    apply!(g, D.direct_model', wres)
    chi2 = dot(res,wres)

    ga = transform_polarimetric_to_array(g, S, x)
    ga[:, :, 2] = fill!(ga[:, :, 2], 0.0)  # Remplit le 2ème canal avec des zéros

    return ga, chi2
end

function comp_residual(x::AbstractArray{T,3}, f::AbstractArray{T,3}, D) where {T<:AbstractFloat}
    """
    Compute the residual: A*x + f - y
    
    Args:
        x: Disk parameters (Iu, Ip, θ) - shape (H, W, 3)
        f: Leakage/stellar contamination - shape (128, 256, 4) matching D.data
        D: Dataset containing data, weights, and direct_model
        
    Returns:
        residual: A*x + f - y (same shape as D.data)
    """
    # Convert x to PolarimetricMap for the direct model
    S = PolarimetricMap("intensities", x[:, :, 1] - x[:, :, 2], x[:, :, 2], x[:, :, 3])
    
    # Compute A*x (forward model applied to disk) - this returns an array with same shape as D.data
    Ax = D.direct_model * S
    
    # Ensure f has the same shape as D.data and Ax
    if size(f) != size(D.data)
        error("f must have the same shape as D.data: $(size(D.data)), but got: $(size(f))")
    end
    
    # Compute residual: A*x + f - y
    residual = Ax .+ f .- D.data
    
    return residual
end

function comp_grad_x(x::AbstractArray{T,3}, f::AbstractArray{T,3}, D) where {T<:AbstractFloat}
    """
    Compute gradient with respect to x: A^T * W * (A*x + f - y)
    
    Args:
        x: Disk parameters (Iu, Ip, θ) - shape (H, W, 3)
        f: Leakage/stellar contamination - shape (128, 256, 4) matching D.data
        D: Dataset containing data, weights, and direct_model
        
    Returns:
        gradient: Gradient with respect to x in same format as input x
        chi2: Chi-square value for monitoring
    """
    # Compute residual using the corrected comp_residual
    residual = comp_residual(x, f, D)
    
    # Apply weights
    weighted_residual = D.weights .* residual
    
    # Compute chi-square for monitoring
    chi2 = dot(residual, weighted_residual)
    
    # Apply adjoint of direct model: A^T * (weighted residual)
    S = PolarimetricMap("intensities", x[:, :, 1] - x[:, :, 2], x[:, :, 2], x[:, :, 3])
    g = copy(S)  # Create output container
    apply!(g, D.direct_model', weighted_residual)
    
    # Transform back to array format
    ga = transform_polarimetric_to_array(g, S, x)
    
    return ga, chi2
end

function comp_grad_f(x::AbstractArray{T,3}, f::AbstractArray{T,3}, D) where {T<:AbstractFloat}
    """
    Compute gradient with respect to f: W * (A*x + f - y)
    
    Args:
        x: Disk parameters (Iu, Ip, θ) - shape (H, W, 3)
        f: Leakage/stellar contamination - shape (128, 256, 4) matching D.data
        D: Dataset containing data, weights, and direct_model
        
    Returns:
        gradient: Gradient with respect to f (same shape as f)
        chi2: Chi-square value for monitoring
    """
    # Compute residual using the corrected comp_residual
    residual = comp_residual(x, f, D)
    
    # Apply weights - this IS the gradient with respect to f
    grad_f = D.weights .* residual
    
    # Compute chi-square for monitoring
    chi2 = dot(residual, grad_f)
    
    return grad_f, chi2
end

function comp_joint_evaluation(x::AbstractArray{T,3}, f::AbstractArray{T,3}, D) where {T<:AbstractFloat}
    """
    Complete evaluation of the joint estimation problem.
    
    Args:
        x: Disk parameters - shape (H, W, 3)
        f: Leakage/stellar contamination - shape (128, 256, 4) matching D.data
        D: Dataset
        
    Returns:
        grad_x: Gradient with respect to x
        grad_f: Gradient with respect to f  
        chi2: Chi-square value
        residual: The residual A*x + f - y
    """
    # Compute residual once using the corrected comp_residual
    residual = comp_residual(x, f, D)
    weighted_residual = D.weights .* residual
    chi2 = dot(residual, weighted_residual)
    
    # Gradient with respect to x: A^T * W * residual
    S = PolarimetricMap("intensities", x[:, :, 1] - x[:, :, 2], x[:, :, 2], x[:, :, 3])
    g_x = copy(S)
    apply!(g_x, D.direct_model', weighted_residual)
    grad_x = transform_polarimetric_to_array(g_x, S, x)
    
    # Gradient with respect to f: W * residual
    grad_f = weighted_residual
    
    return grad_x, grad_f, chi2, residual
end

function generate_star(parameters::ObjectParameters, alpha=1.0)
	Ip=zeros(parameters.size);
	θ=zeros(parameters.size);
	STAR1=zeros(parameters.size);
	STAR2=zeros(parameters.size);
	
	for i=1:parameters.size[1]
    	for j=1:parameters.size[2]
			STAR1[i,j]=200*exp(-((i-parameters.center[1])^2+(j-parameters.center[2])^2)/(2*75^2))
			STAR2[i,j]=100000*exp(-((i-parameters.center[1])^2+(j-parameters.center[2])^2)/(2*7^2))
			if (((parameters.center[1]-i)^2+(parameters.center[2]-j)^2)<=10^2)
        		STAR2[i,j]=800;
    		end
			if (((parameters.center[1]-i)^2+(parameters.center[2]-j)^2)<=70^2)
        		STAR1[i,j]=50;		
    		end
		end
	end    
	STAR=STAR1+STAR2
	#STAR[round(Int64,10*parameters.size[1]/16)-3,round(Int64,10*parameters.size[2]/16)]=20000.0;
	#STAR[round(Int64,10*parameters.size[1]/16),round(Int64,10*parameters.size[2]/16)-3]=100000.0;

    return PolarimetricMap("intensities", alpha*STAR, Ip, θ)
end

function transform_polarimetric_to_array(g::PolarimetricMap{T}, S::PolarimetricMap{T}, x::AbstractArray{T,3}) where {T<:AbstractFloat}
    # Création d'une copie du tableau d'entrée
    ga = copy(x)
    
    # Assignation des composantes transformées
    ga[:, :, 1] = g.I
    ga[:, :, 2] = cos.(2*S.θ).*g.Q + sin.(2*S.θ).*g.U
    # ga[:, :, 3] = 2*S.Ip.*(-sin.(2*S.θ).*g.Q + cos.(2*S.θ).*g.U)  # Pour l'angle de polarisation
    ga[:, :, 3] = fill!(ga[:, :, 3], 0.0)  # Remplit le 3ème canal avec des zéros
    
    return ga
end

end # module compgrad_Rhapsodie
