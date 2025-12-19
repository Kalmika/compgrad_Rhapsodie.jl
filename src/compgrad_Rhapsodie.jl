module compgrad_Rhapsodie

using RhapsodieDirect
using DelimitedFiles
using EasyFITS
using InterpolationKernels
using HDF5
using LinearAlgebra

export 
    comp_grad, comp_grad2, 
    init_rhapsodie, generate_star, 
    init_rhapsodie2, 
    comp_residual, 
    comp_grad_x, 
    comp_grad_f, 
    comp_joint_evaluation, 
    init_rhapsodie_leakage, 
    comp_grad_speckle_scalar_leakage, 
    comp_grad_disk_scalar_leakage, 
    PolarimetricMap, 
    string_to_noise_model,
    apply_direct_model, 
    apply_direct_model_transpose,
    apply_direct_model_inverse,
    get_polar_params

function get_polar_params()
    data_params=DatasetParameters((128,256), 4, 1, 1, (64.,64.))
    indices=get_indices_table(data_params)
    return set_default_polarisation_coefficients(indices)
end

function apply_direct_model(input_array::AbstractArray{T,3}, julia_direct_model::DirectModel{T}) where {T<:AbstractFloat}
    S = PolarimetricMap("intensities", input_array[1, :, :] - input_array[2, :, :], input_array[2, :, :], input_array[3, :, :])
    return julia_direct_model * S
end

function apply_direct_model_transpose(input_measurement_vector::AbstractArray{T,3}, julia_direct_model::DirectModel) where {T<:AbstractFloat}
    return julia_direct_model' * input_measurement_vector
end

function apply_direct_model_inverse(input_measurement_vector::AbstractArray{T,3}, julia_direct_model::DirectModel) where {T<:AbstractFloat}
    return julia_direct_model \ input_measurement_vector
end

function string_to_noise_model(noise_model_str::String, size::Int = 128, star_max_intensity::Float64 = 1.0, corr_amplitude::Float64 = 0.0, corr_filter_size::Float64 = 1.5, verbose::Bool = true)
    noise_type = lowercase(strip(noise_model_str))    
    if noise_type == "correlated"
        model = RhapsodieDirect.CorrelatedNoise(corr_amplitude, corr_filter_size, size)
        if verbose
            println("CorrelatedNoise selected with amplitude: $corr_amplitude, filter_size: $corr_filter_size, size: $size (star max: $star_max_intensity)")
        end
    elseif noise_type == "diagonal_and_correlated"
        model = RhapsodieDirect.DiagonalAndCorrelatedNoise(corr_amplitude, corr_filter_size, size)
        if verbose
            println("DiagonalAndCorrelatedNoise selected with amplitude: $corr_amplitude, filter_size: $corr_filter_size, size: $size (star max: $star_max_intensity)")
        end
    else
        model = RhapsodieDirect.DiagonalNoise()
        if verbose
            println("DiagonalNoise selected with size: $size")
        end
    end
    
    return model
end

function init_rhapsodie_leakage(;alpha = 1e-2, write_files=false, data_folder = "default", noise_model_str::String = "diagonal", corr_amplitude::Float64 = 0.0, corr_filter_size::Float64 = 1.5,
    reg_param_relative::Float64 = 1e-3, verbose::Bool = true, is_zero_star::Bool = false, is_zero_disk::Bool = false)
    
    (data_folder ==  "default") && (data_folder = replace(pathof(compgrad_Rhapsodie), "src/compgrad_Rhapsodie.jl" => "data"))
    path_disk = data_folder*"/sample_for_rhapsodie_128x128.h5"
    path_star = data_folder*"/star_cropped.fits"
    ker = CatmullRomSpline(Float64, Flat)

    if verbose
        println("Paths:")
        println("path_disk: ", path_disk)
        println("path_star: ", path_star)
    end

    # --- 1. Chargement des données et paramètres ---
    if verbose
        println("Chargement des données...")
    end
    object_params=ObjectParameters((128,128),(64.,64.))

    dset = h5open(path_disk, "r")
    # permutedims because Julia h5open open in column-major order
    I_disk = permutedims(read(dset["disk_data"])[:,:,1], (2, 1))
    Ip_disk = permutedims(read(dset["disk_data"])[:,:,2], (2, 1))
    θ_disk = permutedims(read(dset["disk_theta"]), (2, 1))
    close(dset)

    STAR_intensity = readfits(path_star)
    coef_disk = alpha*maximum(STAR_intensity)/maximum(I_disk)
    
    # Convert string to NoiseModel for Python compatibility    
    if verbose
        println("-> noise_model_str: ", noise_model_str)
    end
    noise_model = string_to_noise_model(noise_model_str, object_params.size[1], maximum(STAR_intensity), corr_amplitude, corr_filter_size, verbose)

    # Objet ciel complet (disque + étoile) pour la simulation
    S_disk = PolarimetricMap("intensities", coef_disk*(I_disk - Ip_disk), coef_disk*Ip_disk, θ_disk) 
    is_zero_disk && (S_disk = PolarimetricMap("intensities", zero(I_disk), zero(I_disk), zero(I_disk)))
    S_star = PolarimetricMap("intensities",  STAR_intensity, zero(STAR_intensity), zero(STAR_intensity)) 
    is_zero_star && (S_star = PolarimetricMap("intensities", zero(STAR_intensity), zero(STAR_intensity), zero(STAR_intensity)))

    # --- 2. Construction des modèles directs (A et A') ---
    data_params=DatasetParameters((128,256), 4, 1, 1, (64.,64.))
    indices=get_indices_table(data_params)
    polar_params=set_default_polarisation_coefficients(indices)
    
    field_params=FieldTransformParameters[]
    for i=1:data_params.frames_total
        # push!(field_params, FieldTransformParameters(ker, 0., (0.,0.), (-10.7365 , 1.39344), polar_params[i][1], polar_params[i][2]))
        push!(field_params, FieldTransformParameters(ker, 0., (0.,0.), (0.,0.), polar_params[i][1], polar_params[i][2]))
    end
    field_transforms = load_field_transforms(object_params, data_params, field_params)
   
    psf_center = readdlm(data_folder*"/PSF_centers_Airy.txt");
    psf = readfits(data_folder*"/PSF_parametered_Airy.fits");
    blur = set_fft_operator(object_params,(psf[1:end÷2,:]'), psf_center[1:2])[1];

    # H = Modèle COMPLET (A), avec flou. Pour la simulation et les gradients.
    H = DirectModel(size(S_disk), (128,256,4), S_disk.parameter_type, field_transforms, blur)
    
    # H_noblur = Modèle SANS flou (A'). Pour le terme de fuite et nAS.
    H_noblur = DirectModel(size(S_disk), (128,256,4), S_disk.parameter_type, field_transforms)

    # --- 3. Simulation des données observées ---
    BadPixMap = Float64.(rand(0.0:1e-16:1.0, data_params.size) .< 10.9); #TODO modif
    # data, weight = data_simulator_dual_component_bis(BadPixMap, field_transforms, S_disk, S_star; A_disk=blur);
    data, weights_operator = data_simulator_dual_component_bis(BadPixMap, field_transforms, S_disk, S_star; A_disk=blur, noise_model=noise_model, reg_param_relative=reg_param_relative);
    
    # Le Dataset contient le modèle COMPLET H (A)
    D = Dataset(data, weights_operator, H)
    Dstar = Dataset(data, weights_operator, H_noblur)

    # --- 4. Pré-calcul des termes liés à l'étoile (s) ---

    # Calcul de A'*s
    AS_noblur = H_noblur * S_star

    # star_128_256 = zeros(128, 256)
    # star_128_256[:, 1:128] .= S_star_I
    # star_128_256[:, 128:256] .= S_star_I

    # S_star_3 = zeros(128,128,3)
    # S_star_3[:,:,1] = S_star_I


    # Norme : ||A'*s||^2_W
    # Note: On utilise ici la réponse de l'étoile NON pondérée par lambda.
    # Si votre nAS doit inclure lambda, changez `AS_noblur` en `leakage_term` ci-dessous.
    nAS = sum(D.weights_op * abs2.(AS_noblur))

    # --- 5. Sauvegarde optionnelle et retour ---
    if write_files == true
        # ... (code de sauvegarde inchangé) ...
    end
    
    # On retourne D (avec H), l'intensité de l'étoile, le terme de fuite et nAS.
    return D, Dstar, S_star.I, AS_noblur, nAS, S_disk
end

# init_rhapsodie : L'étoile (STAR) est ajoutée directement au disque. L'intensité de l'étoile est contrôlée par un simple facteur d'échelle alpha. C'est une composition physique simple.
# S = Disque + alpha * Étoile

# init_rhapsodie2 : L'étoile (STAR) est ajoutée au disque, mais l'ensemble du disque (polarisé et non polarisé) est d'abord redimensionné par un coefficient coef. Ce coefficient est calculé pour que le flux maximal du disque redimensionné soit proportionnel au flux maximal de l'étoile. C'est une manière plus contrôlée de mélanger les deux composantes en se basant sur leurs flux relatifs.
# S = coef * Disque + Étoile

function init_rhapsodie2(;alpha = 1e-2, write_files=false, data_folder = "default")
    
    (data_folder ==  "default") && (data_folder = replace(pathof(compgrad_Rhapsodie), "src/compgrad_Rhapsodie.jl" => "data"))
    path_disk = data_folder*"/sample_for_rhapsodie_128x128.h5"
    path_star = data_folder*"/star_cropped.fits"
    ker = CatmullRomSpline(Float64, Flat)

    # load disk
    
    object_params=ObjectParameters((128,128),(64.,64.))

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

    # Le modèle H ici inclut le flou. Il sera utilisé pour générer les données
    # et pour les calculs de gradient via l'objet D.
    H = DirectModel(size(S), (128,256,4),S.parameter_type,field_transforms,blur)
    BadPixMap=Float64.(rand(0.0:1e-16:1.0,data_params.size).< 0.9);

    # compute measurements
    data, weight = data_simulator(BadPixMap, field_transforms, S; A=blur);
    
    # DataSet for gradient computation - Il contient le modèle AVEC flou
    D = Dataset(data, weight , H)

    # Création d'une PolarimetricMap pour l'étoile seule
    Ip_star = zeros(object_params.size);
    θ_star = zeros(object_params.size);
    STAR_p = PolarimetricMap("intensities", STAR, Ip_star, θ_star)


    # --- MODIFICATION START ---
    # Pour calculer nAS avec un modèle SANS flou, nous créons un modèle
    # temporaire en omettant l'argument `blur`.
    # Il utilisera `LazyAlgebra.Id` par défaut pour la convolution.
    println("Création d'un modèle temporaire sans flou pour le calcul de nAS...")
    H_noblur = DirectModel(size(S), (128,256,4), S.parameter_type, field_transforms)

    # Calcul de A*S (sans flou)
    AS_noblur = H_noblur * STAR_p
    
    # Calcul de la norme pondérée ||A*S||^2_W
    nAS = sum(D.weights_op .* abs2.(AS_noblur))
    # --- MODIFICATION END ---


    # --- Section optionnelle pour la sauvegarde des fichiers (inchangée) ---
    if write_files == true
        S_convolved = PolarimetricMap("stokes", cat(blur*S.I, blur*S.Q, blur*S.U, dims=3)) 
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
    
    # On retourne D (qui contient H avec flou) et nAS (calculé sans flou)
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
    nAS = sum(D.weights_op .* abs2.(D.direct_model*STAR))

    return D, STAR.I, nAS
end

function comp_grad(x::AbstractArray{T,3}, D) where {T<:AbstractFloat}  
    S = PolarimetricMap("intensities", x[:, :, 1] - x[:, :, 2], x[:, :, 2], x[:, :, 3])
    g = copy(S)
    res = D.direct_model*S - D.data
    wres = D.weights_op .* res
    apply!(g, D.direct_model', wres)
    chi2 = dot(res,wres)

    ga = transform_polarimetric_to_array(g, S, x)

    return ga, chi2
end

function comp_grad_disk_scalar_leakage(x::AbstractArray{T,3}, alpha_s::T, leakage::AbstractArray{T,3}, D; gamma::T = zero(T)) where {T<:AbstractFloat}  
    """
    Compute gradient with respect to x: A^T * W * (A*x + leakage - y)
    
    Args:
        x: Disk parameters (Iu, Ip, θ) - shape (H, W, 3)
        D: Dataset containing data, weights, and the FULL direct_model (A)
        leakage: Pre-computed leakage term (lambda * A' * s)
        gamma: Optional regularization parameter (default: zero)
        
    Returns:
        gradient: Gradient with respect to x
        chi2: Chi-square value
    """
    # 1. Convertir x en PolarimetricMap
    S_disk = PolarimetricMap("intensities", x[:, :, 1] - x[:, :, 2], x[:, :, 2], x[:, :, 3])
    g = copy(S_disk)
    
    # 2. Calculer A*x (en utilisant le modèle AVEC flou de D)
    Ax = D.direct_model * S_disk
    
    # 3. Calculer le résidu : A*x + (lambda(alpha_s) * A'*s) - y
    residual = Ax .+ alpha_s*leakage .- D.data
    
    # 4. Ajout du terme de régularisation gamma si différent de zéro
    if gamma != zero(T) # TODO....
        # residual = residual ./ (1 .+ gamma^2 * D.weights_op)
        residual_gamma = residual ./ (1 .+ gamma^2 * D.weights_op.weights)
    end

    # 5. Appliquer les poids
    weighted_residual = D.weights_op.weights .* residual_gamma
    
    # 6. Calculer le chi-deux
    chi2 = dot(residual_gamma, weighted_residual)
    
    # 7. Appliquer l'adjoint : A^T * (weighted_residual)
    apply!(g, D.direct_model', weighted_residual)
    
    # 8. Transformer le gradient au format tableau
    grad_x = transform_polarimetric_to_array(g, S_disk, x)
    
    return grad_x, chi2
end



function apply_(x::AbstractArray{T,3}, alpha_s::T, leakage::AbstractArray{T,3}, D; gamma::T = zero(T)) where {T<:AbstractFloat}  
    """
    Compute gradient with respect to x: A^T * W * (A*x + leakage - y)
    
    Args:
        x: Disk parameters (Iu, Ip, θ) - shape (H, W, 3)
        D: Dataset containing data, weights, and the FULL direct_model (A)
        leakage: Pre-computed leakage term (lambda * A' * s)
        gamma: Optional regularization parameter (default: zero)
        
    Returns:
        gradient: Gradient with respect to x
        chi2: Chi-square value
    """
    # 1. Convertir x en PolarimetricMap
    S_disk = PolarimetricMap("intensities", x[:, :, 1] - x[:, :, 2], x[:, :, 2], x[:, :, 3])
    g = copy(S_disk)
    
    # 2. Calculer A*x (en utilisant le modèle AVEC flou de D)
    Ax = D.direct_model * S_disk
    
    # 3. Calculer le résidu : A*x + (lambda(alpha_s) * A'*s) - y
    residual = Ax .+ alpha_s*leakage .- D.data
    
    # 4. Ajout du terme de régularisation gamma si différent de zéro
    if gamma != zero(T) # TODO....
        # residual = residual ./ (1 .+ gamma^2 * D.weights_op)
        residual = residual ./ (1 .+ gamma^2 .* D.weights_op)
    end
    
    # 5. Appliquer les poids
    weighted_residual = D.weights_op * residual
    
    # 6. Calculer le chi-deux
    chi2 = dot(residual, weighted_residual)
    
    # 7. Appliquer l'adjoint : A^T * (weighted_residual)
    apply!(g, D.direct_model', weighted_residual)
    
    # 8. Transformer le gradient au format tableau
    grad_x = transform_polarimetric_to_array(g, S_disk, x)
    
    return grad_x, chi2
end


function comp_grad_speckle_scalar_leakage(x::AbstractArray{T,3}, D, D2) where {T<:AbstractFloat}  
    """
    Compute gradient with respect to x: B^T * W * (A*x - y)
    
    Args:
        x: Disk parameters (Iu, Ip, θ) - shape (H, W, 3)
        D: Dataset containing data, weights, and the FULL direct_model (A)
        D: Dataset containing data, weights, and the FULL second direct_model (B)
        
    Returns:
        gradient: Gradient with respect to x
        chi2: Chi-square value
    """
    # 1. Convertir x en PolarimetricMap
    S_disk = PolarimetricMap("intensities", x[:, :, 1] - x[:, :, 2], x[:, :, 2], x[:, :, 3])
    g = copy(S_disk)
    
    # 2. Calculer A*x (en utilisant le modèle AVEC flou de D)
    Ax = D.direct_model * S_disk
    
    # 3. Calculer le résidu : A*x - y
    residual = Ax .- D.data
    
    # 5. Appliquer les poids
    weighted_residual = D.weights_op * residual

    # 6. Calculer le chi-deux
    chi2 = dot(residual, weighted_residual)

    # 7. Appliquer l'adjoint : A^T * (weighted_residual)
    apply!(g, D2.direct_model', weighted_residual)

    # 8. Transformer le gradient au format tableau
    grad_x = transform_polarimetric_to_array(g, S_disk, x)
    
    return grad_x, chi2
end

function comp_grad_scalar_leakage_alpha_star(x::AbstractArray{T,3}, alpha_s::T, leakage::AbstractArray{T,3}, D) where {T<:AbstractFloat}
    (data_folder ==  "default") && (data_folder = replace(pathof(compgrad_Rhapsodie), "src/compgrad_Rhapsodie.jl" => "data"))
    path_disk = data_folder*"/sample_for_rhapsodie_128x128.h5"
    path_star = data_folder*"/star_cropped.fits"
    ker = CatmullRomSpline(Float64, Flat)

    # --- 1. Chargement des données et paramètres ---
    object_params=ObjectParameters((128,128),(64.,64.))

    dset = h5open(path_disk, "r")
    I_disk = read(dset["disk_data"])[:,:,1]
    Ip_disk = read(dset["disk_data"])[:,:,2]
    θ_disk = read(dset["disk_theta"])
    close(dset)
    STAR_intensity = readfits(path_star)
    S_total = PolarimetricMap("intensities", I_disk - Ip_disk + STAR_intensity, Ip_disk, θ_disk) 
    println("Construction des modèles directs...")
    data_params=DatasetParameters((128,256), 4, 1, 1, (64.,64.))
    indices=get_indices_table(data_params)
    polar_params=set_default_polarisation_coefficients(indices)
    
    field_params=FieldTransformParameters[]
    for i=1:data_params.frames_total
        push!(field_params, FieldTransformParameters(ker, 0., (0.,0.), (-10.7365 , 1.39344), polar_params[i][1], polar_params[i][2]))
    end
    field_transforms = load_field_transforms(object_params, data_params, field_params)
   
    psf_center = readdlm(data_folder*"/PSF_centers_Airy.txt");
    psf = readfits(data_folder*"/PSF_parametered_Airy.fits");
    blur = set_fft_operator(object_params,(psf[1:end÷2,:]'), psf_center[1:2])[1];

    # H = Modèle COMPLET (A), avec flou. Pour la simulation et les gradients.
    H = DirectModel(size(S_total), (128,256,4), S_total.parameter_type, field_transforms, blur)
    
    # H_noblur = Modèle SANS flou (A'). Pour le terme de fuite et nAS.
    H_noblur = DirectModel(size(S_total), (128,256,4), S_total.parameter_type, field_transforms)

    # --- 3. Simulation des données observées ---
    println("Simulation des données observées...")
    BadPixMap = Float64.(rand(0.0:1e-16:1.0, data_params.size) .< 0.9);
    data, weight = data_simulator(BadPixMap, field_transforms, S_total; A=blur);
    
    # Le Dataset contient le modèle COMPLET H (A)
    D = Dataset(data, weight, H)

    # --- 4. Pré-calcul des termes liés à l'étoile (s) ---
    println("Pré-calcul des termes de fuite stellaire...")
    
    # Création d'une PolarimetricMap pour l'étoile seule (s)
    Ip_star = zeros(object_params.size)
    θ_star = zeros(object_params.size)
    STAR_p = PolarimetricMap("intensities", STAR_intensity, Ip_star, θ_star)

    # Calcul de A'*s
    AS_noblur = H_noblur * STAR_p
    
    # Norme : ||A'*s||^2_W
    # Note: On utilise ici la réponse de l'étoile NON pondérée par lambda.
    # Si votre nAS doit inclure lambda, changez `AS_noblur` en `leakage_term` ci-dessous.
    nAS = sum(D.weights_op .* abs2.(AS_noblur))

    # --- 5. Sauvegarde optionnelle et retour ---
    if write_files == true
        # ... (code de sauvegarde inchangé) ...
    end
    
    println("Initialisation terminée.")
    # On retourne D (avec H), l'intensité de l'étoile, le terme de fuite et nAS.
    return D, STAR_p.I, AS_noblur, nAS
end



function comp_grad2(x::AbstractArray{T,3}, D) where {T<:AbstractFloat}  
    S = PolarimetricMap("intensities", x[:, :, 1] - x[:, :, 2], x[:, :, 2], x[:, :, 3])
    g = copy(S)
    res = D.direct_model*S - D.data
    wres = D.weights_op .* res
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
    weighted_residual = D.weights_op .* residual
    
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
    grad_f = D.weights_op .* residual
    
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
    weighted_residual = D.weights_op .* residual
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
