module compgrad_Rhapsodie

using RhapsodieDirect
using DelimitedFiles
using EasyFITS
using InterpolationKernels
using HDF5
using LinearAlgebra

export comp_grad, init_rhapsodie


function init_rhapsodie(; write_files=false, path_disk = "default")
    
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

    S = PolarimetricMap("intensities", I - Ip, Ip, θ)

    # create model
    
    data_params=DatasetParameters((128,256), 4, 2,8, (64.,64.))
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

    # compute measurements

    data, weight = data_simulator(BadPixMap, field_transforms, blur, S);
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
    # -> à vérifier
    D = Dataset(data, weight , H)
    
    return D
end

function comp_grad(x::AbstractArray{T,3}, D) where {T<:AbstractFloat}  
    S = PolarimetricMap("intensities", x[:, :, 1] - x[:, :, 2], x[:, :, 2], x[:, :, 3])
    g = copy(S)
    res = D.direct_model*S - D.data
    wres = D.weights .* res
    apply!(g, D.direct_model', wres)
    chi2 = dot(res,wres)

    ga = copy(x)
     
    ga[:, :, 1] = g.I
    ga[:, :, 2] = g.I + cos.(2*S.θ).*g.Q + sin.(2*S.θ).*g.U
    ga[:, :, 3] = 2*S.Ip.*(-sin.(2*S.θ).*g.Q + cos.(2*S.θ).*g.U)

    #ga[:, :, 2] = g.I + cos.(2*x[:, :, 3]).*g.Q + sin.(2*x[:, :, 3]).*g.U
    #ga[:, :, 3] = 2*S.Ip.*(-sin.(2*x[:, :, 3]).*g.Q + cos.(2*x[:, :, 3]).*g.U)

    return ga, chi2
end


end # module compgrad_Rhapsodie
