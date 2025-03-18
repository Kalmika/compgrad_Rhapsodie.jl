module compgrad_Rhapsodie

using RhapsodieDirect
using DelimitedFiles
using EasyFITS
using InterpolationKernels
using HDF5
using LinearAlgebra

export comp_grad, init_rhapsodie


function init_rhapsodie(path::String, write_files=false)

    ker = CatmullRomSpline(Float64, Flat)

    # load disk
    object_params=ObjectParameters((128,128),(64.,64.))  # check with file!

    dset = h5open(path*"/sample_for_rhapsodie_128x128.h5", "r")
    I = read(dset["disk_data"])[:,:,1]
    Ip = read(dset["disk_data"])[:,:,2]
    theta = read(dset["disk_theta"])
    close(dset)

    S = PolarimetricMap("intensities", I, Ip, theta)

    # create model

    data_params=DatasetParameters((128,256), 64, 2,8, (64.,64.))
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

    
    psf_center=readdlm(path*"data/PSF_centers_Airy.txt");
    psf=readfits(path*"data/PSF_parametered_Airy.fits");
    blur=set_fft_operator(object_params,(psf[1:end÷2,:]'), psf_center[1:2])[1];

    H = DirectModel(size(S), (128,256,64),S.parameter_type,field_transforms,blur)
    BadPixMap=Float64.(rand(0.0:1e-16:1.0,data_params.size).< 0.9);

    # compute measurements

    data, weight = data_simulator(BadPixMap, field_transforms, blur, S);


    S_convolved = PolarimetricMap("stokes", cat(blur*S.I, blur*S.Q, blur*S.U, dims=3)) 

    write_files && true
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

        # DataSet for gradient computation

        # -> à vérifier
        D = Dataset(data, weight , H)

        return D
end

    # test grad

function comp_grad(x::AbstractArray{T,3}, D) where {T<:AbstractFloat}  
    S = PolarimetricMap("intensities", x[:, :, 1], x[:, :, 2], x[:, :, 3])
    g = copy(S)
    res = D.direct_model*S - D.data
    wres = D.weights .* res
    apply!(g, D.direct_model', wres)
    chi2 = dot(res,wres)

    return g, chi2
end


end # module compgrad_Rhapsodie
