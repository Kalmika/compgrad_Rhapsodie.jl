using compgrad_Rhapsodie
using HDF5
using Plots
using Printf

function test_comp_grad()

    # Initialize Rhapsodie system with proper arguments
    println("🚀 Initializing Rhapsodie...")

    # Option 1: Utiliser les valeurs par défaut
    D, S, nAS = init_rhapsodie()

    # Option 2: Spécifier explicitement les paramètres (si tu veux personnaliser)
    # D, S, nAS = init_rhapsodie(alpha=1.0, write_files=false, path_disk="default")

    # Option 3: Utiliser un fichier spécifique
    # path = replace(pathof(compgrad_Rhapsodie), "src/compgrad_Rhapsodie.jl" => "data")
    # path_disk = path * "/sample_for_rhapsodie_128x128.h5"
    # D, S, nAS = init_rhapsodie(path_disk=path_disk)

    println("✅ Rhapsodie initialized successfully")
    println("📊 Dataset dimensions: ", size(D.data))
    println("⭐ Star dimensions: ", size(S))
    println("📐 nAS value: ", nAS)

    # Load disk theta values
    path = replace(pathof(compgrad_Rhapsodie), "src/compgrad_Rhapsodie.jl" => "data")
    path_disk = path * "/sample_for_rhapsodie_128x128.h5"
    dset = h5open(path_disk, "r")
    θ = read(dset["disk_theta"])
    close(dset)
    println("✅ Disk theta loaded, dimensions: ", size(θ))

    # Create test data (128x128x3)
    println("\n🧪 Creating test data...")
    x = rand(Float64, 128, 128, 3)

    # Option: Use real data from the file
    dset = h5open(path_disk, "r")
    disk_data = read(dset["disk_data"])
    close(dset)

    # Fill with realistic data
    x[:, :, 1] = disk_data[:, :, 1]  # I (total intensity)
    x[:, :, 2] = disk_data[:, :, 2]  # Ip (polarized intensity)  
    x[:, :, 3] = θ                   # theta (polarization angle)

    println("✅ Test data created")
    println("   - x dimensions: ", size(x))
    println("   - x range: [", minimum(x), ", ", maximum(x), "]")

    # Test comp_grad function
    println("\n🔬 Testing comp_grad...")
    try
        g, chi2 = comp_grad(x, D)
        
        println("✅ comp_grad executed successfully!")
        println("   - Gradient dimensions: ", size(g))
        println("   - Chi² value: ", @sprintf("%.6e", chi2))
        println("   - Gradient range: [", minimum(g), ", ", maximum(g), "]")
        
        # Visualize results (optional)
        println("\n📈 Creating visualization...")
        
        # Plot input data
        p1 = heatmap(x[:, :, 1], title="Input I", color=:viridis, aspect_ratio=:equal)
        p2 = heatmap(x[:, :, 2], title="Input Ip", color=:viridis, aspect_ratio=:equal)
        p3 = heatmap(x[:, :, 3], title="Input θ", color=:viridis, aspect_ratio=:equal)
        
        # Plot gradients
        p4 = heatmap(g[:, :, 1], title="Gradient I", color=:viridis, aspect_ratio=:equal)
        p5 = heatmap(g[:, :, 2], title="Gradient Ip", color=:viridis, aspect_ratio=:equal)
        p6 = heatmap(g[:, :, 3], title="Gradient θ", color=:viridis, aspect_ratio=:equal)
        
        # Combine plots
        plot_combined = plot(p1, p2, p3, p4, p5, p6, layout=(2, 3), size=(900, 600))
        
        # Create output directory if it doesn't exist
        output_dir = "output"
        if !isdir(output_dir)
            mkdir(output_dir)
        end
        
        # Save plot
        savefig(plot_combined, joinpath(output_dir, "comp_grad_test_results.png"))
        println("✅ Visualization saved as 'output/comp_grad_test_results.png'")
        
    catch e
        println("❌ Error in comp_grad:")
        println(e)
        
        # Debug information
        println("\n🔍 Debug information:")
        println("   - D type: ", typeof(D))
        println("   - x type: ", typeof(x))
        println("   - x size: ", size(x))
        
        if isdefined(Main, :D) && hasfield(typeof(D), :direct_model)
            println("   - D.direct_model type: ", typeof(D.direct_model))
        end
    end

    println("\n🏁 Test completed!")
end 

test_comp_grad()