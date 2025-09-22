using compgrad_Rhapsodie
using RhapsodieDirect
using LinearAlgebra
using Plots
using Statistics
using Printf

function test_joint_estimation_functions()
    println("=== Test des nouvelles fonctions d'estimation conjointe ===\n")
    
    # 1. Initialize a simple dataset
    println("1. Initialisation du dataset...")
    D, star, nAS = init_rhapsodie2(alpha=1/35, write_files=false)
    
    # Get dimensions
    data_shape = size(D.data)
    object_shape = (128, 128, 3)
    
    println("   - Forme des données: $(data_shape)")
    println("   - Forme de l'objet: $(object_shape)")
    
    # 2. Create simple test inputs
    println("\n2. Création des entrées de test...")
    
    # Simple x: small disk in the center
    x = zeros(Float64, object_shape)
    center = (64, 64)
    radius = 10
    
    for i in 1:object_shape[1], j in 1:object_shape[2]
        dist = sqrt((i - center[1])^2 + (j - center[2])^2)
        if dist <= radius
            x[i, j, 1] = 100.0  # I (will become Iu after I - Ip)
            x[i, j, 2] = 20.0   # Ip 
            x[i, j, 3] = π/4    # θ (45 degrees)
        end
    end
    
    # Simple f: uniform small contamination
    f = 5.0 * ones(Float64, data_shape)
    
    println("   - x: disque simple avec Iu_max=$(maximum(x[:,:,1])), Ip_max=$(maximum(x[:,:,2]))")
    println("   - f: contamination uniforme de valeur $(f[1,1,1])")
    
    # 3. Test each function
    println("\n3. Test des fonctions individuelles...")
    
    # Test comp_residual
    println("\n   a) Test comp_residual:")
    residual = comp_residual(x, f, D)
    println("      - Forme du résidu: $(size(residual))")
    println("      - Min/Max résidu: $(minimum(residual)) / $(maximum(residual))")
    
    # Test comp_grad_x  
    println("\n   b) Test comp_grad_x:")
    grad_x, chi2_x = comp_grad_x(x, f, D)
    println("      - Forme grad_x: $(size(grad_x))")
    println("      - Chi2: $(chi2_x)")
    println("      - Norme grad_x: $(norm(grad_x))")
    
    # Test comp_grad_f
    println("\n   c) Test comp_grad_f:")
    grad_f, chi2_f = comp_grad_f(x, f, D)
    println("      - Forme grad_f: $(size(grad_f))")
    println("      - Chi2: $(chi2_f)")
    println("      - Norme grad_f: $(norm(grad_f))")
    
    # Test comp_joint_evaluation
    println("\n   d) Test comp_joint_evaluation:")
    grad_x_joint, grad_f_joint, chi2_joint, residual_joint = comp_joint_evaluation(x, f, D)
    println("      - Chi2 joint: $(chi2_joint)")
    
    # 4. Verification tests
    println("\n4. Tests de vérification...")
    
    # Check consistency between individual and joint functions
    println("\n   a) Cohérence entre fonctions individuelles et jointes:")
    println("      - |grad_x - grad_x_joint| = $(norm(grad_x - grad_x_joint))")
    println("      - |grad_f - grad_f_joint| = $(norm(grad_f - grad_f_joint))")
    println("      - |chi2_x - chi2_joint| = $(abs(chi2_x - chi2_joint))")
    println("      - |residual - residual_joint| = $(norm(residual - residual_joint))")
    
    # Check that chi2_x and chi2_f should be the same (same residual)
    println("\n   b) Chi2 values should be identical:")
    println("      - chi2_x = $(chi2_x)")
    println("      - chi2_f = $(chi2_f)")
    println("      - |chi2_x - chi2_f| = $(abs(chi2_x - chi2_f))")
    
    # 5. Gradient checking with finite differences
    println("\n5. Vérification du gradient par différences finies...")
    
    function objective(x_test, f_test)
        residual_test = comp_residual(x_test, f_test, D)
        return 0.5 * dot(residual_test, D.weights_op .* residual_test)
    end
    
    # Test gradient with respect to x
    eps = 1e-6
    x_pert = copy(x)
    x_pert[32, 32, 1] += eps  # Perturb one element
    
    obj_original = objective(x, f)
    obj_perturbed = objective(x_pert, f)
    finite_diff_x = (obj_perturbed - obj_original) / eps
    analytical_grad_x = grad_x[32, 32, 1]
    
    println("   a) Gradient par rapport à x[32,32,1]:")
    println("      - Différence finie: $(finite_diff_x)")
    println("      - Gradient analytique: $(analytical_grad_x)")
    println("      - Erreur relative: $(abs(finite_diff_x - analytical_grad_x) / abs(finite_diff_x))")
    
    # Test gradient with respect to f
    f_pert = copy(f)
    f_pert[32, 64, 2] += eps  # Perturb one element
    
    obj_perturbed_f = objective(x, f_pert)
    finite_diff_f = (obj_perturbed_f - obj_original) / eps
    analytical_grad_f = grad_f[32, 64, 2]
    
    println("\n   b) Gradient par rapport à f[32,64,2]:")
    println("      - Différence finie: $(finite_diff_f)")
    println("      - Gradient analytique: $(analytical_grad_f)")
    println("      - Erreur relative: $(abs(finite_diff_f - analytical_grad_f) / abs(finite_diff_f))")
    
    # 6. Summary
    println("\n6. Résumé:")
    consistency_ok = (norm(grad_x - grad_x_joint) < 1e-10) && (norm(grad_f - grad_f_joint) < 1e-10)
    chi2_ok = abs(chi2_x - chi2_f) < 1e-10
    grad_x_ok = abs(finite_diff_x - analytical_grad_x) / abs(finite_diff_x) < 1e-3
    grad_f_ok = abs(finite_diff_f - analytical_grad_f) / abs(finite_diff_f) < 1e-3
    
    println("   - Cohérence des fonctions: $(consistency_ok ? "✓" : "✗")")
    println("   - Chi2 identiques: $(chi2_ok ? "✓" : "✗")")
    println("   - Gradient x correct: $(grad_x_ok ? "✓" : "✗")")
    println("   - Gradient f correct: $(grad_f_ok ? "✓" : "✗")")
    
    all_tests_passed = consistency_ok && chi2_ok && grad_x_ok && grad_f_ok
    println("\n   🎯 RÉSULTAT GLOBAL: $(all_tests_passed ? "TOUS LES TESTS PASSENT ✓" : "CERTAINS TESTS ÉCHOUENT ✗")")
    
    return all_tests_passed
end

function test_gradient_descent(n_iterations=1000; lr_x=0.01, lr_f=0.001, plot_results=true)
    println("=== Test de descente de gradient conjointe ===\n")
    
    # 1. Initialize dataset
    println("1. Initialisation du dataset...")
    D, star, nAS = init_rhapsodie2(alpha=1/35, write_files=false)
    
    data_shape = size(D.data)
    object_shape = (128, 128, 3)
    
    println("   - Forme des données: $(data_shape)")
    println("   - Learning rates: lr_x=$lr_x, lr_f=$lr_f")
    
    # 2. Create target (ground truth) 
    println("\n2. Création de la vérité terrain...")
    
    # Create a more interesting target with multiple features
    x_true = zeros(Float64, object_shape)
    center = (64, 64)
    
    # Main disk
    for i in 1:object_shape[1], j in 1:object_shape[2]
        dist = sqrt((i - center[1])^2 + (j - center[2])^2)
        if dist <= 15
            x_true[i, j, 1] = 150.0 * exp(-dist^2 / (2*8^2))  # Iu (Gaussian)
            x_true[i, j, 2] = 30.0 * exp(-dist^2 / (2*8^2))   # Ip 
            x_true[i, j, 3] = atan(j - center[2], i - center[1])  # θ (radial)
        end
        
        # Add a smaller secondary feature
        dist2 = sqrt((i - 90)^2 + (j - 90)^2)
        if dist2 <= 8
            x_true[i, j, 1] += 80.0 * exp(-dist2^2 / (2*4^2))
            x_true[i, j, 2] += 20.0 * exp(-dist2^2 / (2*4^2))
            x_true[i, j, 3] = π/3  # Fixed angle
        end
    end
    
    # Ensure physical constraints on x_true
    x_true[:,:,1] = max.(x_true[:,:,1], x_true[:,:,2])  # Iu ≥ Ip (so that I = Iu - Ip + Ip = Iu ≥ 0)
    x_true[:,:,2] = max.(x_true[:,:,2], 0.0)  # Ip ≥ 0
    
    # Create structured contamination f_true
    f_true = zeros(Float64, data_shape)
    for k in 1:data_shape[3]
        # Different pattern for each frame
        for i in 1:data_shape[1], j in 1:data_shape[2]
            f_true[i, j, k] = 2.0 + 3.0 * sin(2π * i / 40) * cos(2π * j / 60) * (k/data_shape[3])
        end
    end
    
    println("   - x_true: Iu_max=$(maximum(x_true[:,:,1])), Ip_max=$(maximum(x_true[:,:,2]))")
    println("   - f_true: range [$(minimum(f_true)), $(maximum(f_true))]")
    
    # 3. Generate synthetic measurements
    println("\n3. Génération des mesures synthétiques...")
    residual_true = comp_residual(x_true, f_true, D)
    y_synthetic = residual_true + D.data  # This gives us A*x_true + f_true
    
    # Replace D.data with synthetic data
    D_synthetic = Dataset(y_synthetic, D.weights_op, D.direct_model)
    
    # Add some noise
    noise_level = 0.05 * std(y_synthetic)
    y_noisy = y_synthetic + noise_level * randn(size(y_synthetic))
    D_noisy = Dataset(y_noisy, D.weights_op, D.direct_model)
    
    println("   - Niveau de bruit: $(noise_level)")
    
    # 4. Initialize optimization variables (with some offset from truth)
    println("\n4. Initialisation de l'optimisation...")
    
    # Start with zero + small random perturbation, but ensure physical constraints
    x_opt = 0.1 * x_true + 0.01 * randn(size(x_true))
    x_opt[:,:,2] = max.(x_opt[:,:,2], 0.0)  # Ip ≥ 0
    x_opt[:,:,1] = max.(x_opt[:,:,1], x_opt[:,:,2])  # Iu ≥ Ip
    
    f_opt = 0.1 * f_true + 0.01 * randn(size(f_true))
    
    # Storage for convergence tracking
    chi2_history = Float64[]
    grad_x_norm_history = Float64[]
    grad_f_norm_history = Float64[]
    x_error_history = Float64[]  # ||x_opt - x_true||
    f_error_history = Float64[]  # ||f_opt - f_true||
    
    # 5. Gradient descent loop
    println("\n5. Descente de gradient...")
    
    for iter in 1:n_iterations
        # Compute gradients
        grad_x, grad_f, chi2, residual = comp_joint_evaluation(x_opt, f_opt, D_noisy)
        
        # Update parameters
        x_opt .-= lr_x * grad_x
        f_opt .-= lr_f * grad_f
        
        # Enforce physical constraints on x more carefully
        x_opt[:,:,2] = max.(x_opt[:,:,2], 0.0)  # Ip ≥ 0
        x_opt[:,:,1] = max.(x_opt[:,:,1], x_opt[:,:,2])  # Iu ≥ Ip (so I = Iu - Ip + Ip ≥ 0)
        
        # Store metrics
        push!(chi2_history, chi2)
        push!(grad_x_norm_history, norm(grad_x))
        push!(grad_f_norm_history, norm(grad_f))
        push!(x_error_history, norm(x_opt - x_true))
        push!(f_error_history, norm(f_opt - f_true))
        
        # Print progress
        if iter % 100 == 0 || iter == 1
            println("   Iter $iter: Chi2=$(Printf.@sprintf("%.2e", chi2)), |grad_x|=$(Printf.@sprintf("%.2e", norm(grad_x))), |grad_f|=$(Printf.@sprintf("%.2e", norm(grad_f)))")
        end
    end
    
    # 6. Final evaluation
    println("\n6. Évaluation finale...")
    final_chi2 = chi2_history[end]
    final_x_error = x_error_history[end]
    final_f_error = f_error_history[end]
    
    println("   - Chi2 final: $(Printf.@sprintf("%.2e", final_chi2))")
    println("   - Erreur x finale: $(Printf.@sprintf("%.2e", final_x_error))")
    println("   - Erreur f finale: $(Printf.@sprintf("%.2e", final_f_error))")
    println("   - Réduction Chi2: $(Printf.@sprintf("%.1f", chi2_history[1]/final_chi2))x")
    
    # 7. Create plots
    if plot_results
        println("\n7. Création des graphiques...")
        
        # Convergence plots
        p1 = plot(1:n_iterations, chi2_history, yscale=:log10, 
                  title="Chi² Convergence", xlabel="Iteration", ylabel="Chi²", 
                  linewidth=2, legend=false)
        
        p2 = plot(1:n_iterations, [grad_x_norm_history grad_f_norm_history], 
                  yscale=:log10, title="Gradient Norms", xlabel="Iteration", ylabel="||Gradient||",
                  label=["||grad_x||" "||grad_f||"], linewidth=2)
        
        p3 = plot(1:n_iterations, [x_error_history f_error_history],
                  yscale=:log10, title="Reconstruction Errors", xlabel="Iteration", ylabel="||Error||",
                  label=["||x - x_true||" "||f - f_true||"], linewidth=2)
        
        # Parameter comparison plots
        # Plot x comparison (Iu channel)
        p4 = heatmap(x_true[:,:,1], title="x_true (Iu)", aspect_ratio=:equal, color=:viridis)
        p5 = heatmap(x_opt[:,:,1], title="x_optimized (Iu)", aspect_ratio=:equal, color=:viridis)
        p6 = heatmap(abs.(x_opt[:,:,1] - x_true[:,:,1]), title="|x_opt - x_true| (Iu)", 
                     aspect_ratio=:equal, color=:hot)
        
        # Plot f comparison (first frame)
        p7 = heatmap(f_true[:,:,1], title="f_true (frame 1)", aspect_ratio=:equal, color=:viridis)
        p8 = heatmap(f_opt[:,:,1], title="f_optimized (frame 1)", aspect_ratio=:equal, color=:viridis)
        p9 = heatmap(abs.(f_opt[:,:,1] - f_true[:,:,1]), title="|f_opt - f_true| (frame 1)", 
                     aspect_ratio=:equal, color=:hot)
        
        # Combine all plots
        convergence_plot = plot(p1, p2, p3, layout=(1,3), size=(1200, 400))
        x_comparison_plot = plot(p4, p5, p6, layout=(1,3), size=(1200, 400))
        f_comparison_plot = plot(p7, p8, p9, layout=(1,3), size=(1200, 400))
        
        display(convergence_plot)
        display(x_comparison_plot)
        display(f_comparison_plot)
        
        # Create output directory if it doesn't exist
        output_dir = "output"
        if !isdir(output_dir)
            mkdir(output_dir)
        end
        
        # Save plots
        savefig(convergence_plot, joinpath(output_dir, "comp_grad_xf_test_convergence_analysis.png"))
        savefig(x_comparison_plot, joinpath(output_dir, "comp_grad_test_x_parameter_comparison.png"))
        savefig(f_comparison_plot, joinpath(output_dir, "comp_grad_test_f_parameter_comparison.png"))
        
        println("   - Graphiques sauvegardés: convergence_analysis.png, x_parameter_comparison.png, f_parameter_comparison.png")
    end
    
    # 8. Summary
    println("\n8. Résumé:")
    convergence_achieved = (final_chi2 < chi2_history[1] * 0.01)  # 99% reduction
    x_recovered = (final_x_error < norm(x_true) * 0.1)  # Within 10% of true norm
    f_recovered = (final_f_error < norm(f_true) * 0.1)  # Within 10% of true norm
    
    println("   - Convergence Chi2: $(convergence_achieved ? "✓" : "✗") (réduction: $(Printf.@sprintf("%.1f", chi2_history[1]/final_chi2))x)")
    println("   - Reconstruction x: $(x_recovered ? "✓" : "✗") (erreur: $(Printf.@sprintf("%.1f", final_x_error/norm(x_true)*100))%)")
    println("   - Reconstruction f: $(f_recovered ? "✓" : "✗") (erreur: $(Printf.@sprintf("%.1f", final_f_error/norm(f_true)*100))%)")
    
    success = convergence_achieved && x_recovered && f_recovered
    println("\n   🎯 RÉSULTAT: $(success ? "SUCCÈS ✓" : "AMÉLIORATION POSSIBLE ⚠️")")
    
    return Dict(
        "success" => success,
        "final_chi2" => final_chi2,
        "chi2_reduction" => chi2_history[1]/final_chi2,
        "x_error_percent" => final_x_error/norm(x_true)*100,
        "f_error_percent" => final_f_error/norm(f_true)*100,
        "x_true" => x_true,
        "x_opt" => x_opt,
        "f_true" => f_true,
        "f_opt" => f_opt,
        "history" => Dict(
            "chi2" => chi2_history,
            "grad_x_norm" => grad_x_norm_history,
            "grad_f_norm" => grad_f_norm_history,
            "x_error" => x_error_history,
            "f_error" => f_error_history
        )
    )
end

# Run the test
test_joint_estimation_functions()
results = test_gradient_descent(100, lr_x=0.01, lr_f=0.01, plot_results=true)