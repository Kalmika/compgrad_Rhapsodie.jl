#
# test_blur_on_polarized_object_full.jl
#
# Ce script teste l'effet du flou sur un objet synthétique polarisé et
# sauvegarde les résultats dans un dossier "output".
#

using Pkg 
Pkg.activate(".")

using RhapsodieDirect
import RhapsodieDirect: default_kernel
using InterpolationKernels
using EasyFITS
using DelimitedFiles
using Plots
using Images
using HDF5
using LazyAlgebra


# --- 1. Initialisation des paramètres et des dossiers ---
data_folder = "data"
output_folder = "output" # <--- AJOUT : Définir le nom du dossier de sortie

# Créer le dossier de sortie s'il n'existe pas
if !isdir(output_folder) # <--- AJOUT
    mkdir(output_folder) # <--- AJOUT
    println("Dossier '$output_folder' créé.") # <--- AJOUT
end

object_params = RhapsodieDirect.ObjectParameters((128, 128), (64.0, 64.0))
data_params = RhapsodieDirect.DatasetParameters((128, 256), 4, 1, 1, (64.0, 64.0))
ker = CatmullRomSpline(Float64, Flat)

# --- 2. Création d'un objet de test POLARISÉ simple ---
println("Création d'un objet de test simple (carré polarisé)...")
I_map = zeros(object_params.size)
Q_map = zeros(object_params.size)
U_map = zeros(object_params.size)

center_x, center_y = round.(Int, object_params.center)
half_size = 20
y_range = (center_y - half_size):(center_y + half_size)
x_range = (center_x - half_size):(center_x + half_size)

I_map[y_range, x_range] .= 1000.0
Q_map[y_range, x_range] .= 500.0
U_map[y_range, x_range] .= -300.0

test_object = RhapsodieDirect.PolarimetricMap("stokes", I_map, Q_map, U_map)

# --- 3. Construction des modèles et de l'opérateur de flou ---
println("Construction des modèles et de l'opérateur de flou...")
field_params = RhapsodieDirect.FieldTransformParameters[]
indices = RhapsodieDirect.get_indices_table(data_params)
polar_params = RhapsodieDirect.set_default_polarisation_coefficients(indices)
for i in 1:data_params.frames_total
    push!(field_params, RhapsodieDirect.FieldTransformParameters(ker, 0.0, (0.0, 0.0), (-10.7365, 1.39344), polar_params[i][1], polar_params[i][2]))
end
field_transforms = RhapsodieDirect.load_field_transforms(object_params, data_params, field_params)

psf_center = readdlm(joinpath(data_folder, "PSF_centers_Airy.txt"))
psf = readfits(joinpath(data_folder, "PSF_parametered_Airy.fits"))
blur = RhapsodieDirect.set_fft_operator(object_params, (psf[1:end÷2,:]'), psf_center[1:2])[1]

H = RhapsodieDirect.DirectModel(size(test_object), (128, 256, 4), test_object.parameter_type, field_transforms, blur)
H_noblur = RhapsodieDirect.DirectModel(size(test_object), (128, 256, 4), test_object.parameter_type, field_transforms)

# --- 4. Application des modèles ---
println("Application des modèles...")
data_blurred = H * test_object
data_sharp = H_noblur * test_object

# --- 5. Visualisation ---
println("Création des plots...")

# Objet original
p1 = heatmap(I_map, title="I original", aspect_ratio=:equal, color=:viridis)
p2 = heatmap(Q_map, title="Q original", aspect_ratio=:equal, color=:viridis)
p3 = heatmap(U_map, title="U original", aspect_ratio=:equal, color=:viridis)

# Données simulées (frame 1, première moitié = I+Q)
p4 = heatmap(data_sharp[:,:,1], title="Sharp I+Q", aspect_ratio=:equal, color=:viridis)
p5 = heatmap(data_blurred[:,:,1], title="Blurred I+Q", aspect_ratio=:equal, color=:viridis)

# Données simulées (frame 1, seconde moitié = I-Q)
p6 = heatmap(data_sharp[:,129:end,1], title="Sharp I-Q", aspect_ratio=:equal, color=:viridis)
p7 = heatmap(data_blurred[:,129:end,1], title="Blurred I-Q", aspect_ratio=:equal, color=:viridis)

# Comparaison différence
diff_IQ = data_blurred[:,:,1] - data_sharp[:,:,1]
p8 = heatmap(diff_IQ, title="Différence flou", aspect_ratio=:equal, color=:RdBu)

# Assemblage et sauvegarde
plot_combined = plot(p1, p2, p3, p4, p5, p6, p7, p8, layout=(2,4), size=(1600,800))
savefig(plot_combined, joinpath(output_folder, "test_forward_model.png"))


# --- 6. Test des gradients ---
println("Test des gradients...")

# Données de référence (on utilise les données floutées comme "vraies" données)
y_true = data_blurred

# Fonction de loss (moindres carrés)
function compute_loss(obj, model, y_ref)
    y_pred = model * obj
    return 0.5 * sum((y_pred - y_ref).^2)
end

# Gradient analytique via adjoint
function analytical_gradient(obj, model, y_ref)
    y_pred = model * obj
    residual = y_pred - y_ref
    return model' * residual
end

# Gradient numérique par différences finies
function numerical_gradient(obj, model, y_ref, eps=1e-6)
    I_grad = zeros(size(obj.I))
    Q_grad = zeros(size(obj.Q))
    U_grad = zeros(size(obj.U))
    
    loss_base = compute_loss(obj, model, y_ref)
    
    # Test sur I
    for i in 1:size(obj.I, 1)
        for j in 1:size(obj.I, 2)
            I_pert = copy(obj.I)
            I_pert[i, j] += eps
            obj_plus = RhapsodieDirect.PolarimetricMap("stokes", I_pert, obj.Q, obj.U)
            loss_plus = compute_loss(obj_plus, model, y_ref)
            I_grad[i, j] = (loss_plus - loss_base) / eps
        end
    end
    
    # Test sur Q
    for i in 1:size(obj.Q, 1)
        for j in 1:size(obj.Q, 2)
            Q_pert = copy(obj.Q)
            Q_pert[i, j] += eps
            obj_plus = RhapsodieDirect.PolarimetricMap("stokes", obj.I, Q_pert, obj.U)
            loss_plus = compute_loss(obj_plus, model, y_ref)
            Q_grad[i, j] = (loss_plus - loss_base) / eps
        end
    end
    
    # Test sur U
    for i in 1:size(obj.U, 1)
        for j in 1:size(obj.U, 2)
            U_pert = copy(obj.U)
            U_pert[i, j] += eps
            obj_plus = RhapsodieDirect.PolarimetricMap("stokes", obj.I, obj.Q, U_pert)
            loss_plus = compute_loss(obj_plus, model, y_ref)
            U_grad[i, j] = (loss_plus - loss_base) / eps
        end
    end
    
    return RhapsodieDirect.PolarimetricMap("stokes", I_grad, Q_grad, U_grad)
end

# Calcul des gradients (uniquement sur quelques pixels pour la vitesse)
println("Calcul gradients (peut prendre du temps)...")
grad_analytical = analytical_gradient(test_object, H_noblur, y_true)
grad_numerical = numerical_gradient(test_object, H_noblur, y_true)

# Différences
diff_I = grad_analytical.I - grad_numerical.I
diff_Q = grad_analytical.Q - grad_numerical.Q
diff_U = grad_analytical.U - grad_numerical.U

# Plots
g1 = heatmap(grad_analytical.I, title="Grad Analytique I", aspect_ratio=:equal, color=:RdBu)
g2 = heatmap(grad_numerical.I, title="Grad Numérique I", aspect_ratio=:equal, color=:RdBu)
g3 = heatmap(diff_I, title="Différence I", aspect_ratio=:equal, color=:RdBu)

g4 = heatmap(grad_analytical.Q, title="Grad Analytique Q", aspect_ratio=:equal, color=:RdBu)
g5 = heatmap(grad_numerical.Q, title="Grad Numérique Q", aspect_ratio=:equal, color=:RdBu)
g6 = heatmap(diff_Q, title="Différence Q", aspect_ratio=:equal, color=:RdBu)

g7 = heatmap(grad_analytical.U, title="Grad Analytique U", aspect_ratio=:equal, color=:RdBu)
g8 = heatmap(grad_numerical.U, title="Grad Numérique U", aspect_ratio=:equal, color=:RdBu)
g9 = heatmap(diff_U, title="Différence U", aspect_ratio=:equal, color=:RdBu)

plot_gradients = plot(g1, g2, g3, g4, g5, g6, g7, g8, g9, layout=(3,3), size=(1200,1200))
savefig(plot_gradients, joinpath(output_folder, "test_gradients.png"))

# Stats
println("Erreur relative gradients:")
println("I: $(maximum(abs.(diff_I)) / maximum(abs.(grad_analytical.I)))")
println("Q: $(maximum(abs.(diff_Q)) / maximum(abs.(grad_analytical.Q)))")
println("U: $(maximum(abs.(diff_U)) / maximum(abs.(grad_analytical.U)))")


# --- 7bis. Descente de gradient CORRIGÉE ---
println("Test descente de gradient avec bon learning rate...")

obj_current = deepcopy(obj_init)
lr = 0.01  
n_iter = 10000
losses = Float64[]

for i in 1:n_iter
    y_pred = H_noblur * obj_current
    loss = 0.5 * sum((y_pred - y_true).^2)
    grad = H_noblur' * (y_pred - y_true)
    
    obj_current = RhapsodieDirect.PolarimetricMap("stokes",
        obj_current.I - lr * grad.I,
        obj_current.Q - lr * grad.Q, 
        obj_current.U - lr * grad.U)
    
    push!(losses, loss)
    
    if i % 40 == 0
        println("Iter $i: loss = $(loss)")
        println("  Obj I range: $(minimum(obj_current.I)) to $(maximum(obj_current.I))")
    end
end

# Mêmes plots qu'avant
p1 = plot(losses, title="Convergence (LR corrigé)", xlabel="Itération", ylabel="Loss", lw=2, yscale=:log10)
p2 = heatmap(test_object.I, title="I Original", aspect_ratio=:equal, color=:viridis)
p3 = heatmap(obj_current.I, title="I Reconstruit", aspect_ratio=:equal, color=:viridis)
p4 = heatmap(test_object.Q, title="Q Original", aspect_ratio=:equal, color=:viridis)  
p5 = heatmap(obj_current.Q, title="Q Reconstruit", aspect_ratio=:equal, color=:viridis)
p6 = heatmap(test_object.U, title="U Original", aspect_ratio=:equal, color=:viridis)
p7 = heatmap(obj_current.U, title="U Reconstruit", aspect_ratio=:equal, color=:viridis)

plot_gd_fixed = plot(p1, p2, p3, p4, p5, p6, p7, layout=(3,3), size=(1200,900))
savefig(plot_gd_fixed, joinpath(output_folder, "gradient_descent_fixed.png"))