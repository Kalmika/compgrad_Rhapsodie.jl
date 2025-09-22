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

data_folder = "data"
output_folder = "output" 
path_disk = "/home/qvillegas/WORK/Fork/compgrad_Rhapsodie.jl/data/sample_for_rhapsodie_128x128.h5"
path_star = "/home/qvillegas/WORK/Fork/compgrad_Rhapsodie.jl/data/star_cropped.fits"
ker = CatmullRomSpline(Float64, Flat)
my_alpha = 1/35

# --- 1. Chargement des données et paramètres ---
println("Chargement des données...")
object_params=ObjectParameters((128,128),(64.,64.))

dset = h5open(path_disk, "r")
I_disk = read(dset["disk_data"])[:,:,1]
Ip_disk = read(dset["disk_data"])[:,:,2]
θ_disk = read(dset["disk_theta"])
close(dset)

STAR_intensity = readfits(path_star)
coef_disk = my_alpha*maximum(STAR_intensity)/maximum(I_disk)

# TODO : change after debbugging
# center_x, center_y = round.(Int, object_params.center)
# half_size = 20
# y_range = (center_y - half_size):(center_y + half_size)
# x_range = (center_x - half_size):(center_x + half_size)
# I_map = zeros(object_params.size)
# I_map[y_range, x_range] .= maximum(STAR_intensity)  # Point source simple
# Q_map = zeros(object_params.size)
# U_map = zeros(object_params.size)
# S_star = PolarimetricMap("intensities", I_map, Q_map, U_map) 

# Objet ciel complet (disque + étoile) pour la simulation
S_disk = PolarimetricMap("intensities", coef_disk*(I_disk - Ip_disk), coef_disk*Ip_disk, θ_disk) 
S_star = PolarimetricMap("intensities",  STAR_intensity, zero(STAR_intensity), zero(STAR_intensity)) 

# --- 2. Construction des modèles directs (A et A') ---
println("Construction des modèles directs2...")
data_params=DatasetParameters((128,256), 4, 1, 1, (64.,64.))
indices=get_indices_table(data_params)
polar_params=set_default_polarisation_coefficients(indices)

field_params=FieldTransformParameters[]
for i=1:data_params.frames_total
    push!(field_params, FieldTransformParameters(ker, 0., (0.,0.), (-10.7365 , 1.39344), polar_params[i][1], polar_params[i][2]))
end
field_transforms = load_field_transforms(object_params, data_params, field_params)

psf_center = readdlm("/home/qvillegas/WORK/Fork/compgrad_Rhapsodie.jl/data/PSF_centers_Airy.txt");
psf = readfits("/home/qvillegas/WORK/Fork/compgrad_Rhapsodie.jl/data/PSF_parametered_Airy.fits");
blur = set_fft_operator(object_params,(psf[1:end÷2,:]'), psf_center[1:2])[1];

# H = Modèle COMPLET (A), avec flou. Pour la simulation et les gradients.
H = DirectModel(size(S_disk), (128,256,4), S_disk.parameter_type, field_transforms, blur)

# H_noblur = Modèle SANS flou (A'). Pour le terme de fuite et nAS.
H_noblur = DirectModel(size(S_disk), (128,256,4), S_disk.parameter_type, field_transforms)

# --- 3. Simulation des données observées ---
println("Simulation des données observées2...")
BadPixMap = Float64.(rand(0.0:1e-16:1.0, data_params.size) .< 0.9);
data, weight = data_simulator_dual_component_bis(BadPixMap, field_transforms, S_disk, S_star; A_disk=blur);

# Le Dataset contient le modèle COMPLET H (A)
D = Dataset(data, weight, H)
Dstar = Dataset(data, weight, H_noblur)

# --- 4. Pré-calcul des termes liés à l'étoile (s) ---
println("Pré-calcul des termes de fuite stellaire...")

# Calcul de A'*s
AS_noblur = H_noblur * S_star

# Norme : ||A'*s||^2_W
# Note: On utilise ici la réponse de l'étoile NON pondérée par lambda.
# Si votre nAS doit inclure lambda, changez `AS_noblur` en `leakage_term` ci-dessous.
nAS = sum(D.weights_op .* abs2.(AS_noblur))

println("Initialisation terminée.!")
# On retourne D (avec H), l'intensité de l'étoile, le terme de fuite et nAS.
# --- 6. Plots des informations importantes ---
println("Génération des plots d'informations importantes...")


# Plots individuels des frames simulées
for i in 1:data_params.frames_total
    p = heatmap(data[:,:,i], title="Données simulées (frame $i)", c=:viridis, aspect_ratio=1)
    display(p)  # <-- Affiche le plot à l'écran
    savefig(p, joinpath(output_folder, "simulation_frame_$(i).png"))
end

# Plot combiné des 4 frames simulées
plots_frames = [heatmap(data[:,:,i], title="Frame $i", c=:viridis, aspect_ratio=1) for i in 1:data_params.frames_total]
plot_frames_summary = plot(plots_frames..., layout=(2,2), size=(800,800), title="Frames simulées")
savefig(plot_frames_summary, joinpath(output_folder, "simulation_frames_summary.png"))

# Plot 2: Objet disque original (intensité)
p2 = heatmap(S_disk.I, title="Disque original (I)", c=:viridis, aspect_ratio=1)

# Plot 3: Étoile
p3 = heatmap(S_star.I, title="Étoile", c=:viridis, aspect_ratio=1)

# Plot 4: Terme de fuite A'*s
p4 = heatmap(AS_noblur[:,:,1], title="Terme de fuite A'*s", c=:viridis, aspect_ratio=1)

# Combinaison des plots (hors frames simulées)
plot_summary = plot(p2, p3, p4, layout=(1,3), size=(1200,400))
savefig(plot_summary, joinpath(output_folder, "simulation_summary.png"))

for i in 1:4
    p = heatmap(data[:,:,i], title="Données simulées (frame $i)", c=:viridis, aspect_ratio=1)
    display(p)
    savefig(p, joinpath(output_folder, "simulation_frame_$i.png"))
end

println("nAS = $(round(nAS, digits=2))")
println("Plot sauvegardé: simulation_summary.png")




# --- 8. DESCENTE DE GRADIENT ---
println("\n" * "="^60)
println("DÉMARRAGE DE LA DESCENTE DE GRADIENT")
println("="^60)

# Paramètres de l'optimisation
learning_rate = 5.0
alpha_s = 1.0  # Coefficient de fuite stellaire

# --- 8.1. Initialisation des paramètres (image de départ : zéros) ---
println("Initialisation des paramètres à zéro...")
x_init = zeros(Float64, 128, 128, 3)  # [I_unpolarized, I_polarized, theta]

# Créer un objet PolarimetricMap initial pour visualisation
S_init = PolarimetricMap("intensities", x_init[:, :, 1] - x_init[:, :, 2], x_init[:, :, 2], x_init[:, :, 3])

# --- 8.2. Visualisation de l'état initial ---
p_init_I = heatmap(S_init.I, title="État initial - Intensité I", c=:viridis, aspect_ratio=1)
p_init_Q = heatmap(S_init.Q, title="État initial - Composante Q", c=:viridis, aspect_ratio=1)  
p_init_U = heatmap(S_init.U, title="État initial - Composante U", c=:viridis, aspect_ratio=1)
plot_init = plot(p_init_I, p_init_Q, p_init_U, layout=(1,3), size=(1500,400))
display(plot_init)
savefig(plot_init, joinpath(output_folder, "01_etat_initial.png"))

# --- 8.3. Calcul du gradient (première étape) ---
println("Calcul du gradient initial...")

# Convertir x en PolarimetricMap
S_disk_current = PolarimetricMap("intensities", x_init[:, :, 1] - x_init[:, :, 2], x_init[:, :, 2], x_init[:, :, 3])
g_current = copy(S_disk_current)

# 2. Calculer A*x (modèle AVEC flou)
println("  • Calcul de A*x...")
Ax_current = H * S_disk_current

# 3. Calculer le résidu : A*x + alpha_s*leakage - y
println("  • Calcul du résidu...")
alpha_s = 0.0
residual = Ax_current .+ alpha_s * AS_noblur .- data

# 5. Appliquer les poids
println("  • Application des poids...")
weighted_residual = weight .* residual

# 6. Calculer le chi-deux
println("  • Calcul du chi²...")
chi2_initial = dot(residual, weighted_residual)

# 7. Appliquer l'adjoint : A^T * (weighted_residual)
println("  • Application de l'adjoint A^T...")
apply!(g_current, H', weighted_residual)

println("Chi² initial : $(round(chi2_initial, digits=6))")

# --- 8.4. Visualisation du gradient ---
println("Visualisation du gradient...")
v_min_grad, v_max_grad = extrema([extrema(g_current.I)..., extrema(g_current.Q)..., extrema(g_current.U)...])

p_grad_I = heatmap(g_current.I, title="Gradient - Intensité I", c=:viridis, aspect_ratio=1, 
                   clims=(v_min_grad, v_max_grad))
p_grad_Q = heatmap(g_current.Q, title="Gradient - Composante Q", c=:viridis, aspect_ratio=1,
                   clims=(v_min_grad, v_max_grad))
p_grad_U = heatmap(g_current.U, title="Gradient - Composante U", c=:viridis, aspect_ratio=1,
                   clims=(v_min_grad, v_max_grad))
plot_gradient = plot(p_grad_I, p_grad_Q, p_grad_U, layout=(1,3), size=(1500,400))
display(plot_gradient)
savefig(plot_gradient, joinpath(output_folder, "02_premier_gradient.png"))

# --- 8.5. Visualisation des résidus ---
println("Visualisation des résidus...")
plots_residuals = []
for i in 1:4
    p_res = heatmap(residual[:,:,i], title="Résidu frame $i", c=:viridis, aspect_ratio=1)
    push!(plots_residuals, p_res)
end
plot_residuals = plot(plots_residuals..., layout=(2,2), size=(1000,800))
display(plot_residuals)
savefig(plot_residuals, joinpath(output_folder, "03_residus_initiaux.png"))

# --- 8.6. Conversion du gradient au format array et étape de descente ---
println("Conversion du gradient et étape de descente...")

# Fonction helper pour convertir PolarimetricMap vers array
function polarimetric_to_array(pm::PolarimetricMap)
    result = zeros(Float64, 128, 128, 3)
    result[:, :, 1] = pm.I + pm.Ip  # I_unpolarized = I + I_p
    result[:, :, 2] = pm.Ip         # I_polarized = I_p  
    result[:, :, 3] = pm.θ          # theta
    return result
end

# Convertir le gradient
grad_array = polarimetric_to_array(g_current)

# Visualisation du gradient sous forme de array
p_grad_chan1 = heatmap(grad_array[:,:,1], title="Gradient Canal 1 (I_u)", c=:viridis, aspect_ratio=1)
p_grad_chan2 = heatmap(grad_array[:,:,2], title="Gradient Canal 2 (I_p)", c=:viridis, aspect_ratio=1)  
p_grad_chan3 = heatmap(grad_array[:,:,3], title="Gradient Canal 3 (θ)", c=:viridis, aspect_ratio=1)
plot_grad_array = plot(p_grad_chan1, p_grad_chan2, p_grad_chan3, layout=(1,3), size=(1500,400))
display(plot_grad_array)
savefig(plot_grad_array, joinpath(output_folder, "04_gradient_array_format.png"))

# Une étape de descente de gradient
x_updated = x_init - learning_rate * grad_array
println("Étape de descente effectuée avec learning_rate = $learning_rate")

# --- 8.7. Visualisation après descente de gradient ---
println("Visualisation après une étape de descente...")

# Créer le PolarimetricMap mis à jour
S_updated = PolarimetricMap("intensities", x_updated[:, :, 1] - x_updated[:, :, 2], x_updated[:, :, 2], x_updated[:, :, 3])

# Plots après mise à jour
p_up_I = heatmap(S_updated.I, title="Après descente - Intensité I", c=:viridis, aspect_ratio=1)
p_up_Q = heatmap(S_updated.Q, title="Après descente - Composante Q", c=:viridis, aspect_ratio=1)
p_up_U = heatmap(S_updated.U, title="Après descente - Composante U", c=:viridis, aspect_ratio=1)
plot_updated = plot(p_up_I, p_up_Q, p_up_U, layout=(1,3), size=(1500,400))
display(plot_updated)
savefig(plot_updated, joinpath(output_folder, "05_apres_descente.png"))

# --- 8.8. Nouveau calcul du chi² pour vérifier l'amélioration ---
println("Vérification de l'amélioration...")

# Recalculer le modèle direct avec les nouveaux paramètres
Ax_updated = H * S_updated
residual_updated = Ax_updated .+ alpha_s * AS_noblur .- data
weighted_residual_updated = weight .* residual_updated
chi2_updated = dot(residual_updated, weighted_residual_updated)

println("Chi² initial  : $(round(chi2_initial, digits=6))")
println("Chi² après GD : $(round(chi2_updated, digits=6))")
println("Amélioration  : $(round(chi2_initial - chi2_updated, digits=6))")

# --- 8.9. Comparaison avant/après ---
println("Génération du plot de comparaison...")

# Comparaison intensité I
p_comp_I_before = heatmap(S_init.I, title="AVANT - Intensité I", c=:viridis, aspect_ratio=1, clims=(minimum(S_updated.I), maximum(S_updated.I)))
p_comp_I_after = heatmap(S_updated.I, title="APRÈS - Intensité I", c=:viridis, aspect_ratio=1, clims=(minimum(S_updated.I), maximum(S_updated.I)))
p_comp_diff_I = heatmap(S_updated.I - S_init.I, title="DIFFÉRENCE - I", c=:viridis, aspect_ratio=1)

plot_comparison = plot(p_comp_I_before, p_comp_I_after, p_comp_diff_I, layout=(1,3), size=(1500,400))
display(plot_comparison)
savefig(plot_comparison, joinpath(output_folder, "06_comparaison_avant_apres.png"))

# --- 8.10. Analyse des statistiques ---
println("\n" * "="^60)
println("ANALYSE DES STATISTIQUES")
println("="^60)

println("Paramètres utilisés :")
println("  • Learning rate : $learning_rate")
println("  • Alpha_s (coef. fuite) : $alpha_s")
println("  • nAS : $(round(nAS, digits=2))")

println("\nStatistiques du gradient :")
println("  • Max gradient I : $(round(maximum(abs.(g_current.I)), digits=6))")
println("  • Max gradient Q : $(round(maximum(abs.(g_current.Q)), digits=6))")
println("  • Max gradient U : $(round(maximum(abs.(g_current.U)), digits=6))")

println("\nStatistiques des paramètres mis à jour :")
println("  • Max |I| : $(round(maximum(abs.(S_updated.I)), digits=6))")
println("  • Max |Q| : $(round(maximum(abs.(S_updated.Q)), digits=6))")
println("  • Max |U| : $(round(maximum(abs.(S_updated.U)), digits=6))")

println("\nChi² :")
println("  • Initial : $(round(chi2_initial, sigdigits=4))")  
println("  • Après 1 étape : $(round(chi2_updated, sigdigits=4))")
if chi2_updated < chi2_initial
    println("  ✓ Amélioration de $(round(100*(chi2_initial-chi2_updated)/chi2_initial, digits=2))%")
else
    println("  ✗ Pas d'amélioration (le learning rate est peut-être trop élevé)")
end

println("\n" * "="^60)
println("DESCENTE DE GRADIENT TERMINÉE")
println("7 images sauvegardées dans le dossier '$output_folder'")
println("="^60)









































# Fonction d'identité pour les field transforms
function load_identity_field_transforms2(object::RhapsodieDirect.ObjectParameters,
                                      data::RhapsodieDirect.DatasetParameters)
    
    field_transforms = Vector{RhapsodieDirect.FieldTransformOperator{Float64}}()
    
    for k = 1:data.frames_total
        # Coefficients de polarisation identité (matrice identité 3x3)
        polarization_identity = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        
        # Créer un FieldTransformOperator identité sans interpolation
        push!(field_transforms, RhapsodieDirect.FieldTransformOperator((object.size[1], object.size[2], 3),
                                                        data.size,
                                                        polarization_identity,
                                                        polarization_identity,
                                                        LazyAlgebra.Id,  # Identity mapping
                                                        LazyAlgebra.Id)) 
    end
    
    return field_transforms
end

println("Lancement du test complet de l'effet du flou sur un objet POLARISÉ...")

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

# --- Modèles avec transformations identité ---
println("Construction des modèles avec transformations identité...")
# field_transforms_identity = load_identity_field_transforms(object_params, data_params)
# H_identity = RhapsodieDirect.DirectModel(size(test_object), (128, 256, 4), test_object.parameter_type, field_transforms_identity, blur)
# H_identity_noblur = RhapsodieDirect.DirectModel(size(test_object), (128, 256, 4), test_object.parameter_type, field_transforms_identity)

# --- 4. Application des modèles et reconstruction des cartes de Stokes ---
println("Application des modèles et reconstruction des cartes de Stokes...")

recons_blurred = H * test_object
recons_sharp = H_noblur * test_object

# Reconstructions avec transformations identité
# recons_identity_blurred = H_identity * test_object
# recons_identity_sharp = H_identity_noblur * test_object


v_min, v_max = extrema(recons_sharp[:,:,1])
plot(heatmap(recons_blurred[:,:,1], title="Carte I floutée", c=:viridis, aspect_ratio=1, clims=(v_min, v_max)),
     heatmap(recons_sharp[:,:,1], title="Carte I nette", c=:viridis, aspect_ratio=1, clims=(v_min, v_max)),
     layout=(1, 2), size=(1000, 500))

# --- 5. Fonction d'aide pour la visualisation ---
function create_comparison_plot(map_sharp, map_blurred, map_title, suptitle_text)
    map_diff = abs.(map_blurred .- map_sharp)
    v_min, v_max = extrema(map_sharp)
    
    plot1 = heatmap(map_sharp, title="$map_title SANS flou", c=:viridis, aspect_ratio=1, clims=(v_min, v_max))
    plot2 = heatmap(map_blurred, title="$map_title AVEC flou", c=:viridis, aspect_ratio=1, clims=(v_min, v_max))
    plot3 = heatmap(map_diff, title="Différence Absolue", c=:inferno, aspect_ratio=1)
    
    return plot(plot1, plot2, plot3, layout=(1, 3), size=(1500, 500), suptitle=suptitle_text)
end

# Fonction pour comparer les transformations (interpolation vs identité)
function create_transform_comparison_plot(map_interp_sharp, map_interp_blurred, map_identity_sharp, map_identity_blurred, map_title, suptitle_text)
    v_min = min(minimum(map_interp_sharp), minimum(map_identity_sharp))
    v_max = max(maximum(map_interp_sharp), maximum(map_identity_sharp))
    
    plot1 = heatmap(map_interp_sharp, title="$map_title Interpolation SANS flou", c=:viridis, aspect_ratio=1, clims=(v_min, v_max))
    plot2 = heatmap(map_interp_blurred, title="$map_title Interpolation AVEC flou", c=:viridis, aspect_ratio=1, clims=(v_min, v_max))
    plot3 = heatmap(map_identity_sharp, title="$map_title Identité SANS flou", c=:viridis, aspect_ratio=1, clims=(v_min, v_max))
    plot4 = heatmap(map_identity_blurred, title="$map_title Identité AVEC flou", c=:viridis, aspect_ratio=1, clims=(v_min, v_max))
    
    return plot(plot1, plot2, plot3, plot4, layout=(2, 2), size=(1000, 1000), suptitle=suptitle_text)
end

# --- 6. Génération et sauvegarde des graphiques ---

println("Génération des graphiques de comparaison...")

# Comparaison sur la composante I (Intensité totale)
plot_i = create_comparison_plot(recons_sharp.I, recons_blurred.I, "Carte I", "Effet du flou sur l'Intensité Totale (I)")
savefig(plot_i, joinpath(output_folder, "blur_effect_on_I.png")) # <--- MODIFICATION

# Comparaison sur la composante Q
plot_q = create_comparison_plot(recons_sharp.Q, recons_blurred.Q, "Carte Q", "Effet du flou sur la Composante Q")
savefig(plot_q, joinpath(output_folder, "blur_effect_on_Q.png")) # <--- MODIFICATION

# Comparaison sur la composante U
plot_u = create_comparison_plot(recons_sharp.U, recons_blurred.U, "Carte U", "Effet du flou sur la Composante U")
savefig(plot_u, joinpath(output_folder, "blur_effect_on_U.png")) # <--- MODIFICATION

# Comparaison sur l'intensité polarisée (Ip)
ip_sharp = recons_sharp.Ip
ip_blurred = recons_blurred.Ip
plot_ip = create_comparison_plot(ip_sharp, ip_blurred, "Carte Ip", "Effet du flou sur l'Intensité Polarisée (Ip)")
savefig(plot_ip, joinpath(output_folder, "blur_effect_on_Ip.png")) # <--- MODIFICATION

# --- 7. Comparaison des transformations (Interpolation vs Identité) ---
println("Génération des graphiques de comparaison des transformations...")

# Comparaison sur la composante I
plot_transform_i = create_transform_comparison_plot(recons_sharp.I, recons_blurred.I, 
                                                   recons_identity_sharp.I, recons_identity_blurred.I,
                                                   "Carte I", "Comparaison Transformations: Interpolation vs Identité (Composante I)")
savefig(plot_transform_i, joinpath(output_folder, "transform_comparison_I.png"))

# Comparaison sur la composante Q
plot_transform_q = create_transform_comparison_plot(recons_sharp.Q, recons_blurred.Q, 
                                                   recons_identity_sharp.Q, recons_identity_blurred.Q,
                                                   "Carte Q", "Comparaison Transformations: Interpolation vs Identité (Composante Q)")
savefig(plot_transform_q, joinpath(output_folder, "transform_comparison_Q.png"))

# Comparaison sur la composante U
plot_transform_u = create_transform_comparison_plot(recons_sharp.U, recons_blurred.U, 
                                                   recons_identity_sharp.U, recons_identity_blurred.U,
                                                   "Carte U", "Comparaison Transformations: Interpolation vs Identité (Composante U)")
savefig(plot_transform_u, joinpath(output_folder, "transform_comparison_U.png"))

# Comparaison sur l'intensité polarisée (Ip)
plot_transform_ip = create_transform_comparison_plot(ip_sharp, ip_blurred, 
                                                    recons_identity_sharp.Ip, recons_identity_blurred.Ip,
                                                    "Carte Ip", "Comparaison Transformations: Interpolation vs Identité (Intensité Polarisée)")
savefig(plot_transform_ip, joinpath(output_folder, "transform_comparison_Ip.png"))

println("\nTest terminé !")
println("8 graphiques de comparaison ont été sauvegardés dans le dossier '$output_folder'.")
println("- 4 graphiques d'effet du flou (blur_effect_*.png)")
println("- 4 graphiques de comparaison des transformations (transform_comparison_*.png)")