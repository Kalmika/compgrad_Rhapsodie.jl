#
# test_blur_on_polarized_object_full.jl
#
# Ce script teste l'effet du flou sur un objet synthétique polarisé et
# sauvegarde les résultats dans un dossier "output".
#

using Pkg
Pkg.activate(".") 

using RhapsodieDirect
using InterpolationKernels
using EasyFITS
using DelimitedFiles
using Plots
using Images

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

# --- 4. Application des modèles et reconstruction des cartes de Stokes ---
println("Application des modèles et reconstruction des cartes de Stokes...")
model_blurred_data = H * test_object
model_sharp_data = H_noblur * test_object

recons_blurred = H' * model_blurred_data
recons_sharp = H_noblur' * model_sharp_data

# --- 5. Fonction d'aide pour la visualisation ---
function create_comparison_plot(map_sharp, map_blurred, map_title, suptitle_text)
    map_diff = abs.(map_blurred .- map_sharp)
    v_min, v_max = extrema(map_sharp)
    
    plot1 = heatmap(map_sharp, title="$map_title SANS flou", c=:viridis, aspect_ratio=1, clims=(v_min, v_max))
    plot2 = heatmap(map_blurred, title="$map_title AVEC flou", c=:viridis, aspect_ratio=1, clims=(v_min, v_max))
    plot3 = heatmap(map_diff, title="Différence Absolue", c=:inferno, aspect_ratio=1)
    
    return plot(plot1, plot2, plot3, layout=(1, 3), size=(1500, 500), suptitle=suptitle_text)
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

println("\nTest terminé !")
println("4 graphiques de comparaison ont été sauvegardés dans le dossier '$output_folder'.")