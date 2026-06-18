# test_init_sphere_leakage.jl
# Exploration pas à pas de init_sphere_leakage et de ses sorties.
# Chaque section "## N." est exécutable indépendamment depuis VS Code (Shift+Entrée sur la cellule).

using compgrad_Rhapsodie
using EasyFITS          # pour readfits si besoin d'inspecter les FITS bruts
using DelimitedFiles    # pour readdlm

DATA_FOLDER = "/scratch/qvillegas/astro_data/HD109562"

## 1. Appel de la fonction complète
D, Dstar, S_star_I, AS_noblur, nAS = init_sphere_leakage(DATA_FOLDER; verbose=true)

## 2. Vérification des formes de sortie
println("=== Sorties de init_sphere_leakage ===")
println("D.data       : ", size(D.data),    " (attendu : (256, 512, 128))")
println("S_star_I     : ", size(S_star_I),  " (attendu : (256, 256))")
println("AS_noblur    : ", size(AS_noblur), " (attendu : (256, 512, 128))")
println("nAS          : ", nAS, " (scalaire positif)")
println("typeof(D)    : ", typeof(D))
println("typeof(Dstar): ", typeof(Dstar))

## 3. Exploration du Dataset D — données et poids
println("=== Dataset D ===")
println("D.data min/max  : ", minimum(D.data), " / ", maximum(D.data))
println("D.data NaN ?    : ", any(isnan.(D.data)))

# Les poids sont portés par le modèle de bruit
noise_model = D.noise_model
println("type noise_model: ", typeof(noise_model))

## 4. Poids W (inverse variance) — vérification
# Selon le type de noise_model, les poids sont accessibles différemment
if hasproperty(noise_model, :weights)
    W = noise_model.weights
    println("=== Poids W ===")
    println("size(W)      : ", size(W))
    println("W min/max    : ", minimum(W), " / ", maximum(W))
    println("W zeros ?    : ", sum(W .== 0), " pixels nuls sur ", prod(size(W)))
else
    println("Pas de champ .weights direct sur ", typeof(noise_model))
end

## 5. Modèles directs H et H_noblur — via les deux Datasets
H        = D.direct_model
H_noblur = Dstar.direct_model
println("=== DirectModel H (avec flou) ===")
println("typeof(H)       : ", typeof(H))
println("=== DirectModel H_noblur (sans flou) ===")
println("typeof(H_noblur): ", typeof(H_noblur))

## 6. PSF étoile — terme de fuite
println("=== PSF étoile S_star_I ===")
println("size          : ", size(S_star_I))
println("min / max     : ", minimum(S_star_I), " / ", maximum(S_star_I))
println("somme totale  : ", sum(S_star_I))
# Trouver le centre de la PSF (pixel de valeur max)
idx_max = argmax(S_star_I)
println("pixel max à   : row=", idx_max[1], " col=", idx_max[2])

## 7. Terme A'*s (réponse de l'étoile sans flou dans l'espace données)
println("=== AS_noblur ===")
println("size          : ", size(AS_noblur))
println("min / max     : ", minimum(AS_noblur), " / ", maximum(AS_noblur))
println("nAS = ||A'*s||²_W : ", nAS)

## 8. Test d'un appel gradient (sur x = zéros)
# comp_grad attend le format Julia : (H, W, 3) — canaux en dernière dimension
# Il retourne (gradient, chi2)
# NOTE : comp_grad accède à D.weights_op qui n'existe plus dans l'API actuelle
# de RhapsodieDirect (le champ s'appelle maintenant D.noise_model).
# Ce bug pré-existant est visible ici ; il faudra mettre à jour comp_grad.
println("=== Gradient sur x = 0 ===")
x0 = zeros(Float64, 256, 256, 3)   # format Julia : (H, W, channels)
println("Champs disponibles sur D : ", fieldnames(typeof(D)))
try
    grad, chi2 = comp_grad(x0, D)
    println("size(grad)    : ", size(grad))
    println("grad min/max  : ", minimum(grad), " / ", maximum(grad))
    println("chi2          : ", chi2)
catch e
    println("ERREUR attendue (API Dataset) : ", e)
end

## 9. Test du gradient avec terme de fuite
println("=== Gradient avec terme de fuite (alpha_s=1e-3) ===")
alpha_s = 1e-3
try
    grad_leak, chi2_leak = comp_grad_disk_scalar_leakage(x0, alpha_s, AS_noblur, D)
    println("size(grad_leak)  : ", size(grad_leak))
    println("grad_leak min/max: ", minimum(grad_leak), " / ", maximum(grad_leak))
    println("chi2_leak        : ", chi2_leak)
catch e
    println("ERREUR attendue (API Dataset) : ", e)
end

## 10. Étapes intermédiaires — reconstruction manuelle du setup
# (utile pour le debugger : on refait les étapes de init_sphere_leakage morceau par morceau)
println("=== Reconstruction manuelle ===")

meta = read_sphere_parameters(DATA_FOLDER * "/Parameters.txt")
println("dim_y=$(meta.dim_y), n_frames=$(meta.n_frames), ndit=$(meta.ndit)")

coeffs_matrix, n_hwp, _ = read_crosstalk_coefficients(
    DATA_FOLDER * "/instruments_values_with_crosstalk.txt"; ndit=meta.ndit)
println("coeffs shape : ", size(coeffs_matrix))
println("trame 1 cam gauche (I,Q,U) : ", coeffs_matrix[1, 1:3])
println("trame 1 cam droite (I,Q,U) : ", coeffs_matrix[1, 4:6])

dithering = readdlm(DATA_FOLDER * "/Ditering.txt")
println("dithering : ", dithering)

psf_raw  = readfits(DATA_FOLDER * "/PSF_parametered.fits")
psf_star = readfits(DATA_FOLDER * "/PSF_estimated.fits")
println("PSF blur shape  : ", size(psf_raw), " → kernel ", size(psf_raw, 1) > size(psf_raw, 2) ? "premier demi" : "entier")
println("PSF étoile shape: ", size(psf_star))
println("rot_angles[1:4] : ", meta.rot_angles[1:4])
