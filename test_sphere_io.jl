# test_sphere_io.jl
# Sandbox script to test the SPHERE I/O parser 
# and verify that all variables are initialized correctly.

using compgrad_Rhapsodie

println("--- SANDBOX TEST START ---")

## 1. Read Parameters.txt
parameters = read_sphere_parameters()

println("\n--- PARAMETERS.txt VERIFICATION ---")
println("- Image Dimension Y (l1) : ", parameters.dim_y)
println("- Total frames (l2) : ", parameters.n_frames)
println("- NDIT factor (l3) : ", parameters.ndit)
println("- Total angles read (l4) : ", parameters.n_angles)
println("- Star center (l5, l6), swapped to (row, col): $(parameters.center)")

println("\n- Sample of the angles (l7 to l(6+n_angles)), converted to radians:")
println("    First 3 : ", parameters.rot_angles[1:3])
println("    Last 3 : ", parameters.rot_angles[end-2:end])
println("    First 3, back in degrees : ", rad2deg.(parameters.rot_angles[1:3]))

# Last two lines: right-channel offset, swapped to (row, col) and negated.
# Expected for AB Aurigae: (-10.7492, 1.2050); measured on the data: (-11.06, 1.10).
println("\n- Right-channel offset (last two lines) : ", parameters.epsilon_right)

## 2. Read Crosstalk coefficients (now generic)
coefficients, n_hwp, n_cols = read_crosstalk_coefficients(ndit=parameters.ndit)

println("\n--- CROSSTALK COEFFICIENTS VERIFICATION ---")
println("- Number of HWP positions : ", n_hwp)
println("- Number of columns : ", n_cols)
println("- Expanded shape : ", size(coefficients))
println("- Expected shape : ($(parameters.n_frames), $n_cols)")

if size(coefficients, 1) == parameters.n_frames
    println("✓ Shape matches frames count!")
else
    println("✗ Shape mismatch!")
end

println("\n- Sample coefficients (first 3 positions, all columns):")
println("    Position 1: ", coefficients[1, :])
println("    Position 2: ", coefficients[2, :])
println("    Position 3: ", coefficients[3, :])

println("\n- After NDIT expansion (rows 5-6 should repeat row 1 if NDIT=4):")
println("    Frame 5: ", coefficients[5, :])
println("    Frame 6: ", coefficients[6, :])

println("\n--- SANDBOX TEST COMPLETE ---")

