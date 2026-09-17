# src/SPHERE_IO.jl

# Parameters.txt layout (checked against HD109562, AB Aurigae and RY Lup):
#
#   line 1                 dim_y
#   line 2                 n_frames (NTOT)
#   line 3                 ndit      (frames per HWP position)
#   line 4                 n_angles
#   lines 5-6              star centre,             written (x, y) = (column, row)
#   lines 7 .. 6+n_angles  field rotation angles, in DEGREES, one per frame
#   lines 7+n, 8+n         right-channel offset,    written (x, y) = (column, row)
#
# The last two values are NOT a true-north offset and a plate scale, despite the
# field names this struct used before: Rhapsodie.jl (test_separable_reconstruction.jl:22)
# and PADI (PADI_reconstruction.jl:35) both read them as `Epsilon = ([0,0], par[end-1:end])`,
# the offset of the right IRDIS channel. A plate scale would not vary from 9.65 to
# 10.75 across three datasets of the same instrument; a channel offset does.
#
# Two conversions are applied here, so that SphereMetadata is entirely in Julia
# array order (dim1, dim2) = (row, column) with angles in radians, which is what
# every transform downstream expects:
#
#   1. centre and epsilon are swapped from the file's (x, y) to (row, column);
#   2. epsilon is negated, and angles are converted with deg2rad.
#
# The epsilon convention was measured on the data rather than inferred, by
# cross-correlating the two channels of a frame and by taking the centroid of the
# coronagraphic mask in the weight map:
#
#   HD109562   measured right-left offset  (-9.77, +2.22) and (-9.61, +2.40) px
#              file epsilon                (-2.485, +9.646)
#   AB Aur     measured right-left offset  (-11.06, +1.10) px
#              file epsilon                (-1.205, +10.749)
#
# so epsilon_right = (-file[end], -file[end-1]). Note that a plain negation without
# the swap, which is what RhapsodieDirect/examples/generate_data.jl does for the
# simulated demo file, puts the 10-pixel offset on the wrong axis for these files:
# the demo Parameters.txt stores epsilon in the opposite component order.
#
# The angle SIGN is deliberately not applied here -- see `angle_sign` in
# init_sphere_leakage.
struct SphereMetadata
    dim_y::Int
    n_frames::Int
    ndit::Int
    n_angles::Int
    center::NTuple{2,Float64}         # (row, column), Julia array order
    rot_angles::Vector{Float64}       # radians, one per frame, unsigned
    epsilon_right::NTuple{2,Float64}  # (row, column) offset of the right channel
end

# Reader function with your default path
function read_sphere_parameters(filepath::String="/scratch/qvillegas/astro_data/Ab Aurigae/Parameters.txt")
    println("Reading: ", filepath)

    # Drop blank lines, including trailing ones, before indexing by line number.
    lines = filter(!isempty, strip.(readlines(filepath)))
    if length(lines) < 9
        error("Parameters file seems too short: ", filepath)
    end

    dim_y    = Int(parse(Float64, lines[1]))
    n_frames = Int(parse(Float64, lines[2]))
    ndit     = Int(parse(Float64, lines[3]))
    n_angles = Int(parse(Float64, lines[4]))

    expected = 6 + n_angles + 2
    if length(lines) != expected
        error("Parameters file has $(length(lines)) non-empty lines, expected " *
              "$expected (= 6 + n_angles + 2) with n_angles=$n_angles: $filepath")
    end
    if n_angles != n_frames
        @warn "n_angles ($n_angles) differs from n_frames ($n_frames): this file does not provide one field angle per frame" filepath
    end

    # File stores (x, y) = (column, row); swap into Julia array order.
    center = (parse(Float64, lines[6]), parse(Float64, lines[5]))

    rot_angles = [deg2rad(parse(Float64, lines[6 + i])) for i in 1:n_angles]

    # Swap into (row, column) and negate: see the comment block above.
    epsilon_right = (-parse(Float64, lines[end]), -parse(Float64, lines[end-1]))

    return SphereMetadata(dim_y, n_frames, ndit, n_angles, center, rot_angles, epsilon_right)
end

# Generic reader for delimited matrix files (space or tab-separated)
function read_matrix_file(filepath::String)
    """
    Generically read any text file with rows of space/tab-separated values.
    Automatically detects dimensions from file content.

    Returns:
        - matrix::Matrix{Float64} with shape (n_rows, n_cols)
        - n_rows::Int
        - n_cols::Int
    """
    lines = readlines(filepath)
    non_empty_lines = [line for line in lines if !isempty(strip(line))]
    n_rows = length(non_empty_lines)
    if n_rows == 0
        error("Empty matrix file: ", filepath)
    end

    first_row_values = split(strip(non_empty_lines[1]))
    n_cols = length(first_row_values)
    matrix = zeros(Float64, n_rows, n_cols)

    for i in 1:n_rows
        values = split(strip(non_empty_lines[i]))
        if length(values) != n_cols
            @warn "Row $i has $(length(values)) values, expected $n_cols. Truncating/ignoring extras."
        end
        for j in 1:min(length(values), n_cols)
            matrix[i, j] = parse(Float64, values[j])
        end
    end

    return matrix, n_rows, n_cols
end

# Utility function to expand array by NDIT (duplicate each row ndit times)
function expand_ndit(data::Matrix{Float64}, ndit::Int)
    n_rows = size(data, 1)
    n_cols = size(data, 2)
    expanded = zeros(Float64, n_rows * ndit, n_cols)
    for i in 1:n_rows
        for rep in 1:ndit
            expanded[(i-1)*ndit + rep, :] = data[i, :]
        end
    end
    return expanded
end

# Reader function for polarimetric coefficients with crosstalk
function read_crosstalk_coefficients(filepath::String="/scratch/qvillegas/astro_data/Ab Aurigae/instruments_values_with_crosstalk.txt"; ndit::Int=4)
    println("Reading coefficients from: ", filepath)
    raw_coeffs, n_hwp, n_cols = read_matrix_file(filepath)
    println("Detected: $n_hwp rows × $n_cols columns")
    expanded_coeffs = expand_ndit(raw_coeffs, ndit)
    println("Expanded to: $(size(expanded_coeffs, 1)) rows × $(size(expanded_coeffs, 2)) columns")
    return expanded_coeffs, n_hwp, n_cols
end
