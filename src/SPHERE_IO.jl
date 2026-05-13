# src/SPHERE_IO.jl

# Metadata structure matching your specifications
struct SphereMetadata
    dim_y::Int
    n_frames::Int
    ndit::Int
    n_angles::Int
    center::Tuple{Float64, Float64}  # Tuple (x, y)
    rot_angles::Vector{Float64}      # 1D angle array
    true_north_offset::Float64
    plate_scale::Float64
end

# Reader function with your default path
function read_sphere_parameters(filepath::String="/scratch/qvillegas/astro_data/Ab Aurigae/Parameters.txt")
    println("Reading: ", filepath)

    lines = readlines(filepath)
    # Basic validation
    if length(lines) < 8
        error("Parameters file seems too short: ", filepath)
    end

    dim_y    = Int(parse(Float64, strip(lines[1])))
    n_frames = Int(parse(Float64, strip(lines[2])))
    ndit     = Int(parse(Float64, strip(lines[3])))
    n_angles = Int(parse(Float64, strip(lines[4])))

    center_x = parse(Float64, strip(lines[5]))
    center_y = parse(Float64, strip(lines[6]))
    center = (center_x, center_y)

    rot_angles = Float64[]
    for i in 1:n_angles
        angle = parse(Float64, strip(lines[6 + i]))
        push!(rot_angles, angle)
    end

    true_north_offset = parse(Float64, strip(lines[7 + n_angles]))
    plate_scale       = parse(Float64, strip(lines[8 + n_angles]))

    return SphereMetadata(dim_y, n_frames, ndit, n_angles, center, rot_angles, true_north_offset, plate_scale)
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
