"""
	BirkhoffBLMO

A bounded LMO for the Birkhoff polytope. This oracle computes an extreme point subject to  
node-specific bounds on the integer variables.
"""
struct BirkhoffLMO
    append_by_column::Bool
    dim::Int
    lower_bounds::Vector{Float64}
    upper_bounds::Vector{Float64}
    int_vars::Vector{Int}
    fixed_to_one_rows::Vector{Int}
    fixed_to_one_cols::Vector{Int}
    index_map_rows::Vector{Int}
    index_map_cols::Vector{Int}
    atol::Float64
    rtol::Float64
end

struct BosciaBirkhoffBLMO <: Boscia.BoundedLinearMinimizationOracle
    birkhoffBLMO::BirkhoffLMO
end

struct FWBirkhoffLMO <: FrankWolfe.LinearMinimizationOracle
    birkhoffLMO::BirkhoffLMO
end

# define the mixed-integer Birkhoff polytope
function BirkhoffBLMO(dim, int_vars; append_by_column=true, atol=1e-6, rtol=1e-3)
    birkhoffLMO = BirkhoffLMO(
        append_by_column,
        dim,
        fill(0.0, length(int_vars)),
        fill(1.0, length(int_vars)),
        int_vars,
        Int[],
        Int[],
        collect(1:dim),
        collect(1:dim),
        atol,
        rtol,
    )
    return BosciaBirkhoffBLMO(birkhoffLMO)
end

function BirkhoffBLMO(dim; append_by_column=true, use_Boscia=true, atol=1e-6, rtol=1e-3)
    # define the all-integer Birkhoff polytope
    if use_Boscia
        birkhoffLMO = BirkhoffLMO(
            append_by_column,
            dim,
            fill(0.0, length(int_vars)),
            fill(1.0, length(int_vars)),
            collect(1:(dim^2)),
            Int[],
            Int[],
            collect(1:dim),
            collect(1:dim),
            atol,
            rtol,
        )
        return BosciaBirkhoffBLMO(birkhoffLMO)
    else
        # define the continous Birkhoff polytope
        birkhoffLMO = BirkhoffLMO(
            append_by_column,
            dim,
            [],
            [],
            [],
            Int[],
            Int[],
            collect(1:dim),
            collect(1:dim),
            atol,
            rtol,
        )
        return FWBirkhoffLMO(birkhoffLMO)
    end

end

"""
Computes the extreme point given an direction d, the current lower and upper bounds on the integer variables, and the set of integer variables.
"""
_compute_extreme_point_impl(blmo::BirkhoffLMO, d; kwargs...) = begin
    n = blmo.dim

    # Precompute index mapping to avoid repeated `findfirst` calls,
    # which would be very costly inside the loop.
    if length(blmo.int_vars) !== n^2
        idx_map_ub = zeros(Int, n^2)
        @inbounds for (c_idx, var) in enumerate(blmo.int_vars)
            idx_map_ub[var] = c_idx
        end
    end

    if size(d, 2) == 1
        d = blmo.append_by_column ? reshape(d, (n, n)) : transpose(reshape(d, (n, n)))
    end

    fixed_to_one_rows = blmo.fixed_to_one_rows
    fixed_to_one_cols = blmo.fixed_to_one_cols
    index_map_rows = blmo.index_map_rows
    index_map_cols = blmo.index_map_cols
    int_vars = blmo.int_vars
    ub = blmo.upper_bounds

    nreduced = length(index_map_rows)
    type = typeof(d[1, 1])
    d2 = ones(Union{type,Missing}, nreduced, nreduced)
    for j in 1:nreduced
        col_orig = index_map_cols[j]
        for i in 1:nreduced
            row_orig = index_map_rows[i]
            if blmo.append_by_column
                orig_linear_idx = (col_orig - 1) * n + row_orig
            else
                orig_linear_idx = (row_orig - 1) * n + col_orig
            end
            # the problem can only be integer types,
            # either all-integer or mixed-integer.
            if orig_linear_idx in int_vars
                idx = length(int_vars) < n^2 ? idx_map_ub[orig_linear_idx] : orig_linear_idx
                # interdict arc when fixed to zero
                if ub[idx] <= eps()
                    if blmo.append_by_column
                        d2[i, j] = missing
                    else
                        d2[j, i] = missing
                    end
                else
                    if blmo.append_by_column
                        d2[i, j] = d[row_orig, col_orig]
                    else
                        d2[j, i] = d[col_orig, row_orig]
                    end
                end
            else
                if blmo.append_by_column
                    d2[i, j] = d[row_orig, col_orig]
                else
                    d2[j, i] = d[col_orig, row_orig]
                end
            end
        end
    end

    m = SparseArrays.spzeros(n, n)
    for (i, j) in zip(fixed_to_one_rows, fixed_to_one_cols)
        m[i, j] = 1
    end

    res_mat = Hungarian.munkres(d2)
    (rows, cols, vals) = SparseArrays.findnz(res_mat)
    @inbounds for i in eachindex(cols)
        m[index_map_rows[rows[i]], index_map_cols[cols[i]]] = (vals[i] == 2)
    end

    # if size(d, 2) == 1
    m = if blmo.append_by_column
        # Convert sparse matrix to sparse vector by columns
        I, J, V = SparseArrays.findnz(m)
        linear_indices = (J .- 1) .* n .+ I
        SparseArrays.sparsevec(linear_indices, V, n^2)
    else
        # Convert sparse matrix to sparse vector by rows (transpose first)
        mt = SparseArrays.sparse(LinearAlgebra.transpose(m))
        I, J, V = SparseArrays.findnz(mt)
        linear_indices = (J .- 1) .* n .+ I
        SparseArrays.sparsevec(linear_indices, V, n^2)
    end
    # end
    return m
end

"""
Computes the extreme point given an direction d, the current lower and upper bounds on the integer variables, and the set of integer variables.
"""
_compute_inface_extreme_point_impl(blmo::BirkhoffLMO, direction, x; kwargs...) = begin
    n = blmo.dim

    if size(direction, 2) == 1
        direction =
            blmo.append_by_column ? reshape(direction, (n, n)) :
            transpose(reshape(direction, (n, n)))

        x = blmo.append_by_column ? reshape(x, (n, n)) : transpose(reshape(x, (n, n)))
    end

    # Precompute index mapping to avoid repeated `findfirst` calls,
    # which would be very costly inside the loop.
    if length(blmo.int_vars) !== n^2
        idx_map_ub = zeros(Int, n^2)
        @inbounds for (c_idx, var) in enumerate(blmo.int_vars)
            idx_map_ub[var] = c_idx
        end
    end

    fixed_to_one_rows = copy(blmo.fixed_to_one_rows)
    fixed_to_one_cols = copy(blmo.fixed_to_one_cols)
    index_map_rows = copy(blmo.index_map_rows)
    index_map_cols = copy(blmo.index_map_cols)
    int_vars = blmo.int_vars
    ub = blmo.upper_bounds

    nreduced = length(blmo.index_map_rows)

    delete_index_map_rows = Int[]
    delete_index_map_cols = Int[]
    for j in 1:nreduced
        for i in 1:nreduced
            row_orig = index_map_rows[i]
            col_orig = index_map_cols[j]
            if x[row_orig, col_orig] >= 1-eps()
                push!(fixed_to_one_rows, row_orig)
                push!(fixed_to_one_cols, col_orig)

                push!(delete_index_map_rows, i)
                push!(delete_index_map_cols, j)
            end
        end
    end

    unique!(delete_index_map_rows)
    unique!(delete_index_map_cols)
    sort!(delete_index_map_rows)
    sort!(delete_index_map_cols)
    deleteat!(index_map_rows, delete_index_map_rows)
    deleteat!(index_map_cols, delete_index_map_cols)

    nreduced = length(index_map_rows)
    type = typeof(direction[1, 1])
    d2 = ones(Union{type,Missing}, nreduced, nreduced)
    for j in 1:nreduced
        for i in 1:nreduced
            row_orig = index_map_rows[i]
            col_orig = index_map_cols[j]
            if blmo.append_by_column
                orig_linear_idx = (col_orig-1)*n+row_orig
            else
                orig_linear_idx = (row_orig-1)*n+col_orig
            end
            # idx = findfirst(x -> x == orig_linear_idx, blmo.int_vars)

            if x[row_orig, col_orig] <= eps()
                if blmo.append_by_column
                    d2[i, j] = missing
                else
                    d2[j, i] = missing
                end
            elseif orig_linear_idx in int_vars
                idx = length(int_vars) < n^2 ? idx_map_ub[orig_linear_idx] : orig_linear_idx
                # interdict arc when fixed to zero
                if ub[idx] <= eps()
                    if blmo.append_by_column
                        d2[i, j] = missing
                    else
                        d2[j, i] = missing
                    end
                else
                    if blmo.append_by_column
                        d2[i, j] = direction[row_orig, col_orig]
                    else
                        d2[j, i] = direction[col_orig, row_orig]
                    end
                end
            else
                if blmo.append_by_column
                    d2[i, j] = direction[row_orig, col_orig]
                else
                    d2[j, i] = direction[col_orig, row_orig]
                end
            end
        end
    end

    m = SparseArrays.spzeros(n, n)
    for (i, j) in zip(fixed_to_one_rows, fixed_to_one_cols)
        m[i, j] = 1
    end

    res_mat = Hungarian.munkres(d2)
    (rows, cols, vals) = SparseArrays.findnz(res_mat)
    @inbounds for i in eachindex(cols)
        m[index_map_rows[rows[i]], index_map_cols[cols[i]]] = (vals[i] == 2)
    end

    # if size(direction, 2) == 1
    m = if blmo.append_by_column
        # Convert sparse matrix to sparse vector by columns
        I, J, V = SparseArrays.findnz(m)
        linear_indices = (J .- 1) .* n .+ I
        SparseArrays.sparsevec(linear_indices, V, n^2)
    else
        # Convert sparse matrix to sparse vector by rows (transpose first)
        mt = SparseArrays.sparse(LinearAlgebra.transpose(m))
        I, J, V = SparseArrays.findnz(mt)
        linear_indices = (J .- 1) .* n .+ I
        SparseArrays.sparsevec(linear_indices, V, n^2)
    end
    # end

    return m
end

"""
LMO-like operation which computes a vertex minimizing in `direction` on the face defined by the current fixings.
Fixings are maintained by the oracle (or deduced from `x` itself).
"""
_dicg_maximum_step_impl(blmo::BirkhoffLMO, direction, x; kwargs...) = begin
    n = blmo.dim

    T = promote_type(eltype(x), eltype(direction))
    if size(direction, 2) == 1
        direction =
            blmo.append_by_column ? reshape(direction, (n, n)) :
            transpose(reshape(direction, (n, n)))

        x = blmo.append_by_column ? reshape(x, (n, n)) : transpose(reshape(x, (n, n)))
    end

    gamma_max = one(T)
    for idx in eachindex(x)
        if direction[idx] != 0.0
            # iterate already on the boundary
            if (direction[idx] < 0 && x[idx] ≈ 1) || (direction[idx] > 0 && x[idx] ≈ 0)
                return zero(gamma_max)
            end
            # clipping with the zero boundary
            if direction[idx] > 0
                gamma_max = min(gamma_max, x[idx] / direction[idx])
            else
                @assert direction[idx] < 0
                gamma_max = min(gamma_max, -(1 - x[idx]) / direction[idx])
            end
        end
    end
    return gamma_max
end

_is_decomposition_invariant_oracle_impl(lmo::BirkhoffLMO) = true


# Boscia API: method extensions for BirkhoffBLMO
Boscia.compute_extreme_point(blmo::BosciaBirkhoffBLMO, d; kwargs...) =
    _compute_extreme_point_impl(blmo.birkhoffBLMO, d; kwargs...)

Boscia.compute_inface_extreme_point(blmo::BosciaBirkhoffBLMO, direction, x; kwargs...) =
    _compute_inface_extreme_point_impl(blmo.birkhoffBLMO, direction, x; kwargs...)


Boscia.dicg_maximum_step(blmo::BosciaBirkhoffBLMO, direction, x; kwargs...) =
    _dicg_maximum_step_impl(blmo.birkhoffBLMO, direction, x; kwargs...)

Boscia.is_decomposition_invariant_oracle(blmo::BosciaBirkhoffBLMO) =
    _is_decomposition_invariant_oracle_impl(blmo.birkhoffBLMO)


# FrankWolfe API: method extensions for BirkhoffLMO
FrankWolfe.compute_extreme_point(lmo::FWBirkhoffLMO, d; kwargs...) =
    _compute_extreme_point_impl(lmo.birkhoffLMO, d; kwargs...)

FrankWolfe.compute_inface_extreme_point(lmo::FWBirkhoffLMO, direction, x; kwargs...) =
    _compute_inface_extreme_point_impl(lmo.birkhoffLMO, direction, x; kwargs...)

FrankWolfe.dicg_maximum_step(lmo::FWBirkhoffLMO, direction, x; kwargs...) =
    _dicg_maximum_step_impl(lmo.birkhoffLMO, direction, x; kwargs...)

FrankWolfe.is_decomposition_invariant_oracle(lmo::FWBirkhoffLMO) =
    _is_decomposition_invariant_oracle_impl(lmo.birkhoffLMO)


## Necessary for Boscia package

"""
The sum of each row and column has to be equal to 1.
"""
function Boscia.is_linear_feasible(lmo::BosciaBirkhoffBLMO, v::AbstractVector)
    blmo = lmo.birkhoffBLMO
    n = blmo.dim
    for i in 1:n
        # append by column ? column sum : row sum 
        if !isapprox(sum(v[((i-1)*n+1):(i*n)]), 1.0, atol=1e-6, rtol=1e-3)
            @debug "Column sum not 1: $(sum(v[((i-1)*n+1):(i*n)]))"
            return false
        end
        # append by column ? row sum : column sum
        if !isapprox(sum(v[i:n:(n^2)]), 1.0, atol=1e-6, rtol=1e-3)
            @debug "Row sum not 1: $(sum(v[i:n:n^2]))"
            return false
        end
    end
    return true
end

# Read global bounds from the problem.
function Boscia.build_global_bounds(lmo::BosciaBirkhoffBLMO, integer_variables)
    blmo = lmo.birkhoffBLMO
    global_bounds = Boscia.IntegerBounds()
    for (idx, int_var) in enumerate(blmo.int_vars)
        push!(global_bounds, (int_var, blmo.lower_bounds[idx]), :greaterthan)
        push!(global_bounds, (int_var, blmo.upper_bounds[idx]), :lessthan)
    end
    return global_bounds
end

# Get list of variables indices. 
# If the problem has n variables, they are expected to contiguous and ordered from 1 to n.
function Boscia.get_list_of_variables(lmo::BosciaBirkhoffBLMO)
    blmo = lmo.birkhoffBLMO
    n = blmo.dim^2
    return n, collect(1:n)
end

# Get list of integer variables
function Boscia.get_integer_variables(lmo::BosciaBirkhoffBLMO)
    blmo = lmo.birkhoffBLMO
    return blmo.int_vars
end

# Get the index of the integer variable the bound is working on.
function Boscia.get_int_var(lmo::BosciaBirkhoffBLMO, cidx)
    blmo = lmo.birkhoffBLMO
    return blmo.int_vars[cidx]
end

# Get the list of lower bounds.
function Boscia.get_lower_bound_list(lmo::BosciaBirkhoffBLMO)
    blmo = lmo.birkhoffBLMO
    return collect(1:length(blmo.lower_bounds))
end

# Get the list of upper bounds.
function Boscia.get_upper_bound_list(lmo::BosciaBirkhoffBLMO)
    blmo = lmo.birkhoffBLMO
    return collect(1:length(blmo.upper_bounds))
end

# Read bound value for c_idx.
function Boscia.get_bound(lmo::BosciaBirkhoffBLMO, c_idx, sense::Symbol)
    blmo = lmo.birkhoffBLMO
    if sense == :lessthan
        return blmo.upper_bounds[c_idx]
    elseif sense == :greaterthan
        return blmo.lower_bounds[c_idx]
    else
        error("Allowed value for sense are :lessthan and :greaterthan!")
    end
end

## Changing the bounds constraints.

# Change the value of the bound c_idx.
function Boscia.set_bound!(lmo::BosciaBirkhoffBLMO, c_idx, value, sense::Symbol)
    blmo = lmo.birkhoffBLMO
    if sense == :greaterthan
        blmo.lower_bounds[c_idx] = value
        if value == 1.0
            n0 = blmo.dim
            fixed_int_var = blmo.int_vars[c_idx]
            # Convert linear index to (row, col) based on storage format
            if blmo.append_by_column
                j = ceil(Int, fixed_int_var / n0)  # column index
                i = Int(fixed_int_var - n0 * (j - 1))  # row index
            else
                i = ceil(Int, fixed_int_var / n0)  # row index  
                j = Int(fixed_int_var - n0 * (i - 1))  # column index
            end
            push!(blmo.fixed_to_one_rows, i)
            push!(blmo.fixed_to_one_cols, j)
        end
    elseif sense == :lessthan
        blmo.upper_bounds[c_idx] = value
    else
        error("Allowed values for sense are :lessthan and :greaterthan.")
    end
end

# Delete bounds.
function Boscia.delete_bounds!(lmo::BosciaBirkhoffBLMO, cons_delete)
    blmo = lmo.birkhoffBLMO
    for (d_idx, sense) in cons_delete
        if sense == :greaterthan
            blmo.lower_bounds[d_idx] = -Inf
        else
            blmo.upper_bounds[d_idx] = Inf
        end
    end

    # sanity check 
    check_feasibility(blmo, i::Int, j::Int) =
        blmo.lower_bounds[blmo.append_by_column ? (j-1)*blmo.dim + i : (i-1)*blmo.dim + j] != 0.0

    fixed_to_one_rows = blmo.fixed_to_one_rows
    fixed_to_one_cols = blmo.fixed_to_one_cols

    feasible_flags = check_feasibility.(Ref(blmo), fixed_to_one_rows, fixed_to_one_cols)
    invalid_indices = findall(.!feasible_flags)

    deleteat!(fixed_to_one_rows, invalid_indices)
    deleteat!(fixed_to_one_cols, invalid_indices)

    # remove the duplicate indices
    pairs = collect(zip(blmo.fixed_to_one_rows, blmo.fixed_to_one_cols))
    unique!(pairs)
    resize!(blmo.fixed_to_one_rows, length(pairs))
    resize!(blmo.fixed_to_one_cols, length(pairs))

    blmo.fixed_to_one_rows .= first.(pairs)
    blmo.fixed_to_one_cols .= last.(pairs)

    nfixed = length(blmo.fixed_to_one_rows)
    nreduced = blmo.dim - nfixed

    # stores the indices of the original matrix that are still in the reduced matrix
    index_map_rows = fill(1, nreduced)
    index_map_cols = fill(1, nreduced)
    idx_in_map_row = 1
    idx_in_map_col = 1
    for orig_idx in 1:blmo.dim
        if orig_idx ∉ blmo.fixed_to_one_rows
            index_map_rows[idx_in_map_row] = orig_idx
            idx_in_map_row += 1
        end
        if orig_idx ∉ blmo.fixed_to_one_cols
            index_map_cols[idx_in_map_col] = orig_idx
            idx_in_map_col += 1
        end
    end

    empty!(blmo.index_map_rows)
    empty!(blmo.index_map_cols)
    append!(blmo.index_map_rows, index_map_rows)
    append!(blmo.index_map_cols, index_map_cols)

    return blmo.modify_LMO = false
end

# Add bound constraint.
function Boscia.add_bound_constraint!(lmo::BosciaBirkhoffBLMO, key, value, sense::Symbol)
    blmo = lmo.birkhoffBLMO
    idx = findfirst(x -> x == key, blmo.int_vars)
    if sense == :greaterthan
        blmo.lower_bounds[idx] = value

    elseif sense == :lessthan
        blmo.upper_bounds[idx] = value
    else
        error("Allowed value of sense are :lessthan and :greaterthan!")
    end
end

## Checks

# Check if the subject of the bound c_idx is an integer variable (recorded in int_vars).
function Boscia.is_constraint_on_int_var(lmo::BosciaBirkhoffBLMO, c_idx, int_vars)
    blmo = lmo.birkhoffBLMO
    return blmo.int_vars[c_idx] in int_vars
end

# To check if there is bound for the variable in the global or node bounds.
function Boscia.is_bound_in(lmo::BosciaBirkhoffBLMO, c_idx, bounds)
    blmo = lmo.birkhoffBLMO
    return haskey(bounds, blmo.int_vars[c_idx])
end

# Has variable an integer constraint?
function Boscia.has_integer_constraint(lmo::BosciaBirkhoffBLMO, idx)
    blmo = lmo.birkhoffBLMO
    return idx in blmo.int_vars
end

## Optional

function Boscia.check_feasibility(lmo::BosciaBirkhoffBLMO)
    blmo = lmo.birkhoffBLMO
    for (lb, ub) in zip(blmo.lower_bounds, blmo.upper_bounds)
        if ub < lb
            return Boscia.INFEASIBLE
        end
    end
    # For double stochastic matrices, each row and column must sum to 1
    # We check if the bounds allow for feasible assignments
    n0 = blmo.dim
    n = n0^2
    int_vars = blmo.int_vars
    # Initialize row and column bound tracking
    row_min_sum = zeros(n0)  # minimum possible sum for each row
    row_max_sum = zeros(n0)  # maximum possible sum for each row
    col_min_sum = zeros(n0)  # minimum possible sum for each column
    col_max_sum = zeros(n0)  # maximum possible sum for each column

    rows_with_integer_variables = Int[]
    cols_with_integer_variables = Int[]

    # Process each integer variable
    for idx in eachindex(int_vars)
        var_idx = int_vars[idx]

        # Convert linear index to (row, col) based on storage format
        if blmo.append_by_column
            j = ceil(Int, var_idx / n0)  # column index
            i = Int(var_idx - n0 * (j - 1))  # row index
        else
            i = ceil(Int, var_idx / n0)  # row index  
            j = Int(var_idx - n0 * (i - 1))  # column index
        end

        # Add bounds to row and column sums
        row_min_sum[i] += blmo.lower_bounds[idx]
        row_max_sum[i] += blmo.upper_bounds[idx]
        col_min_sum[j] += blmo.lower_bounds[idx]
        col_max_sum[j] += blmo.upper_bounds[idx]

        push!(rows_with_integer_variables, i)
        push!(cols_with_integer_variables, j)
    end

    # Check feasibility: each row and column must be able to sum to exactly 1
    for i in 1:n0
        if i in rows_with_integer_variables
            # Check row sum constraints
            if row_min_sum[i] > 1 + eps() || row_max_sum[i] < 1 - eps()
                return Boscia.INFEASIBLE
            end
        end

        if i in cols_with_integer_variables
            # Check column sum constraints  
            if col_min_sum[i] > 1 + eps() || col_max_sum[i] < 1 - eps()
                return Boscia.INFEASIBLE
            end
        end
    end

    return Boscia.OPTIMAL
end

