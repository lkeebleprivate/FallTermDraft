#=

    Jacobians depend on data at time level n and n+1, so must distinguish between these. Also must distinguish between variables v and ε, and between the spatial grid locations j-1, j, j+1. This leads to the following dictionary

    'v0m' = v_{j-1}^{n}
    'v0'  = v_{j}^{n}
    'v0p' = v_{j+1}^{n}
    'ε0m' = ε_{j-1}^{n}
    'ε0'  = ε_{j}^{n}
    'ε0p' = ε_{j+1}^{n}

    'v1m' = v_{j-1}^{n+1}
    'v1'  = v_{j}^{n+1}
    'v1p' = v_{j+1}^{n+1}
    'ε1m' = ε_{j-1}^{n+1}
    'ε1'  = ε_{j}^{n+1}
    'ε1p' = ε_{j+1}^{n+1}

=#
include("StressEnergyGradients.jl")
module EulerSystem
using LinearAlgebra, Printf, SparseArrays
using HDF5
using ..StressEnergyGradients

one_norm(x::AbstractArray) = sum(abs, x) / length(x)

sol_fname(k::Float64, h::Float64, xmax::Float64, total_time::Float64, path::String) = path* "Euler_conformal_sol_h_$(h)_k_$(k)_xmax_$(xmax)_T_$(total_time).h5"

function save_sol(data::AbstractArray, residuals::AbstractArray, ind_resid_t::AbstractArray, ind_resid_x::AbstractArray, num_iterations::AbstractArray, L::Int, k::Float64, h::Float64, xmax::Float64, total_time::Float64, path::String)
    fname = sol_fname(k, h, xmax,  total_time, path)

    h5open(fname, "w") do file
        file["t"] = range(start = 0.0, step = k, length = size(data, 2)) |> collect
        file["x"] = range(start = -xmax, stop = xmax, length = L) |> collect
        file["v"] = data[2, :, :]
        file["eps"] = data[1, :, :]
        file["ind_resid_t"] = ind_resid_t
        file["ind_resid_x"] = ind_resid_x
        file["residuals"] = residuals
        file["num_iterations"] = num_iterations
    end

    println("File created: " * fname)
end

function load_sol(k::Float64, h::Float64, xmax::Float64, total_time::Float64, path::String)
    fname = sol_fname(k, h, xmax,  total_time, path)
    file = h5open(fname, "r")
    finalizer(file) do f
        close(f)  # Automatically close when the file object is garbage collected
    end
    return file 
end


function get_jacobian_sparsity(L::Int)
    row_indices::Vector{Int} = []
    column_indices::Vector{Int} = []
    @inbounds for i in 1:L
        if i == 1
            # first L - 1 rows
            # include nonzero elements in first block diagional (idx = 1 to idx = L - 1)  
            push!(row_indices, i)
            push!(column_indices, i)

            push!(row_indices, i)
            push!(column_indices, i+1)

            push!(row_indices, i)
            push!(column_indices, L)

            # include nonzero elements in second block diagonal (idx = L+1 to idx = 2L - 1)  
            push!(row_indices, i)
            push!(column_indices, i+L)

            push!(row_indices, i)
            push!(column_indices, i+L+1)

            push!(row_indices, i)
            push!(column_indices, 2*L)

            # rows L to 2L - 2
            # include nonzero elements in first block diagional (idx = 1 to idx = L - 1)  
            push!(row_indices, i+L)
            push!(column_indices, i)

            push!(row_indices, i+L)
            push!(column_indices, i+1)

            push!(row_indices, i+L)
            push!(column_indices, L)

            # include nonzero elements in second block diagonal (idx = L+1 to idx = 2L - 1)  
            push!(row_indices, i+L)
            push!(column_indices, i+L)

            push!(row_indices, i+L)
            push!(column_indices, i+L+1)

            push!(row_indices, i+L)
            push!(column_indices, 2*L)
        elseif i==L
            # first L - 1 rows
            # include nonzero elements in first block diagional (idx = 1 to idx = L - 1)  
            push!(row_indices, i)
            push!(column_indices, 1)

            push!(row_indices, i)
            push!(column_indices, i-1)

            push!(row_indices, i)
            push!(column_indices, i)

            # include nonzero elements in second block diagonal (idx = L+1 to idx = 2L - 1)  
            push!(row_indices, i)
            push!(column_indices, L+1)

            push!(row_indices, i)
            push!(column_indices, i+L-1)

            push!(row_indices, i)
            push!(column_indices, i+L)

            # rows L to 2L - 2
            # include nonzero elements in first block diagional (idx = 1 to idx = L - 1)  
            push!(row_indices, i+L)
            push!(column_indices, 1)

            push!(row_indices, i+L)
            push!(column_indices, i-1)

            push!(row_indices, i+L)
            push!(column_indices, i)

            # include nonzero elements in second block diagonal (idx = L+1 to idx = 2L - 1)  
            push!(row_indices, i+L)
            push!(column_indices, L+1)

            push!(row_indices, i+L)
            push!(column_indices, i+L-1)

            push!(row_indices, i+L)
            push!(column_indices, i+L)
        else
            # outside boundaries of each of the four blocks, it is tridiagonal
            # first L - 1 rows
            # include nonzero elements in first block diagional (idx = 1 to idx = L - 1)  
            push!(row_indices, i)
            push!(column_indices, i-1)

            push!(row_indices, i)
            push!(column_indices, i)

            push!(row_indices, i)
            push!(column_indices, i+1)

            # include nonzero elements in second block diagonal (idx = L to idx = 2L - 2)  
            push!(row_indices, i)
            push!(column_indices, i+L-1)

            push!(row_indices, i)
            push!(column_indices, i+L)

            push!(row_indices, i)
            push!(column_indices, i+L+1)

            # row L to 2L - 2
            # include nonzero elements in first block diagional (idx = 1 to idx = L - 1)  
            push!(row_indices, i+L)
            push!(column_indices, i-1)

            push!(row_indices, i+L)
            push!(column_indices, i)

            push!(row_indices, i+L)
            push!(column_indices, i+1)

            # include nonzero elements in second block diagonal (idx = L to idx = 2L - 2)  
            push!(row_indices, i+L)
            push!(column_indices, i+L-1)

            push!(row_indices, i+L)
            push!(column_indices, i+L)

            push!(row_indices, i+L)
            push!(column_indices, i+L+1)
        end
    end
    return row_indices, column_indices
end

function generate_update_mask(row_indices, col_indices, num_rows, num_cols)
    # Create a sparse matrix with placeholder values to establish the nzval order
    initial_values = ones(length(row_indices))  # Placeholder values
    A = sparse(row_indices, col_indices, initial_values, num_rows, num_cols)
    
    # Create a dictionary to map (row, col) -> nzval index for the generated sparse matrix
    position_to_nzval_index = Dict{Tuple{Int, Int}, Int}()
    
    @inbounds for col in 1:num_cols
        for idx in A.colptr[col]:(A.colptr[col + 1] - 1)
            row = A.rowval[idx]
            position_to_nzval_index[(row, col)] = idx
        end
    end

    # Generate the mask array by looking up each (row, col) pair in the dictionary
    mask = [position_to_nzval_index[(row_indices[i], col_indices[i])] for i in 1:length(row_indices)]

    return mask
end


function F(k::Float64, h::Float64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    return (-ε0 + ε1)/k + (2*(v0m - v0p + v1m - v1p)*(ε0 + ε1) + (v0 + v1)*(ε0m - ε0p + ε1m - ε1p))/(h*(-12 + (v0 + v1)^2))
end

function G(k::Float64, h::Float64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    return (-v0 + v1)/k + (2*(v0 + v1)*(v0m - v0p + v1m - v1p)*(ε0 + ε1) + 3*(1 - (v0 + v1)^2/4.)^2*(ε0m - ε0p + ε1m - ε1p))/(2.0*h*(-12 + (v0 + v1)^2)*(ε0 + ε1))
end

function compute_function_vector!(F::AbstractArray, v0::AbstractArray, ε0::AbstractArray, uk::AbstractArray, k::Float64, h::Float64, L::Int64)
    @inbounds for i=1:L
        if i == 1
            v0m = v0[L]; v00 = v0[i]; v0p = v0[i+1]; ε0m = ε0[L]; ε00 = ε0[i]; ε0p = ε0[i+1]; v1m = uk[2L]; v1 = uk[i+L]; v1p = uk[i+L+1]; ε1m = uk[L]; ε1 = uk[i]; ε1p = uk[i+1]
        elseif i==L
            v0m = v0[i-1]; v00 = v0[i]; v0p = v0[1]; ε0m = ε0[i-1]; ε00 = ε0[i]; ε0p = ε0[1]; v1m = uk[i+L-1]; v1 = uk[i+L]; v1p = uk[L+1]; ε1m = uk[i-1]; ε1 = uk[i]; ε1p = uk[1]
        else
            v0m = v0[i-1]; v00 = v0[i]; v0p = v0[i+1]; ε0m = ε0[i-1]; ε00 = ε0[i]; ε0p = ε0[i+1]; v1m = uk[i+L-1]; v1 = uk[i+L]; v1p = uk[i+L+1]; ε1m = uk[i-1]; ε1 = uk[i]; ε1p = uk[i+1]
        end

        F[i] = EulerSystem.F(k, h, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
        F[i+L] = EulerSystem.G(k, h, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    end
end

function Jacobian_F_v_j_j(k::Float64, h::Float64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    return (-((12 + (v0 + v1)^2)*ε0m) + 12*(ε0p - ε1m + ε1p) + (v0 + v1)*(-4*(v0m - v0p + v1m - v1p)*(ε0 + ε1) + (v0 + v1)*(ε0p - ε1m + ε1p)))/(h*(-12 + (v0 + v1)^2)^2)
end

function Jacobian_F_v_j_m(k::Float64, h::Float64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    return (2*(ε0 + ε1))/(h*(-12 + (v0 + v1)^2))
end

function Jacobian_F_v_j_p(k::Float64, h::Float64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    return (-2*(ε0 + ε1))/(h*(-12 + (v0 + v1)^2))
end

function Jacobian_F_v(L::Int, k::Float64, h::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    if j == i
        return Jacobian_F_v_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    elseif j == i - 1
        return Jacobian_F_v_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    elseif j == i + 1
        return Jacobian_F_v_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    elseif i == 1 && j == L
        return Jacobian_F_v_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    elseif i == L && j == 1
        return Jacobian_F_v_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F_eps_j_j(k::Float64, h::Float64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    return 1/k + (2*(v0m - v0p + v1m - v1p))/(h*(-12 + (v0 + v1)^2))
end

function Jacobian_F_eps_j_m(k::Float64, h::Float64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    return (v0 + v1)/(h*(-12 + (v0 + v1)^2))
end

function Jacobian_F_eps_j_p(k::Float64, h::Float64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    return -((v0 + v1)/(h*(-12 + (v0 + v1)^2)))
end

function Jacobian_F_eps(L::Int, k::Float64, h::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    if j == i
        return Jacobian_F_eps_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    elseif j == i - 1
        return Jacobian_F_eps_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    elseif j == i + 1
        return Jacobian_F_eps_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    elseif i == 1 && j == L
        return Jacobian_F_eps_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    elseif i == L && j == 1
        return Jacobian_F_eps_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_G_v_j_j(k::Float64, h::Float64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    return 1/k + (2*(v0m - v0p + v1m - v1p)*(ε0 + ε1) + (3*(-2 + v0 + v1)*(v0 + v1)*(2 + v0 + v1)*(ε0m - ε0p + ε1m - ε1p))/4.)/(2.0*h*(-12 + (v0 + v1)^2)*(ε0 + ε1)) - ((v0 + v1)*(2*(v0 + v1)*(v0m - v0p + v1m - v1p)*(ε0 + ε1) + 3*(1 - (v0 + v1)^2/4.)^2*(ε0m - ε0p + ε1m - ε1p)))/(h*(-12 + (v0 + v1)^2)^2*(ε0 + ε1))
end

function Jacobian_G_v_j_m(k::Float64, h::Float64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    return (v0 + v1)/(h*(-12 + (v0 + v1)^2))
end

function Jacobian_G_v_j_p(k::Float64, h::Float64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    return -((v0 + v1)/(h*(-12 + (v0 + v1)^2)))
end

function Jacobian_G_v(L::Int, k::Float64, h::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    if j == i
        return Jacobian_G_v_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    elseif j == i - 1
        return Jacobian_G_v_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    elseif j == i + 1
        return Jacobian_G_v_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    elseif i == 1 && j == L
        return Jacobian_G_v_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    elseif i == L && j == 1
        return Jacobian_G_v_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_G_eps_j_j(k::Float64, h::Float64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    return (-3*(1 - (v0 + v1)^2/4.)^2*(ε0m - ε0p + ε1m - ε1p))/(2.0*h*(-12 + (v0 + v1)^2)*(ε0 + ε1)^2)
end

function Jacobian_G_eps_j_m(k::Float64, h::Float64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    return (3*(1 - (v0 + v1)^2/4.)^2)/(2.0*h*(-12 + (v0 + v1)^2)*(ε0 + ε1))
end

function Jacobian_G_eps_j_p(k::Float64, h::Float64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    return (-3*(1 - (v0 + v1)^2/4.)^2)/(2.0*h*(-12 + (v0 + v1)^2)*(ε0 + ε1))
end

function Jacobian_G_eps(L::Int, k::Float64, h::Float64, i::Int, j::Int, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64)
    if j == i
        return Jacobian_G_eps_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    elseif j == i - 1
        return Jacobian_G_eps_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    elseif j == i + 1
        return Jacobian_G_eps_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    elseif i == 1 && j == L
        return Jacobian_G_eps_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    elseif i == L && j == 1
        return Jacobian_G_eps_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

# v0, ε0 are solution variables from previous time step—each of length L. uk is kth newton iterate of length 2L (i.e., all the variables vk, εk)
function compute_sparse_euler_jacobian!(J::AbstractArray, JVector::Vector{Float64}, mask::Vector{Int}, v0::AbstractArray, ε0::AbstractArray, uk::AbstractArray, k::Float64, h::Float64, L::Int64)
    idx = 1
    @inbounds for i=1:L
        # solution values at relevant points in a nearby stencil (enforcing periodic BCs)
        if i == 1
            v0m = v0[L]; v00 = v0[i]; v0p = v0[i+1]; ε0m = ε0[L]; ε00 = ε0[i]; ε0p = ε0[i+1]; v1m = uk[2L]; v1 = uk[i+L]; v1p = uk[i+L+1]; ε1m = uk[L]; ε1 = uk[i]; ε1p = uk[i+1]
        elseif i==L
            v0m = v0[i-1]; v00 = v0[i]; v0p = v0[1]; ε0m = ε0[i-1]; ε00 = ε0[i]; ε0p = ε0[1]; v1m = uk[i+L-1]; v1 = uk[i+L]; v1p = uk[L+1]; ε1m = uk[i-1]; ε1 = uk[i]; ε1p = uk[1]
        else
            v0m = v0[i-1]; v00 = v0[i]; v0p = v0[i+1]; ε0m = ε0[i-1]; ε00 = ε0[i]; ε0p = ε0[i+1]; v1m = uk[i+L-1]; v1 = uk[i+L]; v1p = uk[i+L+1]; ε1m = uk[i-1]; ε1 = uk[i]; ε1p = uk[i+1]
        end

        if i == 1
            JVector[idx] = Jacobian_F_eps(L, k, h, i, i, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_F_eps(L, k, h, i, i+1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_F_eps(L, k, h, i, L, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;

            JVector[idx] = Jacobian_F_v(L, k, h, i, i, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_F_v(L, k, h, i, i+1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_F_v(L, k, h, i, L, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;

            JVector[idx] = Jacobian_G_eps(L, k, h, i, i, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_G_eps(L, k, h, i, i+1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_G_eps(L, k, h, i, L, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;

            JVector[idx] = Jacobian_G_v(L, k, h, i, i, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_G_v(L, k, h, i, i+1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_G_v(L, k, h, i, L, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
        elseif i == L
            JVector[idx] = Jacobian_F_eps(L, k, h, i, 1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_F_eps(L, k, h, i, i-1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_F_eps(L, k, h, i, i, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;

            JVector[idx] = Jacobian_F_v(L, k, h, i, 1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_F_v(L, k, h, i, i-1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_F_v(L, k, h, i, i, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;

            JVector[idx] = Jacobian_G_eps(L, k, h, i, 1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_G_eps(L, k, h, i, i-1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_G_eps(L, k, h, i, i, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;

            JVector[idx] = Jacobian_G_v(L, k, h, i, 1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_G_v(L, k, h, i, i-1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_G_v(L, k, h, i, i, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
        else
            JVector[idx] = Jacobian_F_eps(L, k, h, i, i-1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_F_eps(L, k, h, i, i, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_F_eps(L, k, h, i, i+1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;

            JVector[idx] = Jacobian_F_v(L, k, h, i, i-1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_F_v(L, k, h, i, i, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_F_v(L, k, h, i, i+1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;

            JVector[idx] = Jacobian_G_eps(L, k, h, i, i-1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_G_eps(L, k, h, i, i, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_G_eps(L, k, h, i, i+1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;

            JVector[idx] = Jacobian_G_v(L, k, h, i, i-1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_G_v(L, k, h, i, i, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
            JVector[idx] = Jacobian_G_v(L, k, h, i, i+1, v0m, v00, v0p, ε0m, ε00, ε0p, v1m, v1, v1p, ε1m, ε1, ε1p); idx += 1;
        end
    end
    J.nzval[mask] .= JVector
end

function data_array_access(u::AbstractArray, size::Int, var::Int, time::Int, i::Int)
    if i < 1
        return u[var, time, end + i]
    elseif i > size
        return u[var, time, i - size]
    else
        return u[var, time, i]
    end
end

function kreiss_oliger(data::AbstractArray, size::Int, var::Int, time::Int, j::Int, coef::Float64)
    u_minus_2 = data_array_access(data, size, var, time, j-2)
    u_minus_1 = data_array_access(data, size, var, time, j-1)
    u = data_array_access(data, size, var, time, j)
    u_plus_1 = data_array_access(data, size, var, time, j+1)
    u_plus_2 = data_array_access(data, size, var, time, j+2)
    return coef * (u_plus_2 - 4.0 * u_plus_1 + 6.0 * u - 4.0 * u_minus_1 + u_minus_2) / 16.0
end


# for now use separate loops but eventually merge
function solve!(L::Int, k::Float64, h::Float64, ε0::Vector{Float64}, v0::Vector{Float64}, xmax::Float64, total_time::Float64, kreiss_coef::Float64, target_tol::Float64, max_tol::Float64, max_iter::Int, min_iter::Int, path::String)
    # allocate memory
    newton_iterates = zeros(2 * L)
    function_vector = zeros(2 * L)
    εn = zeros(L)
    vn = zeros(L)

    # initialize sparse jacobiam and get mask for Jacobian update
    row_indices, column_indices = EulerSystem.get_jacobian_sparsity(L);
    vals = ones(length(row_indices))
    sparse_jacobian = sparse(row_indices, column_indices, vals, 2 * L, 2 * L)
    JVector = zeros(length(row_indices))
    mask = EulerSystem.generate_update_mask(row_indices, column_indices, 2 * L, 2 * L);

    # set up grid
    N = Int(ceil(total_time / k))
    data = zeros(2, N, L)

    residuals = zeros(N)
    num_iterations = zeros(N)

    data[1, 1, :] = ε0[:]
    data[2, 1, :] = v0[:]

    # residual diagnostics
    resid_t_component = zeros(N-2, L)
    resid_x_component = zeros(N-2, L)

    stop = false
    resid = NaN
    n_iter = 0
    
    try
        for n = 1:N-1
            print_string = "Time Level: $n, Time: $(round(k * n; digits=4)), Completion: $(round(100 * n/(N-1); digits=5))%, Latest resid: $resid, Latest number of iterations: $(n_iter)   \r" 
            print(print_string)

            newton_iterates[1:L] =  data[1, n, :][:]
            newton_iterates[L+1:end] = data[2, n, :][:]

            # add KO dissipation at most retarded time level
            εn[:] = data[1, n, :][:]
            vn[:] = data[2, n, :][:]

            for i=1:L
                εn[i] += -kreiss_oliger(data, L, 1, n, i, kreiss_coef)
                vn[i] += -kreiss_oliger(data, L, 2, n, i, kreiss_coef)
            end

            compute_function_vector!(function_vector, vn, εn, newton_iterates, k, h, L)

            n_iter = 0

            for step = 1:max_iter
                n_iter += 1
                compute_sparse_euler_jacobian!(sparse_jacobian, JVector, mask, vn, εn, newton_iterates, k, h, L)

                Δ = sparse_jacobian \ (-function_vector)
                newton_iterates += Δ

                compute_function_vector!(function_vector, vn, εn, newton_iterates, k, h, L)
        
                resid = maximum(abs.(function_vector))
  
                if resid < target_tol && step >= min_iter
                    break
                elseif resid < max_tol && step == max_iter
                    break
                elseif resid > max_tol && step == max_iter
                    println("Newton iteration did not converge. Resid = $resid. Stopping at time level $n.")
                    stop = true
                end
            end

            # return newton_iterates
            if stop
                break
            end

            num_iterations[n] = n_iter
            residuals[n] = resid

            # println("Updating data")
            data[1, n+1, :] = newton_iterates[1:L]
            data[2, n+1, :] = newton_iterates[L+1:end]

            # compute independent residual
            if n > 1
                for j=1:L
                    v0 = data[2, n-1, j]; ε0 = data[1, n-1, j];
                    v2 = data[2, n+1, j]; ε2 = data[1, n+1, j];
                    if j == 1
                        v1m = data[2, n, end]; v1 = data[2, n, j]; v1p = data[2, n, j+1]; ε1m = data[1, n, end]; ε1 = data[1, n, j]; ε1p = data[1, n, j+1];
                    elseif j==L
                        v1m = data[2, n, j-1]; v1 = data[2, n, j]; v1p = data[2, n, 1]; ε1m = data[1, n, j-1]; ε1 = data[1, n, j]; ε1p = data[1, n, 1];
                    else
                        v1m = data[2, n, j-1]; v1 = data[2, n, j]; v1p = data[2, n, j+1]; ε1m = data[1, n, j-1]; ε1 = data[1, n, j]; ε1p = data[1, n, j+1];
                    end

                    resid_t_component[n-1, j] = abs(StressEnergyGradients.Euler.stress_energy_gradient_t(v0, ε0, v1m, v1, v1p, ε1m, ε1, ε1p, v2, ε2, k, h))
                    resid_x_component[n-1, j] = abs(StressEnergyGradients.Euler.stress_energy_gradient_x(v0, ε0, v1m, v1, v1p, ε1m, ε1, ε1p, v2, ε2, k, h))
                end
            end
        end
    catch e
        println("\nERROR OCCURRED. SAVING SOLUTION AND RETRHOWING ERROR.\n")
        save_sol(data, residuals, resid_t_component, resid_x_component, num_iterations, L, k, h, xmax,  total_time, path)
        # rethrow(e)
        return resid
    end
    save_sol(data, residuals, resid_t_component, resid_x_component, num_iterations, L, k, h, xmax,  total_time, path)
end

end