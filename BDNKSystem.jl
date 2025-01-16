include("BDNKJacobian.jl")
include("BDNKConformalTensor.jl")
include("StressEnergyGradients.jl")
include("FiniteDiff.jl")
module BDNKSystem
using ..BDNKJacobian
using ..BDNKConformalTensor
using ..StressEnergyGradients
using ...FiniteDiffOrder5
using LinearAlgebra
using Printf
using SparseArrays
using HDF5

one_norm(x::AbstractArray) = sum(abs, x) / length(x)

function load_sol(type::String, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, kreiss_coef::Float64, xmax::Float64, total_time::Float64, path::String)
    if isequal(type, "PERIODIC")
        fname = BDNKSystem.Periodic.sol_fname(k, h, η0, λ0, χ0, kreiss_coef, xmax,  total_time, path)
    elseif isequal(type, "APERIODIC")
        fname = BDNKSystem.Aperiodic.sol_fname(k, h, η0, λ0, χ0, kreiss_coef, xmax,  total_time, path)
    else
        throw("Invalid type. Choose either PERIODIC or APERIODIC.")
    end

    file = h5open(fname, "r")
    finalizer(file) do f
        close(f)  # Automatically close when the file object is garbage collected
    end
    return file 
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


# function to compute the initial energy and velocity time derivatives from the choice of v, ε, Π^{tt} and Π^{tx}
function εdot(pi1tt::Float64, pi1tx::Float64, v::Float64, ε::Float64, v_prime::Float64, ε_prime::Float64, η0::Float64, λ0::Float64, χ0::Float64)
    return (-4*sqrt(1 - v^2)*(-3*pi1tt*λ0 + 3*pi1tx*(2*λ0 + χ0)*v + pi1tt*(4*η0 - 3*λ0 - 4*χ0)*v^2 + pi1tx*(-4*η0 + χ0)*v^3)*ε^0.25 + 4*λ0*(-3*χ0 + (-4*η0 + χ0)*v^2)*ε*v_prime + 2*v*(-3*λ0*χ0 + (2*η0*λ0 + 6*η0*χ0 +
        λ0*χ0)*v^2)*ε_prime)/(9*λ0*χ0 - 6*(2*η0 + λ0)*χ0*v^2 + λ0*(-4*η0 + χ0)*v^4)
end

function vdot(pi1tt::Float64, pi1tx::Float64, v::Float64, ε::Float64, v_prime::Float64, ε_prime::Float64, η0::Float64, λ0::Float64, χ0::Float64)
    return (12*(1 - v^2)^1.5*(3*pi1tx*χ0 - pi1tt*(λ0 + 4*χ0)*v + pi1tx*(2*λ0 + χ0)*v^2 - pi1tt*λ0*v^3)*ε^0.25 + 8*v*(6*η0*χ0 - 3*λ0*χ0 + λ0*(2*η0 + χ0)*v^2)*ε*v_prime + 3*λ0*χ0*(-3 + v^2)*(-1 + v^2)^2*ε_prime)/(4.0*(9*λ0*χ0 -
        6*(2*η0 + λ0)*χ0*v^2 + λ0*(-4*η0 + χ0)*v^4)*ε)        
end

function initial_conditions(v::Vector{Float64}, ε::Vector{Float64}, pi1tt::Vector{Float64}, pi1tx::Vector{Float64}, v_prime::Vector{Float64}, ε_prime::Vector{Float64}, η0::Float64, λ0::Float64, χ0::Float64)
    v_dot = zeros(length(v))
    ε_dot = zeros(length(v))
    for i in eachindex(v_dot)
        v_dot[i] = vdot(pi1tt[i], pi1tx[i], v[i], ε[i], v_prime[i], ε_prime[i], η0, λ0, χ0)
        ε_dot[i] = εdot(pi1tt[i], pi1tx[i], v[i], ε[i], v_prime[i], ε_prime[i], η0, λ0, χ0)
    end
    return v_dot, ε_dot
end

function compute_initial_data(L::Int, ε0_func::Function, v0_func::Function, ε0_prime_func::Function, v0_prime_func::Function, frame::Vector{Float64}, kreiss_coef::Float64, xmax::Float64)
    x_h = range(start=-xmax, stop=xmax, length=L) |> collect;
    ε0 = ε0_func.(x_h)
    v0 = v0_func.(x_h)
    ε0_prime = ε0_prime_func.(x_h)
    v0_prime = v0_prime_func.(x_h)
    pi1tt0 = zeros(L)
    pi1tx0 = zeros(L)
    vdot0, εdot0 = initial_conditions(v0, ε0, pi1tt0, pi1tx0, v0_prime, ε0_prime, frame...)
    return v0, ε0, vdot0, εdot0
end

module Periodic
using ..BDNKSystem
using ...BDNKJacobian
using ...BDNKConformalTensor
using ...StressEnergyGradients
using ...FiniteDiffOrder5
using LinearAlgebra
using Printf
using SparseArrays
using HDF5

sol_fname(k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, kreiss_coef::Float64, xmax::Float64, total_time::Float64, path::String) = path * "BDNK_periodic_conformal_sol_k_$(k)_h_$(h)_eta0_$(η0)_lambda0_$(λ0)_chi0_$(χ0)_kreiss_$(kreiss_coef)_xmax_$(xmax)_T_$(total_time).h5"

function save_sol(data::AbstractArray, diagnostics::AbstractArray, residuals::AbstractArray, ind_resid_t::AbstractArray, ind_resid_x::AbstractArray, v_prime_1::AbstractArray, v_prime_2::AbstractArray, v_prime_3::AbstractArray, eps_prime_1::AbstractArray,
    eps_prime_2::AbstractArray, eps_prime_3::AbstractArray, num_iterations::AbstractArray, L::Int, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, kreiss_coef::Float64, xmax::Float64, total_time::Float64, path::String)
    fname = sol_fname(k, h, η0, λ0, χ0, kreiss_coef, xmax,  total_time, path)
    h5open(fname, "w") do file
        file["t"] = range(start = 0.0, step = k, length = size(data, 2)) |> collect
        file["x"] = range(start = -xmax, stop = xmax, length = L) |> collect
        file["v"] = data[4, :, :]
        file["eps"] = data[3, :, :]
        file["vdot"] = data[2, :, :]
        file["epsdot"] = data[1, :, :]
        file["v_prime_1"] = v_prime_1
        file["v_prime_2"] = v_prime_2
        file["v_prime_3"] = v_prime_3
        file["eps_prime_1"] = eps_prime_1
        file["eps_prime_2"] = eps_prime_2
        file["eps_prime_3"] = eps_prime_3
        file["T_tt_0"] = diagnostics[1, :, :]
        file["T_tt_1"] = diagnostics[3, :, :]
        file["T_xx_0"] = diagnostics[2, :, :]
        file["T_xx_1"] = diagnostics[4, :, :]
        file["vel_dot_T"] = diagnostics[5, :, :]    # inner product u_{a} u_{b} T^{ab} to check weak energy condition
        file["residuals"] = residuals
        file["ind_resid_t"] = ind_resid_t
        file["ind_resid_x"] = ind_resid_x
        file["num_iterations"] = num_iterations
    end
    println("File created: " * fname)
end

# the jacobain matrix is a block matrix with 16 blocks. The first L rows corresponds to the crank nicholson equation for \dot{ε} evaluate at the L grid points. The next L rows correspond to the equation for \dot{v}, the next L for ε and the final L for v. The first L columns correspond to derivatives with respect to \dot{ε} at the advanced time level (n+1) at the L grid points, the next L columns on \dot{v} at n+1, the next L for ε at n+1 and the final L for v at n+1. Note that the crank nicholson equations for ε and v are trivial as a result of the time reduction E==\dot{ε} ==> \partial_{t}(ε) = E.
function get_jacobian_sparsity(L::Int)
    row_indices::Vector{Int} = []
    column_indices::Vector{Int} = []
    @inbounds for i in 1:L
        # since we include Kreiss-Oliger dissipation in the implicit equations, each block in the jacobian will include up to 5 nontrivial entries per row. We must implement special cases for rows 1, 2, L-1, and L when the length-5 stencils will wrap around due to the periodic boundary conditions
        if i == 1
            # first L rows correspond to the crank nicholson equation for \dot{ε} with kreiss oliger dissipation for \dot{ε}. The first L columns will thus contain 5 nontrivial entries per row, while the remaining columns  (3 blocks) will have 3 nontrivial entries per row.
            # include nonzero elements in first block diagional (idx = 1 to idx = L)  
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

            # include nonzero elements in third block diagonal (idx = 2L+1 to idx = 3L - 1)  
            push!(row_indices, i)
            push!(column_indices, i+2*L)

            push!(row_indices, i)
            push!(column_indices, i+2*L+1)

            push!(row_indices, i)
            push!(column_indices, 3*L)

            # include nonzero elements in fourth block diagonal (idx = 3L+1 to idx = 4L)  
            push!(row_indices, i)
            push!(column_indices, i+3*L)

            push!(row_indices, i)
            push!(column_indices, i+3*L+1)

            push!(row_indices, i)
            push!(column_indices, 4*L)

            # rows L + 1 to 2L — CN equation for \dot{v} with Kreiss-Oliger dissipation. Blocks 1, 3, 4 will have 3 nontrivial entries, block 2 will have 5 nontrivial entries.
            # include nonzero elements in first block diagional (idx = 1 to idx = L)  
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

            # include nonzero elements in third block diagonal (idx = 2L+1 to idx = 3L - 1)  
            push!(row_indices, i+L)
            push!(column_indices, i+2*L)

            push!(row_indices, i+L)
            push!(column_indices, i+2*L+1)

            push!(row_indices, i+L)
            push!(column_indices, 3*L)

            # include nonzero elements in fourth block diagonal (idx = 3L+1 to idx = 4L)  
            push!(row_indices, i+L)
            push!(column_indices, i+3*L)

            push!(row_indices, i+L)
            push!(column_indices, i+3*L+1)

            push!(row_indices, i+L)
            push!(column_indices, 4*L)

            # rows 2L+1 to 3L — CN equation for ε with Kreiss-Oliger dissipation. Equation is independent of \dot{v} and v, so blocks 2 and 4 will be identically zero. Block 1 will have one nontrivial entry due to the trivial CN equation for ε, while block 3 will have 5 nontrivial entries due to the KO dissipation
            push!(row_indices, i+2*L)
            push!(column_indices, i)

            push!(row_indices, i+2*L)
            push!(column_indices, i+2*L)

            # rows 3L+1 to 4L  — CN equation for v with Kreiss-Oliger dissipation. Equation is independent of \dot{ε} and ε, so blocks 1 and 3 will be identically zero. Block 2 will have one nontrivial entry due to the trivial CN equation for v, while block 4 will have 5 nontrivial entries due to the KO dissipation
            push!(row_indices, i+3*L)
            push!(column_indices, i+L)

            push!(row_indices, i+3*L)
            push!(column_indices, i+3*L)

            # no nonzero elements in fourth block diagonal (idx = 3L+1 to idx = 4L - 4) since ∂F4/∂ε = 0

        elseif i==L
            # first L rows
            # include nonzero elements in first block diagional (idx = 1 to idx = L)  
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

            # include nonzero elements in third block diagonal (idx = 2L+1 to idx = 3L - 1)  
            push!(row_indices, i)
            push!(column_indices, 2*L+1)

            push!(row_indices, i)
            push!(column_indices, i+2*L-1)

            push!(row_indices, i)
            push!(column_indices, i+2*L)

            # include nonzero elements in fourth block diagonal (idx = 3L+1 to idx = 4L)  
            push!(row_indices, i)
            push!(column_indices, 3*L+1)

            push!(row_indices, i)
            push!(column_indices, i+3*L-1)

            push!(row_indices, i)
            push!(column_indices, i+3*L)

            # rows L + 1 to 2L
            # include nonzero elements in first block diagional (idx = 1 to idx = L)  
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

            # include nonzero elements in third block diagonal (idx = 2L+1 to idx = 3L - 1)  
            push!(row_indices, i+L)
            push!(column_indices, 2*L+1)
            
            push!(row_indices, i+L)
            push!(column_indices, i+2*L-1)

            push!(row_indices, i+L)
            push!(column_indices, i+2*L)

            # include nonzero elements in fourth block diagonal (idx = 3L+1 to idx = 4L)  
            push!(row_indices, i+L)
            push!(column_indices, 3*L+1)
            
            push!(row_indices, i+L)
            push!(column_indices, i+3*L-1)

            push!(row_indices, i+L)
            push!(column_indices, i+3*L)

            # rows 2L-1 to 3L - 3
            # no nonzero elements in first block diagional (idx = 1 to idx = L) since ∂F3/∂V = 0
            # include nonzero elements in second block diagonal (idx = L+1 to idx = 2L - 1)  
            push!(row_indices, i+2*L)
            push!(column_indices, i)

            push!(row_indices, i+2*L)
            push!(column_indices, i+2*L)

            # rows 3L+1 to 4L
            # include nonzero elements in first block diagional (idx = 1 to idx = L)  
            push!(row_indices, i+3*L)
            push!(column_indices, i+L)

            push!(row_indices, i+3*L)
            push!(column_indices, i+3*L)

            # no nonzero elements in fourth block diagonal (idx = 3L+1 to idx = 4L - 4) since ∂F4/∂ε = 0
        else
            # outside boundaries of each of the four blocks, it is tridiagonal
            # first L rows
            # include nonzero elements in first block diagional (idx = 1 to idx = L)  
            push!(row_indices, i)
            push!(column_indices, i-1)

            push!(row_indices, i)
            push!(column_indices, i)

            push!(row_indices, i)
            push!(column_indices, i+1)

            # include nonzero elements in second block diagonal (idx = L+1 to idx = 2L - 1)  
            push!(row_indices, i)
            push!(column_indices, i+L-1)

            push!(row_indices, i)
            push!(column_indices, i+L)

            push!(row_indices, i)
            push!(column_indices, i+L+1)

            # include nonzero elements in third block diagonal (idx = 2L+1 to idx = 3L - 1)  
            push!(row_indices, i)
            push!(column_indices, i+2*L-1)

            push!(row_indices, i)
            push!(column_indices, i+2*L)

            push!(row_indices, i)
            push!(column_indices, i+2*L+1)

            # include nonzero elements in fourth block diagonal (idx = 2L+1 to idx = 3L - 1)  
            push!(row_indices, i)
            push!(column_indices, i+3*L-1)

            push!(row_indices, i)
            push!(column_indices, i+3*L)

            push!(row_indices, i)
            push!(column_indices, i+3*L+1)

            # row L to 2L - 2
            # include nonzero elements in first block diagional (idx = 1 to idx = L)  
            push!(row_indices, i+L)
            push!(column_indices, i-1)

            push!(row_indices, i+L)
            push!(column_indices, i)

            push!(row_indices, i+L)
            push!(column_indices, i+1)
            

            # include nonzero elements in second block diagonal (idx = L+1 to idx = 2L - 1)  
            push!(row_indices, i+L)
            push!(column_indices, i+L-1)

            push!(row_indices, i+L)
            push!(column_indices, i+L)

            push!(row_indices, i+L)
            push!(column_indices, i+L+1)

            # include nonzero elements in third block diagonal (idx = 2L+1 to idx = 3L - 1)  
            push!(row_indices, i+L)
            push!(column_indices, i+2*L-1)

            push!(row_indices, i+L)
            push!(column_indices, i+2*L)

            push!(row_indices, i+L)
            push!(column_indices, i+2*L+1)

            # include nonzero elements in fourth block diagonal (idx = 2L+1 to idx = 3L - 1)  
            push!(row_indices, i+L)
            push!(column_indices, i+3*L-1)

            push!(row_indices, i+L)
            push!(column_indices, i+3*L)

            push!(row_indices, i+L)
            push!(column_indices, i+3*L+1)

            # rows 2L-1 to 3L - 3

            push!(row_indices, i+2*L)
            push!(column_indices, i)

            push!(row_indices, i+2*L)
            push!(column_indices, i+2*L)

            # rows 3L+1 to 4L
            push!(row_indices, i+3*L)
            push!(column_indices, i+L)

            push!(row_indices, i+3*L)
            push!(column_indices, i+3*L)
        end
    end
    return row_indices, column_indices
end

# u is current newton iterate of length 4L-4 (i.e., all the variables Vk, Ek, vk, εk evaluated at 1,..., L --- ignore point L since it is the same as 1 due to periodic BCs)
function compute_sparse_BDNK_jacobian!(J::AbstractArray, JVector::Vector{Float64}, mask::Vector{Int}, u::AbstractArray, u0::AbstractArray, k::Float64, h::Float64, L::Int64, η0::Float64, λ0::Float64, χ0::Float64)
    idx = 1
    @inbounds for i=1:L
        # solution values at relevant points in a nearby stencil (enforcing periodic BCs)
        if i == 1
            E0m = u0[L]; E0 = u0[i]; E0p = u0[i+1]; V0m = u0[2L]; V0 = u0[i+L]; V0p = u0[i+L+1]; ε0m = u0[3L]; ε0 = u0[i+2L]; ε0p = u0[i+2L+1]; v0m = u0[4L]; v0 = u0[i+3L]; v0p = u0[i+3L+1];
            E1m = u[L]; E1 = u[i]; E1p = u[i+1]; V1m = u[2L]; V1 = u[i+L]; V1p = u[i+L+1]; ε1m = u[3L]; ε1 = u[i+2L]; ε1p = u[i+2L+1]; v1m = u[4L]; v1 = u[i+3L]; v1p = u[i+3L+1];
        elseif i==L
            E0m = u0[i-1]; E0 = u0[i]; E0p = u0[1]; V0m = u0[i+L-1]; V0 = u0[i+L]; V0p = u0[L+1]; ε0m = u0[i+2L-1]; ε0 = u0[i+2L]; ε0p = u0[2L+1]; v0m = u0[i+3L-1]; v0 = u0[i+3L]; v0p = u0[3L+1];
            E1m = u[i-1]; E1 = u[i]; E1p = u[1]; V1m = u[i+L-1]; V1 = u[i+L]; V1p = u[L+1]; ε1m = u[i+2L-1]; ε1 = u[i+2L]; ε1p = u[2L+1]; v1m = u[i+3L-1]; v1 = u[i+3L]; v1p = u[3L+1];
        else
            E0m = u0[i-1]; E0 = u0[i]; E0p = u0[i+1]; V0m = u0[i+L-1]; V0 = u0[i+L]; V0p = u0[i+L+1]; ε0m = u0[i+2L-1]; ε0 = u0[i+2L]; ε0p = u0[i+2L+1]; v0m = u0[i+3L-1]; v0 = u0[i+3L]; v0p = u0[i+3L+1];
            E1m = u[i-1]; E1 = u[i]; E1p = u[i+1]; V1m = u[i+L-1]; V1 = u[i+L]; V1p = u[i+L+1]; ε1m = u[i+2L-1]; ε1 = u[i+2L]; ε1p = u[i+2L+1]; v1m = u[i+3L-1]; v1 = u[i+3L]; v1p = u[i+3L+1];
        end

        if i == 1
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_E(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_E(L, k, h, η0, λ0, χ0, i, L, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_V(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_V(L, k, h, η0, λ0, χ0, i, L, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_eps(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_eps(L, k, h, η0, λ0, χ0, i, L, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_v(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_v(L, k, h, η0, λ0, χ0, i, L, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_E(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_E(L, k, h, η0, λ0, χ0, i, L, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_V(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_V(L, k, h, η0, λ0, χ0, i, L, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_eps(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_eps(L, k, h, η0, λ0, χ0, i, L, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_v(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_v(L, k, h, η0, λ0, χ0, i, L, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F3_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F3_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F4_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F4_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
        elseif i == L
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_E(L, k, h, η0, λ0, χ0, i, 1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_E(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_V(L, k, h, η0, λ0, χ0, i, 1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_V(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_eps(L, k, h, η0, λ0, χ0, i, 1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_eps(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_v(L, k, h, η0, λ0, χ0, i, 1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_v(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_E(L, k, h, η0, λ0, χ0, i, 1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_E(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_V(L, k, h, η0, λ0, χ0, i, 1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_V(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_eps(L, k, h, η0, λ0, χ0, i, 1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_eps(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_v(L, k, h, η0, λ0, χ0, i, 1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_v(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F3_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F3_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F4_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F4_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
        else
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_E(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_E(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_V(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_V(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_eps(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_eps(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_v(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F1_v(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_E(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_E(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_V(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_V(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_eps(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_eps(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_v(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F2_v(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F3_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F3_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Periodic.Jacobian_F4_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Periodic.Jacobian_F4_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

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

function kreiss_oliger_low_wavelength(data::AbstractArray, size::Int, var::Int, time::Int, j::Int, coef::Float64, spacing::Int)
    u_minus_2 = data_array_access(data, size, var, time, j-2*spacing)
    u_minus_1 = data_array_access(data, size, var, time, j-1*spacing)
    u = data_array_access(data, size, var, time, j)
    u_plus_1 = data_array_access(data, size, var, time, j+1*spacing)
    u_plus_2 = data_array_access(data, size, var, time, j+2*spacing)
    return coef * (u_plus_2 - 4.0 * u_plus_1 + 6.0 * u - 4.0 * u_minus_1 + u_minus_2) / 16.0
end

# u denotes current newton iteration
function compute_function_vector!(F::AbstractArray, u::AbstractArray, u0::AbstractArray, k::Float64, h::Float64, L::Int64, η0::Float64, λ0::Float64, χ0::Float64)
    @inbounds for i=1:L
        # solution values at relevant points in a nearby stencil (enforcing periodic BCs)
        if i == 1
            E0m = u0[L]; E0 = u0[i]; E0p = u0[i+1]; V0m = u0[2L]; V0 = u0[i+L]; V0p = u0[i+L+1]; ε0m = u0[3L]; ε0 = u0[i+2L]; ε0p = u0[i+2L+1]; v0m = u0[4L]; v0 = u0[i+3L]; v0p = u0[i+3L+1];
            E1m = u[L]; E1 = u[i]; E1p = u[i+1]; V1m = u[2L]; V1 = u[i+L]; V1p = u[i+L+1]; ε1m = u[3L]; ε1 = u[i+2L]; ε1p = u[i+2L+1]; v1m = u[4L]; v1 = u[i+3L]; v1p = u[i+3L+1];
        elseif i==L
            E0m = u0[i-1]; E0 = u0[i]; E0p = u0[1]; V0m = u0[i+L-1]; V0 = u0[i+L]; V0p = u0[L+1]; ε0m = u0[i+2L-1]; ε0 = u0[i+2L]; ε0p = u0[2L+1]; v0m = u0[i+3L-1]; v0 = u0[i+3L]; v0p = u0[3L+1];
            E1m = u[i-1]; E1 = u[i]; E1p = u[1]; V1m = u[i+L-1]; V1 = u[i+L]; V1p = u[L+1]; ε1m = u[i+2L-1]; ε1 = u[i+2L]; ε1p = u[2L+1]; v1m = u[i+3L-1]; v1 = u[i+3L]; v1p = u[3L+1];
        else
            E0m = u0[i-1]; E0 = u0[i]; E0p = u0[i+1]; V0m = u0[i+L-1]; V0 = u0[i+L]; V0p = u0[i+L+1]; ε0m = u0[i+2L-1]; ε0 = u0[i+2L]; ε0p = u0[i+2L+1]; v0m = u0[i+3L-1]; v0 = u0[i+3L]; v0p = u0[i+3L+1];
            E1m = u[i-1]; E1 = u[i]; E1p = u[i+1]; V1m = u[i+L-1]; V1 = u[i+L]; V1p = u[i+L+1]; ε1m = u[i+2L-1]; ε1 = u[i+2L]; ε1p = u[i+2L+1]; v1m = u[i+3L-1]; v1 = u[i+3L]; v1p = u[i+3L+1];
        end


        # println("i=$i")
        F[i] = BDNKJacobian.F1(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
        F[i+L] = BDNKJacobian.F2(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p,η0, λ0, χ0)
        F[i+2*L] = BDNKJacobian.F3(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p,η0, λ0, χ0)
        F[i+3*L] = BDNKJacobian.F4(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p,η0, λ0, χ0)
    end
end

function Jacobian_F1_V(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function1.Jacobian_F1_V_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i - 1
        return BDNKJacobian.Function1.Jacobian_F1_V_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i + 1
        return BDNKJacobian.Function1.Jacobian_F1_V_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif i == 1 && j == L
        return BDNKJacobian.Function1.Jacobian_F1_V_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif i == L && j == 1
        return BDNKJacobian.Function1.Jacobian_F1_V_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F1_E(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function1.Jacobian_F1_E_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i - 1
        return BDNKJacobian.Function1.Jacobian_F1_E_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i + 1
        return BDNKJacobian.Function1.Jacobian_F1_E_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif i == 1 && j == L
        return BDNKJacobian.Function1.Jacobian_F1_E_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif i == L && j == 1
        return BDNKJacobian.Function1.Jacobian_F1_E_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F1_v(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function1.Jacobian_F1_v_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i - 1
        return BDNKJacobian.Function1.Jacobian_F1_v_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i + 1
        return BDNKJacobian.Function1.Jacobian_F1_v_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif i == 1 && j == L
        return BDNKJacobian.Function1.Jacobian_F1_v_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif i == L && j == 1
        return BDNKJacobian.Function1.Jacobian_F1_v_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F1_eps(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function1.Jacobian_F1_eps_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i - 1
        return BDNKJacobian.Function1.Jacobian_F1_eps_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i + 1
        return BDNKJacobian.Function1.Jacobian_F1_eps_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif i == 1 && j == L
        return BDNKJacobian.Function1.Jacobian_F1_eps_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif i == L && j == 1
        return BDNKJacobian.Function1.Jacobian_F1_eps_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F2_V(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function2.Jacobian_F2_V_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i - 1
        return BDNKJacobian.Function2.Jacobian_F2_V_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i + 1
        return BDNKJacobian.Function2.Jacobian_F2_V_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif i == 1 && j == L
        return BDNKJacobian.Function2.Jacobian_F2_V_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif i == L && j == 1
        return BDNKJacobian.Function2.Jacobian_F2_V_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F2_E(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function2.Jacobian_F2_E_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i - 1
        return BDNKJacobian.Function2.Jacobian_F2_E_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i + 1
        return BDNKJacobian.Function2.Jacobian_F2_E_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif i == 1 && j == L
        return BDNKJacobian.Function2.Jacobian_F2_E_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif i == L && j == 1
        return BDNKJacobian.Function2.Jacobian_F2_E_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F2_v(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function2.Jacobian_F2_v_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i - 1
        return BDNKJacobian.Function2.Jacobian_F2_v_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i + 1
        return BDNKJacobian.Function2.Jacobian_F2_v_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif i == 1 && j == L
        return BDNKJacobian.Function2.Jacobian_F2_v_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif i == L && j == 1
        return BDNKJacobian.Function2.Jacobian_F2_v_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F2_eps(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function2.Jacobian_F2_eps_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i - 1
        return BDNKJacobian.Function2.Jacobian_F2_eps_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i + 1
        return BDNKJacobian.Function2.Jacobian_F2_eps_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif i == 1 && j == L
        return BDNKJacobian.Function2.Jacobian_F2_eps_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif i == L && j == 1
        return BDNKJacobian.Function2.Jacobian_F2_eps_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F3_E(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function3.Jacobian_F3_E_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F3_eps(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)

    if j == i
        return BDNKJacobian.Function3.Jacobian_F3_eps_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F4_V(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function4.Jacobian_F4_V_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F4_v(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function4.Jacobian_F4_v_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

# for now use separate loops but eventually merge
function solve!(L::Int, k::Float64, h::Float64, ε0::Vector{Float64}, εdot0::Vector{Float64}, v0::Vector{Float64}, vdot0::Vector{Float64}, η0::Float64, λ0::Float64, χ0::Float64, kreiss_coef::Float64, xmax::Float64, total_time::Float64, target_tol::Float64, max_tol::Float64, max_iter::Int, min_iter::Int, path::String)
    
    # allocate memory
    newton_iterates = zeros(4 * L)  # \dot{v}, \dot{ε}, v, ε
    newton_iterates0 = zeros(4 * L)  # \dot{v}, \dot{ε}, v, ε
    function_vector = zeros(4 * L)
    En_KO = zeros(L)
    Vn_KO = zeros(L)
    εn_KO = zeros(L)
    vn_KO = zeros(L)
    En = zeros(L)
    Vn = zeros(L)
    εn = zeros(L)
    vn = zeros(L)

    # initialize sparse jacobiam and get mask for Jacobian update
    row_indices, column_indices = get_jacobian_sparsity(L);
    vals = ones(length(row_indices))
    sparse_jacobian = sparse(row_indices, column_indices, vals, 4 * L, 4 * L)
    JVector = zeros(length(row_indices))
    mask = BDNKSystem.generate_update_mask(row_indices, column_indices, 4 * L, 4 * L);

    # set up grid
    N = Int(ceil(total_time / k))
    data = zeros(4, N, L)
    newton_residuals = zeros(N)
    num_iterations = zeros(N)

    # spatial derivatives
    eps_prime_1 = zeros(N, L)
    eps_prime_2 = zeros(N, L)
    eps_prime_3 = zeros(N, L)
    v_prime_1 = zeros(N, L)
    v_prime_2 = zeros(N, L)
    v_prime_3 = zeros(N, L)

    data[1, 1, :] = εdot0[:]
    data[2, 1, :] = vdot0[:]
    data[3, 1, :] = ε0[:]
    data[4, 1, :] = v0[:]

    # residual diagnostics
    resid_t_component = zeros(N-2, L)
    resid_x_component = zeros(N-2, L)
    sim_diagnostics = zeros(5, N, L)

    stop = false
    resid = NaN
    n_iter = 0
    spacing = 2
    
    try
        for n = 1:N-1
            print_string = "Time Level: $n, Time: $(round(k * n; digits=4)), Completion: $(round(100 * n/(N-1); digits=5))%, Latest resid: $resid, Latest number of iterations: $(n_iter)   \r" 
            print(print_string)

            # add KO dissipation at most retarded time level
            En_KO[:] = data[1, n, :][:]
            Vn_KO[:] = data[2, n, :][:]
            εn_KO[:] = data[3, n, :][:]
            vn_KO[:] = data[4, n, :][:]

            for i=1:L
                En_KO[i] += -kreiss_oliger(data, L, 1, n, i, kreiss_coef) - kreiss_oliger_low_wavelength(data, L, 1, n, i, kreiss_coef, spacing)
                Vn_KO[i] += -kreiss_oliger(data, L, 2, n, i, kreiss_coef) - kreiss_oliger_low_wavelength(data, L, 2, n, i, kreiss_coef, spacing)
                εn_KO[i] += -kreiss_oliger(data, L, 3, n, i, kreiss_coef) - kreiss_oliger_low_wavelength(data, L, 3, n, i, kreiss_coef, spacing)
                vn_KO[i] += -kreiss_oliger(data, L, 4, n, i, kreiss_coef) - kreiss_oliger_low_wavelength(data, L, 4, n, i, kreiss_coef, spacing)
            end

            # vdot, εdot
            newton_iterates0[1:L] = En_KO
            newton_iterates0[L+1:2L] = Vn_KO
            newton_iterates0[2L+1:3L] = εn_KO
            newton_iterates0[3L+1:end] = vn_KO

            # proceed with Newton iteration
            En[:] = data[1, n, :][:]
            Vn[:] = data[2, n, :][:]
            εn[:] = data[3, n, :][:]
            vn[:] = data[4, n, :][:]
            
            # vdot, εdot
            newton_iterates[1:L] = En
            newton_iterates[L+1:2L] = Vn
            newton_iterates[2L+1:3L] = εn
            newton_iterates[3L+1:end] = vn
            compute_function_vector!(function_vector, newton_iterates, newton_iterates0, k, h, L, η0, λ0, χ0)

            n_iter = 0
            for step = 1:max_iter
                n_iter += 1
                compute_sparse_BDNK_jacobian!(sparse_jacobian, JVector, mask, newton_iterates, newton_iterates0, k, h, L, η0, λ0, χ0)

                Δ = sparse_jacobian \ (-function_vector)
                newton_iterates += Δ

                compute_function_vector!(function_vector, newton_iterates, newton_iterates0, k, h, L, η0, λ0, χ0)
        
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
            if stop
                break
            end

            num_iterations[n] = n_iter
            newton_residuals[n] = resid

            # println("Updating data")
            data[1, n+1, :] = newton_iterates[1:L]
            data[2, n+1, :] = newton_iterates[L+1:2L]
            data[3, n+1, :] = newton_iterates[2L+1:3L]
            data[4, n+1, :] = newton_iterates[3L+1:end]

            # compute independent residual
            if n == 1
                @inbounds for j=1:L
                    # compute spatial derivatives
                    @views v_prime_1[n, j] = FiniteDiffOrder5.compute_first_derivative(j,  data[4, n, :], h, L)
                    @views v_prime_2[n, j] = FiniteDiffOrder5.compute_second_derivative(j,  data[4, n, :], h, L)
                    @views v_prime_3[n, j] = FiniteDiffOrder5.compute_third_derivative(j,  data[4, n, :], h, L)
                    @views eps_prime_1[n, j] = FiniteDiffOrder5.compute_first_derivative(j,  data[3, n, :], h, L)
                    @views eps_prime_2[n, j] = FiniteDiffOrder5.compute_second_derivative(j,  data[3, n, :], h, L)
                    @views eps_prime_3[n, j] = FiniteDiffOrder5.compute_third_derivative(j,  data[3, n, :], h, L)

                    if j == 1
                        v1m = data[4, n, end]; v1 = data[4, n, j]; v1p = data[4, n, j+1]; ε1m = data[3, n, end]; ε1 = data[3, n, j]; ε1p = data[3, n, j+1];
                        V1m = data[2, n, end]; V1 = data[2, n, j]; V1p = data[2, n, j+1]; E1m = data[1, n, end]; E1 = data[1, n, j]; E1p = data[1, n, j+1];
                    elseif j==L
                        v1m = data[4, n, j-1]; v1 = data[4, n, j]; v1p = data[4, n, 1]; ε1m = data[3, n, j-1]; ε1 = data[3, n, j]; ε1p = data[3, n, 1];
                        V1m = data[2, n, j-1]; V1 = data[2, n, j]; V1p = data[2, n, 1]; E1m = data[1, n, j-1]; E1 = data[1, n, j]; E1p = data[1, n, 1];
                    else
                        v1m = data[4, n, j-1]; v1 = data[4, n, j]; v1p = data[4, n, j+1]; ε1m = data[3, n, j-1]; ε1 = data[3, n, j]; ε1p = data[3, n, j+1];
                        V1m = data[2, n, j-1]; V1 = data[2, n, j]; V1p = data[2, n, j+1]; E1m = data[1, n, j-1]; E1 = data[1, n, j]; E1p = data[1, n, j+1];
                    end
            
                    
                    sim_diagnostics[1, n, j] = BDNKConformalTensor.tt_zeroth_order(v1, ε1)
                    sim_diagnostics[2, n, j] = BDNKConformalTensor.xx_zeroth_order(v1, ε1)
                    sim_diagnostics[3, n, j] = BDNKConformalTensor.tt_first_order(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
                    sim_diagnostics[4, n, j] = BDNKConformalTensor.xx_first_order(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
                    sim_diagnostics[5, n, j] = BDNKConformalTensor.four_velocity_inner_prod(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
                end
            elseif n > 1
                for j=1:L
                    # compute spatial derivatives
                    @views v_prime_1[n, j] = FiniteDiffOrder5.compute_first_derivative(j,  data[4, n, :], h, L)
                    @views v_prime_2[n, j] = FiniteDiffOrder5.compute_second_derivative(j,  data[4, n, :], h, L)
                    @views v_prime_3[n, j] = FiniteDiffOrder5.compute_third_derivative(j,  data[4, n, :], h, L)
                    @views eps_prime_1[n, j] = FiniteDiffOrder5.compute_first_derivative(j,  data[3, n, :], h, L)
                    @views eps_prime_2[n, j] = FiniteDiffOrder5.compute_second_derivative(j,  data[3, n, :], h, L)
                    @views eps_prime_3[n, j] = FiniteDiffOrder5.compute_third_derivative(j,  data[3, n, :], h, L)

                    v0 = data[4, n-1, j]; ε0 = data[3, n-1, j];
                    V0 = data[2, n-1, j]; E0 = data[1, n-1, j];
                    V2 = data[2, n+1, j]; E2 = data[1, n+1, j];
                    if j == 1
                        v1m = data[4, n, end]; v1 = data[4, n, j]; v1p = data[4, n, j+1]; ε1m = data[3, n, end]; ε1 = data[3, n, j]; ε1p = data[3, n, j+1];
                        V1m = data[2, n, end]; V1 = data[2, n, j]; V1p = data[2, n, j+1]; E1m = data[1, n, end]; E1 = data[1, n, j]; E1p = data[1, n, j+1];
                    elseif j==L
                        v1m = data[4, n, j-1]; v1 = data[4, n, j]; v1p = data[4, n, 1]; ε1m = data[3, n, j-1]; ε1 = data[3, n, j]; ε1p = data[3, n, 1];
                        V1m = data[2, n, j-1]; V1 = data[2, n, j]; V1p = data[2, n, 1]; E1m = data[1, n, j-1]; E1 = data[1, n, j]; E1p = data[1, n, 1];
                    else
                        v1m = data[4, n, j-1]; v1 = data[4, n, j]; v1p = data[4, n, j+1]; ε1m = data[3, n, j-1]; ε1 = data[3, n, j]; ε1p = data[3, n, j+1];
                        V1m = data[2, n, j-1]; V1 = data[2, n, j]; V1p = data[2, n, j+1]; E1m = data[1, n, j-1]; E1 = data[1, n, j]; E1p = data[1, n, j+1];
                    end
                    
                    resid_t_component[n-1, j] = abs(StressEnergyGradients.BDNK.stress_energy_gradient_t(v1m, v1, v1p, ε1m, ε1, ε1p, V0, V1m, V1, V1p, E0, E1m, E1, E1p, V2, E2, k, h, η0, λ0, χ0))
                    resid_x_component[n-1, j] = abs(StressEnergyGradients.BDNK.stress_energy_gradient_x(v1m, v1, v1p, ε1m, ε1, ε1p, V0, V1m, V1, V1p, E0, E1m, E1, E1p, V2, E2, k, h, η0, λ0, χ0))

                    sim_diagnostics[1, n, j] = BDNKConformalTensor.tt_zeroth_order(v1, ε1)
                    sim_diagnostics[2, n, j] = BDNKConformalTensor.xx_zeroth_order(v1, ε1)
                    sim_diagnostics[3, n, j] = BDNKConformalTensor.tt_first_order(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
                    sim_diagnostics[4, n, j] = BDNKConformalTensor.xx_first_order(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
                    sim_diagnostics[5, n, j] = BDNKConformalTensor.four_velocity_inner_prod(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
                end
            end
        end
    catch e
        println("\nERROR OCCURRED. SAVING SOLUTION AND RETRHOWING ERROR.\n")
        save_sol(data, sim_diagnostics, newton_residuals, resid_t_component, resid_x_component, v_prime_1, v_prime_2, v_prime_3, eps_prime_1, eps_prime_2, eps_prime_3,
            num_iterations, L, k, h, η0, λ0, χ0, kreiss_coef, xmax, total_time, path)
        rethrow(e)
    end

    @inbounds for j=1:L
        # compute spatial derivatives
        @views v_prime_1[N, j] = FiniteDiffOrder5.compute_first_derivative(j,  data[4, N, :], h, L)
        @views v_prime_2[N, j] = FiniteDiffOrder5.compute_second_derivative(j,  data[4, N, :], h, L)
        @views v_prime_3[N, j] = FiniteDiffOrder5.compute_third_derivative(j,  data[4, N, :], h, L)
        @views eps_prime_1[N, j] = FiniteDiffOrder5.compute_first_derivative(j,  data[3, N, :], h, L)
        @views eps_prime_2[N, j] = FiniteDiffOrder5.compute_second_derivative(j,  data[3, N, :], h, L)
        @views eps_prime_3[N, j] = FiniteDiffOrder5.compute_third_derivative(j,  data[3, N, :], h, L)

        if j == 1
            v1m = data[4, N, end]; v1 = data[4, N, j]; v1p = data[4, N, j+1]; ε1m = data[3, N, end]; ε1 = data[3, N, j]; ε1p = data[3, N, j+1];
            V1m = data[2, N, end]; V1 = data[2, N, j]; V1p = data[2, N, j+1]; E1m = data[1, N, end]; E1 = data[1, N, j]; E1p = data[1, N, j+1];
        elseif j==L
            v1m = data[4, N, j-1]; v1 = data[4, N, j]; v1p = data[4, N, 1]; ε1m = data[3, N, j-1]; ε1 = data[3, N, j]; ε1p = data[3, N, 1];
            V1m = data[2, N, j-1]; V1 = data[2, N, j]; V1p = data[2, N, 1]; E1m = data[1, N, j-1]; E1 = data[1, N, j]; E1p = data[1, N, 1];
        else
            v1m = data[4, N, j-1]; v1 = data[4, N, j]; v1p = data[4, N, j+1]; ε1m = data[3, N, j-1]; ε1 = data[3, N, j]; ε1p = data[3, N, j+1];
            V1m = data[2, N, j-1]; V1 = data[2, N, j]; V1p = data[2, N, j+1]; E1m = data[1, N, j-1]; E1 = data[1, N, j]; E1p = data[1, N, j+1];
        end

        
        sim_diagnostics[1, N, j] = BDNKConformalTensor.tt_zeroth_order(v1, ε1)
        sim_diagnostics[2, N, j] = BDNKConformalTensor.xx_zeroth_order(v1, ε1)
        sim_diagnostics[3, N, j] = BDNKConformalTensor.tt_first_order(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
        sim_diagnostics[4, N, j] = BDNKConformalTensor.xx_first_order(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
        sim_diagnostics[5, N, j] = BDNKConformalTensor.four_velocity_inner_prod(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
    end

    save_sol(data, sim_diagnostics, newton_residuals, resid_t_component, resid_x_component, v_prime_1, v_prime_2, v_prime_3, eps_prime_1, eps_prime_2, eps_prime_3,
    num_iterations, L, k, h, η0, λ0, χ0, kreiss_coef, xmax, total_time, path)
end

end

module Aperiodic
using ..BDNKSystem
using ...BDNKJacobian
using ...BDNKConformalTensor
using ...StressEnergyGradients
using ...FiniteDiffOrder5
using LinearAlgebra
using Printf
using SparseArrays
using HDF5

sol_fname(k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, kreiss_coef::Float64, xmax::Float64, total_time::Float64, path::String) = path * "BDNK_aperiodic_conformal_sol_k_$(k)_h_$(h)_eta0_$(η0)_lambda0_$(λ0)_chi0_$(χ0)_kreiss_$(kreiss_coef)_xmax_$(xmax)_T_$(total_time).h5"

function save_sol(data::AbstractArray, diagnostics::AbstractArray, residuals::AbstractArray, ind_resid_t::AbstractArray, ind_resid_x::AbstractArray, v_prime_1::AbstractArray, v_prime_2::AbstractArray, v_prime_3::AbstractArray, eps_prime_1::AbstractArray,
    eps_prime_2::AbstractArray, eps_prime_3::AbstractArray, num_iterations::AbstractArray, L::Int, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, kreiss_coef::Float64, xmax::Float64, total_time::Float64, path::String)
    fname = sol_fname(k, h, η0, λ0, χ0, kreiss_coef, xmax,  total_time, path)
    h5open(fname, "w") do file
        file["t"] = range(start = 0.0, step = k, length = size(data, 2)) |> collect
        file["x"] = range(start = -xmax, stop = xmax, length = L) |> collect
        file["v"] = data[4, :, :]
        file["eps"] = data[3, :, :]
        file["vdot"] = data[2, :, :]
        file["epsdot"] = data[1, :, :]
        file["v_prime_1"] = v_prime_1
        file["v_prime_2"] = v_prime_2
        file["v_prime_3"] = v_prime_3
        file["eps_prime_1"] = eps_prime_1
        file["eps_prime_2"] = eps_prime_2
        file["eps_prime_3"] = eps_prime_3
        file["T_tt_0"] = diagnostics[1, :, :]
        file["T_tt_1"] = diagnostics[3, :, :]
        file["T_xx_0"] = diagnostics[2, :, :]
        file["T_xx_1"] = diagnostics[4, :, :]
        file["vel_dot_T"] = diagnostics[5, :, :]    # inner product u_{a} u_{b} T^{ab} to check weak energy condition
        file["residuals"] = residuals
        file["ind_resid_t"] = ind_resid_t
        file["ind_resid_x"] = ind_resid_x
        file["num_iterations"] = num_iterations
    end
    println("File created: " * fname)
end

# the jacobain matrix is a block matrix with 16 blocks. The first L rows corresponds to the crank nicholson equation for \dot{ε} evaluate at the L grid points. The next L rows correspond to the equation for \dot{v}, the next L for ε and the final L for v. The first L columns correspond to derivatives with respect to \dot{ε} at the advanced time level (n+1) at the L grid points, the next L columns on \dot{v} at n+1, the next L for ε at n+1 and the final L for v at n+1. Note that the crank nicholson equations for ε and v are trivial as a result of the time reduction E==\dot{ε} ==> \partial_{t}(ε) = E.
function get_jacobian_sparsity(L::Int)
    row_indices::Vector{Int} = []
    column_indices::Vector{Int} = []
    @inbounds for i in 1:L
        # since we include Kreiss-Oliger dissipation in the implicit equations, each block in the jacobian will include up to 5 nontrivial entries per row. We must implement special cases for rows 1, 2, L-1, and L when the length-5 stencils will wrap around due to the periodic boundary conditions
        if i == 1
            # first L rows correspond to the crank nicholson equation for \dot{ε} with kreiss oliger dissipation for \dot{ε}. The first L columns will thus contain 5 nontrivial entries per row, while the remaining columns  (3 blocks) will have 3 nontrivial entries per row.
            # include nonzero elements in first block diagional (idx = 1 to idx = L)  
            push!(row_indices, i)
            push!(column_indices, i)

            push!(row_indices, i)
            push!(column_indices, i+1)


            # include nonzero elements in second block diagonal (idx = L+1 to idx = 2L - 1)  
            push!(row_indices, i)
            push!(column_indices, i+L)

            push!(row_indices, i)
            push!(column_indices, i+L+1)

            # include nonzero elements in third block diagonal (idx = 2L+1 to idx = 3L - 1)  
            push!(row_indices, i)
            push!(column_indices, i+2*L)

            push!(row_indices, i)
            push!(column_indices, i+2*L+1)

            # include nonzero elements in fourth block diagonal (idx = 3L+1 to idx = 4L)  
            push!(row_indices, i)
            push!(column_indices, i+3*L)

            push!(row_indices, i)
            push!(column_indices, i+3*L+1)

            # rows L + 1 to 2L — CN equation for \dot{v} with Kreiss-Oliger dissipation. Blocks 1, 3, 4 will have 3 nontrivial entries, block 2 will have 5 nontrivial entries.
            # include nonzero elements in first block diagional (idx = 1 to idx = L)  
            push!(row_indices, i+L)
            push!(column_indices, i)

            push!(row_indices, i+L)
            push!(column_indices, i+1)

            # include nonzero elements in second block diagonal (idx = L+1 to idx = 2L - 1)  
            push!(row_indices, i+L)
            push!(column_indices, i+L)

            push!(row_indices, i+L)
            push!(column_indices, i+L+1)

            # include nonzero elements in third block diagonal (idx = 2L+1 to idx = 3L - 1)  
            push!(row_indices, i+L)
            push!(column_indices, i+2*L)

            push!(row_indices, i+L)
            push!(column_indices, i+2*L+1)

            # include nonzero elements in fourth block diagonal (idx = 3L+1 to idx = 4L)  
            push!(row_indices, i+L)
            push!(column_indices, i+3*L)

            push!(row_indices, i+L)
            push!(column_indices, i+3*L+1)

            # rows 2L+1 to 3L — CN equation for ε with Kreiss-Oliger dissipation. Equation is independent of \dot{v} and v, so blocks 2 and 4 will be identically zero. Block 1 will have one nontrivial entry due to the trivial CN equation for ε, while block 3 will have 5 nontrivial entries due to the KO dissipation
            push!(row_indices, i+2*L)
            push!(column_indices, i)

            push!(row_indices, i+2*L)
            push!(column_indices, i+2*L)

            # rows 3L+1 to 4L  — CN equation for v with Kreiss-Oliger dissipation. Equation is independent of \dot{ε} and ε, so blocks 1 and 3 will be identically zero. Block 2 will have one nontrivial entry due to the trivial CN equation for v, while block 4 will have 5 nontrivial entries due to the KO dissipation
            push!(row_indices, i+3*L)
            push!(column_indices, i+L)

            push!(row_indices, i+3*L)
            push!(column_indices, i+3*L)

            # no nonzero elements in fourth block diagonal (idx = 3L+1 to idx = 4L - 4) since ∂F4/∂ε = 0

        elseif i==L
            # first L rows
            # include nonzero elements in first block diagional (idx = 1 to idx = L)  

            push!(row_indices, i)
            push!(column_indices, i-1)

            push!(row_indices, i)
            push!(column_indices, i)

            # include nonzero elements in second block diagonal (idx = L+1 to idx = 2L - 1)  
            push!(row_indices, i)
            push!(column_indices, i+L-1)

            push!(row_indices, i)
            push!(column_indices, i+L)

            # include nonzero elements in third block diagonal (idx = 2L+1 to idx = 3L - 1)  
            push!(row_indices, i)
            push!(column_indices, i+2*L-1)

            push!(row_indices, i)
            push!(column_indices, i+2*L)

            # include nonzero elements in fourth block diagonal (idx = 3L+1 to idx = 4L)  
            push!(row_indices, i)
            push!(column_indices, i+3*L-1)

            push!(row_indices, i)
            push!(column_indices, i+3*L)

            # rows L + 1 to 2L
            # include nonzero elements in first block diagional (idx = 1 to idx = L)  
            push!(row_indices, i+L)
            push!(column_indices, i-1)

            push!(row_indices, i+L)
            push!(column_indices, i)

            # include nonzero elements in second block diagonal (idx = L+1 to idx = 2L - 1)  

            push!(row_indices, i+L)
            push!(column_indices, i+L-1)

            push!(row_indices, i+L)
            push!(column_indices, i+L)

            # include nonzero elements in third block diagonal (idx = 2L+1 to idx = 3L - 1)  
            
            push!(row_indices, i+L)
            push!(column_indices, i+2*L-1)

            push!(row_indices, i+L)
            push!(column_indices, i+2*L)

            # include nonzero elements in fourth block diagonal (idx = 3L+1 to idx = 4L)  
            
            push!(row_indices, i+L)
            push!(column_indices, i+3*L-1)

            push!(row_indices, i+L)
            push!(column_indices, i+3*L)

            # rows 2L-1 to 3L - 3
            # no nonzero elements in first block diagional (idx = 1 to idx = L) since ∂F3/∂V = 0
            # include nonzero elements in second block diagonal (idx = L+1 to idx = 2L - 1)  
            push!(row_indices, i+2*L)
            push!(column_indices, i)

            push!(row_indices, i+2*L)
            push!(column_indices, i+2*L)

            # rows 3L+1 to 4L
            # include nonzero elements in first block diagional (idx = 1 to idx = L)  
            push!(row_indices, i+3*L)
            push!(column_indices, i+L)

            push!(row_indices, i+3*L)
            push!(column_indices, i+3*L)

            # no nonzero elements in fourth block diagonal (idx = 3L+1 to idx = 4L - 4) since ∂F4/∂ε = 0
        else
            # outside boundaries of each of the four blocks, it is tridiagonal
            # first L rows
            # include nonzero elements in first block diagional (idx = 1 to idx = L)  
            push!(row_indices, i)
            push!(column_indices, i-1)

            push!(row_indices, i)
            push!(column_indices, i)

            push!(row_indices, i)
            push!(column_indices, i+1)

            # include nonzero elements in second block diagonal (idx = L+1 to idx = 2L - 1)  
            push!(row_indices, i)
            push!(column_indices, i+L-1)

            push!(row_indices, i)
            push!(column_indices, i+L)

            push!(row_indices, i)
            push!(column_indices, i+L+1)

            # include nonzero elements in third block diagonal (idx = 2L+1 to idx = 3L - 1)  
            push!(row_indices, i)
            push!(column_indices, i+2*L-1)

            push!(row_indices, i)
            push!(column_indices, i+2*L)

            push!(row_indices, i)
            push!(column_indices, i+2*L+1)

            # include nonzero elements in fourth block diagonal (idx = 2L+1 to idx = 3L - 1)  
            push!(row_indices, i)
            push!(column_indices, i+3*L-1)

            push!(row_indices, i)
            push!(column_indices, i+3*L)

            push!(row_indices, i)
            push!(column_indices, i+3*L+1)

            # row L to 2L - 2
            # include nonzero elements in first block diagional (idx = 1 to idx = L)  
            push!(row_indices, i+L)
            push!(column_indices, i-1)

            push!(row_indices, i+L)
            push!(column_indices, i)

            push!(row_indices, i+L)
            push!(column_indices, i+1)
            

            # include nonzero elements in second block diagonal (idx = L+1 to idx = 2L - 1)  
            push!(row_indices, i+L)
            push!(column_indices, i+L-1)

            push!(row_indices, i+L)
            push!(column_indices, i+L)

            push!(row_indices, i+L)
            push!(column_indices, i+L+1)

            # include nonzero elements in third block diagonal (idx = 2L+1 to idx = 3L - 1)  
            push!(row_indices, i+L)
            push!(column_indices, i+2*L-1)

            push!(row_indices, i+L)
            push!(column_indices, i+2*L)

            push!(row_indices, i+L)
            push!(column_indices, i+2*L+1)

            # include nonzero elements in fourth block diagonal (idx = 2L+1 to idx = 3L - 1)  
            push!(row_indices, i+L)
            push!(column_indices, i+3*L-1)

            push!(row_indices, i+L)
            push!(column_indices, i+3*L)

            push!(row_indices, i+L)
            push!(column_indices, i+3*L+1)

            # rows 2L-1 to 3L - 3

            push!(row_indices, i+2*L)
            push!(column_indices, i)

            push!(row_indices, i+2*L)
            push!(column_indices, i+2*L)

            # rows 3L+1 to 4L
            push!(row_indices, i+3*L)
            push!(column_indices, i+L)

            push!(row_indices, i+3*L)
            push!(column_indices, i+3*L)
        end
    end
    return row_indices, column_indices
end

# u is current newton iterate of length 4L-4 (i.e., all the variables Vk, Ek, vk, εk evaluated at 1,..., L --- ignore point L since it is the same as 1 due to periodic BCs)
function compute_sparse_BDNK_jacobian!(J::AbstractArray, JVector::Vector{Float64}, mask::Vector{Int}, u::AbstractArray, u0::AbstractArray, k::Float64, h::Float64, L::Int64, η0::Float64, λ0::Float64, χ0::Float64)
    idx = 1
    @inbounds for i=1:L
        # solution values at relevant points in a nearby stencil enforcing ghost cells at boundary (e.g., copying from nearest cell)
        if i == 1
            E0m = u0[i]; E0 = u0[i]; E0p = u0[i+1]; V0m = u0[i+L]; V0 = u0[i+L]; V0p = u0[i+L+1]; ε0m = u0[i+2L]; ε0 = u0[i+2L]; ε0p = u0[i+2L+1]; v0m = u0[i+3L]; v0 = u0[i+3L]; v0p = u0[i+3L+1];
            E1m = u[i]; E1 = u[i]; E1p = u[i+1]; V1m = u[i+L]; V1 = u[i+L]; V1p = u[i+L+1]; ε1m = u[i+2L]; ε1 = u[i+2L]; ε1p = u[i+2L+1]; v1m = u[i+3L]; v1 = u[i+3L]; v1p = u[i+3L+1];
        elseif i==L
            E0m = u0[i-1]; E0 = u0[i]; E0p = u0[i]; V0m = u0[i+L-1]; V0 = u0[i+L]; V0p = u0[i+L]; ε0m = u0[i+2L-1]; ε0 = u0[i+2L]; ε0p = u0[i+2L]; v0m = u0[i+3L-1]; v0 = u0[i+3L]; v0p = u0[i+3L];
            E1m = u[i-1]; E1 = u[i]; E1p = u[i]; V1m = u[i+L-1]; V1 = u[i+L]; V1p = u[i+L]; ε1m = u[i+2L-1]; ε1 = u[i+2L]; ε1p = u[i+2L]; v1m = u[i+3L-1]; v1 = u[i+3L]; v1p = u[i+3L];
        else
            E0m = u0[i-1]; E0 = u0[i]; E0p = u0[i+1]; V0m = u0[i+L-1]; V0 = u0[i+L]; V0p = u0[i+L+1]; ε0m = u0[i+2L-1]; ε0 = u0[i+2L]; ε0p = u0[i+2L+1]; v0m = u0[i+3L-1]; v0 = u0[i+3L]; v0p = u0[i+3L+1];
            E1m = u[i-1]; E1 = u[i]; E1p = u[i+1]; V1m = u[i+L-1]; V1 = u[i+L]; V1p = u[i+L+1]; ε1m = u[i+2L-1]; ε1 = u[i+2L]; ε1p = u[i+2L+1]; v1m = u[i+3L-1]; v1 = u[i+3L]; v1p = u[i+3L+1];
        end

        if i == 1
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_E(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_V(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_eps(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_v(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_E(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_V(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_eps(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_v(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F3_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F3_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F4_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F4_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
        elseif i == L
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_E(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_V(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_eps(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_v(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_E(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_V(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_eps(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_v(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F3_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F3_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F4_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F4_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
        else
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_E(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_E(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_V(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_V(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_eps(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_eps(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_v(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F1_v(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_E(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_E(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_V(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_V(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_eps(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_eps(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_v(L, k, h, η0, λ0, χ0, i, i-1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F2_v(L, k, h, η0, λ0, χ0, i, i+1, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F3_E(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F3_eps(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F4_V(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;
            JVector[idx] = BDNKSystem.Aperiodic.Jacobian_F4_v(L, k, h, η0, λ0, χ0, i, i, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p); idx += 1;

        end
    end
    J.nzval[mask] .= JVector
end


function data_array_access(u::AbstractArray, size::Int, var::Int, time::Int, i::Int)
    if i < 1
        return u[var, time, 1]
    elseif i > size
        return u[var, time, size]
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

function kreiss_oliger_low_wavelength(data::AbstractArray, size::Int, var::Int, time::Int, j::Int, coef::Float64, spacing::Int)
    u_minus_2 = data_array_access(data, size, var, time, j-2*spacing)
    u_minus_1 = data_array_access(data, size, var, time, j-1*spacing)
    u = data_array_access(data, size, var, time, j)
    u_plus_1 = data_array_access(data, size, var, time, j+1*spacing)
    u_plus_2 = data_array_access(data, size, var, time, j+2*spacing)
    return coef * (u_plus_2 - 4.0 * u_plus_1 + 6.0 * u - 4.0 * u_minus_1 + u_minus_2) / 16.0
end
# u denotes current newton iteration
function compute_function_vector!(F::AbstractArray, u::AbstractArray, u0::AbstractArray, k::Float64, h::Float64, L::Int64, η0::Float64, λ0::Float64, χ0::Float64)

    @inbounds for i=1:L
        # solution values at relevant points in a nearby stencil enforcing ghost cells at boundary (e.g., copying from nearest cell)
        if i == 1
            E0m = u0[i]; E0 = u0[i]; E0p = u0[i+1]; V0m = u0[i+L]; V0 = u0[i+L]; V0p = u0[i+L+1]; ε0m = u0[i+2L]; ε0 = u0[i+2L]; ε0p = u0[i+2L+1]; v0m = u0[i+3L]; v0 = u0[i+3L]; v0p = u0[i+3L+1];
            E1m = u[i]; E1 = u[i]; E1p = u[i+1]; V1m = u[i+L]; V1 = u[i+L]; V1p = u[i+L+1]; ε1m = u[i+2L]; ε1 = u[i+2L]; ε1p = u[i+2L+1]; v1m = u[i+3L]; v1 = u[i+3L]; v1p = u[i+3L+1];
        elseif i==L
            E0m = u0[i-1]; E0 = u0[i]; E0p = u0[i]; V0m = u0[i+L-1]; V0 = u0[i+L]; V0p = u0[i+L]; ε0m = u0[i+2L-1]; ε0 = u0[i+2L]; ε0p = u0[i+2L]; v0m = u0[i+3L-1]; v0 = u0[i+3L]; v0p = u0[i+3L];
            E1m = u[i-1]; E1 = u[i]; E1p = u[i]; V1m = u[i+L-1]; V1 = u[i+L]; V1p = u[i+L]; ε1m = u[i+2L-1]; ε1 = u[i+2L]; ε1p = u[i+2L]; v1m = u[i+3L-1]; v1 = u[i+3L]; v1p = u[i+3L];
        else
            E0m = u0[i-1]; E0 = u0[i]; E0p = u0[i+1]; V0m = u0[i+L-1]; V0 = u0[i+L]; V0p = u0[i+L+1]; ε0m = u0[i+2L-1]; ε0 = u0[i+2L]; ε0p = u0[i+2L+1]; v0m = u0[i+3L-1]; v0 = u0[i+3L]; v0p = u0[i+3L+1];
            E1m = u[i-1]; E1 = u[i]; E1p = u[i+1]; V1m = u[i+L-1]; V1 = u[i+L]; V1p = u[i+L+1]; ε1m = u[i+2L-1]; ε1 = u[i+2L]; ε1p = u[i+2L+1]; v1m = u[i+3L-1]; v1 = u[i+3L]; v1p = u[i+3L+1];
        end

        F[i] = BDNKJacobian.F1(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
        F[i+L] = BDNKJacobian.F2(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
        F[i+2*L] = BDNKJacobian.F3(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
        F[i+3*L] = BDNKJacobian.F4(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    end
end

function Jacobian_F1_V(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function1.Jacobian_F1_V_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i - 1
        return BDNKJacobian.Function1.Jacobian_F1_V_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i + 1
        return BDNKJacobian.Function1.Jacobian_F1_V_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F1_E(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function1.Jacobian_F1_E_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i - 1
        return BDNKJacobian.Function1.Jacobian_F1_E_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i + 1
        return BDNKJacobian.Function1.Jacobian_F1_E_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F1_v(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function1.Jacobian_F1_v_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i - 1
        return BDNKJacobian.Function1.Jacobian_F1_v_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i + 1
        return BDNKJacobian.Function1.Jacobian_F1_v_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F1_eps(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function1.Jacobian_F1_eps_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i - 1
        return BDNKJacobian.Function1.Jacobian_F1_eps_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i + 1
        return BDNKJacobian.Function1.Jacobian_F1_eps_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F2_V(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function2.Jacobian_F2_V_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i - 1
        return BDNKJacobian.Function2.Jacobian_F2_V_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i + 1
        return BDNKJacobian.Function2.Jacobian_F2_V_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F2_E(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function2.Jacobian_F2_E_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i - 1
        return BDNKJacobian.Function2.Jacobian_F2_E_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i + 1
        return BDNKJacobian.Function2.Jacobian_F2_E_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F2_v(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function2.Jacobian_F2_v_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i - 1
        return BDNKJacobian.Function2.Jacobian_F2_v_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i + 1
        return BDNKJacobian.Function2.Jacobian_F2_v_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F2_eps(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function2.Jacobian_F2_eps_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i - 1
        return BDNKJacobian.Function2.Jacobian_F2_eps_j_m(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    elseif j == i + 1
        return BDNKJacobian.Function2.Jacobian_F2_eps_j_p(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F3_E(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function3.Jacobian_F3_E_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F3_eps(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)

    if j == i
        return BDNKJacobian.Function3.Jacobian_F3_eps_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F4_V(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function4.Jacobian_F4_V_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

function Jacobian_F4_v(L::Int64, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, i::Int64, j::Int64, v0m::Float64, v0::Float64, v0p::Float64, ε0m::Float64, ε0::Float64, ε0p::Float64, V0m::Float64, V0::Float64, V0p::Float64, E0m::Float64, E0::Float64, E0p::Float64, v1m::Float64, v1::Float64, v1p::Float64, ε1m::Float64, ε1::Float64, ε1p::Float64, V1m::Float64, V1::Float64, V1p::Float64, E1m::Float64, E1::Float64, E1p::Float64)
    
    if j == i
        return BDNKJacobian.Function4.Jacobian_F4_v_j_j(k, h, v0m, v0, v0p, ε0m, ε0, ε0p, V0m, V0, V0p, E0m, E0, E0p, v1m, v1, v1p, ε1m, ε1, ε1p, V1m, V1, V1p, E1m, E1, E1p, η0, λ0, χ0)
    else
        throw("Error in jacobian construction. Evaluated at i = $i, j = $j")
    end
end

# for now use separate loops but eventually merge
function solve!(L::Int, k::Float64, h::Float64, ε0::Vector{Float64}, εdot0::Vector{Float64}, v0::Vector{Float64}, vdot0::Vector{Float64}, η0::Float64, λ0::Float64, χ0::Float64, kreiss_coef::Float64, xmax::Float64, total_time::Float64, target_tol::Float64, max_tol::Float64, max_iter::Int, min_iter::Int, path::String)
    
    # allocate memory
    newton_iterates = zeros(4 * L)  # \dot{v}, \dot{ε}, v, ε
    newton_iterates0 = zeros(4 * L)  # \dot{v}, \dot{ε}, v, ε
    function_vector = zeros(4 * L)
    En_KO = zeros(L)
    Vn_KO = zeros(L)
    εn_KO = zeros(L)
    vn_KO = zeros(L)
    En = zeros(L)
    Vn = zeros(L)
    εn = zeros(L)
    vn = zeros(L)

    # initialize sparse jacobiam and get mask for Jacobian update
    row_indices, column_indices = get_jacobian_sparsity(L);
    vals = ones(length(row_indices))
    sparse_jacobian = sparse(row_indices, column_indices, vals, 4 * L, 4 * L)
    JVector = zeros(length(row_indices))
    mask = BDNKSystem.generate_update_mask(row_indices, column_indices, 4 * L, 4 * L);

    # set up grid
    N = Int(ceil(total_time / k))
    data = zeros(4, N, L)
    newton_residuals = zeros(N)
    num_iterations = zeros(N)

    # spatial derivatives
    eps_prime_1 = zeros(N, L)
    eps_prime_2 = zeros(N, L)
    eps_prime_3 = zeros(N, L)
    v_prime_1 = zeros(N, L)
    v_prime_2 = zeros(N, L)
    v_prime_3 = zeros(N, L)

    data[1, 1, :] = εdot0[:]
    data[2, 1, :] = vdot0[:]
    data[3, 1, :] = ε0[:]
    data[4, 1, :] = v0[:]

    # residual diagnostics
    resid_t_component = zeros(N-2, L)
    resid_x_component = zeros(N-2, L)
    sim_diagnostics = zeros(5, N, L)

    stop = false
    resid = NaN
    n_iter = 0
    spacing = 2
    
    try
        for n = 1:N-1
            print_string = "Time Level: $n, Time: $(round(k * n; digits=4)), Completion: $(round(100 * n/(N-1); digits=5))%, Latest resid: $resid, Latest number of iterations: $(n_iter)   \r" 
            print(print_string)

            # add KO dissipation at most retarded time level
            En_KO[:] = data[1, n, :][:]
            Vn_KO[:] = data[2, n, :][:]
            εn_KO[:] = data[3, n, :][:]
            vn_KO[:] = data[4, n, :][:]

            for i=1:L
                En_KO[i] += -kreiss_oliger(data, L, 1, n, i, kreiss_coef) - kreiss_oliger_low_wavelength(data, L, 1, n, i, kreiss_coef, spacing)
                Vn_KO[i] += -kreiss_oliger(data, L, 2, n, i, kreiss_coef) - kreiss_oliger_low_wavelength(data, L, 2, n, i, kreiss_coef, spacing)
                εn_KO[i] += -kreiss_oliger(data, L, 3, n, i, kreiss_coef) - kreiss_oliger_low_wavelength(data, L, 3, n, i, kreiss_coef, spacing)
                vn_KO[i] += -kreiss_oliger(data, L, 4, n, i, kreiss_coef) - kreiss_oliger_low_wavelength(data, L, 4, n, i, kreiss_coef, spacing)
            end

            
            # vdot, εdot
            newton_iterates0[1:L] = En_KO
            newton_iterates0[L+1:2L] = Vn_KO
            newton_iterates0[2L+1:3L] = εn_KO
            newton_iterates0[3L+1:end] = vn_KO

            # proceed with Newton iteration
            En[:] = data[1, n, :][:]
            Vn[:] = data[2, n, :][:]
            εn[:] = data[3, n, :][:]
            vn[:] = data[4, n, :][:]
            
            # vdot, εdot
            newton_iterates[1:L] = En
            newton_iterates[L+1:2L] = Vn
            newton_iterates[2L+1:3L] = εn
            newton_iterates[3L+1:end] = vn
            compute_function_vector!(function_vector, newton_iterates, newton_iterates0, k, h, L, η0, λ0, χ0)

            n_iter = 0
            for step = 1:max_iter
                n_iter += 1
                compute_sparse_BDNK_jacobian!(sparse_jacobian, JVector, mask, newton_iterates, newton_iterates0, k, h, L, η0, λ0, χ0)

                Δ = sparse_jacobian \ (-function_vector)
                newton_iterates += Δ

                compute_function_vector!(function_vector, newton_iterates, newton_iterates0, k, h, L, η0, λ0, χ0)
        
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
            if stop
                break
            end

            num_iterations[n] = n_iter
            newton_residuals[n] = resid

            # println("Updating data")
            data[1, n+1, :] = newton_iterates[1:L]
            data[2, n+1, :] = newton_iterates[L+1:2L]
            data[3, n+1, :] = newton_iterates[2L+1:3L]
            data[4, n+1, :] = newton_iterates[3L+1:end]

            # compute independent residual
            if n == 1
                @inbounds for j=1:L
                    # compute spatial derivatives
                    @views v_prime_1[n, j] = FiniteDiffOrder5.compute_first_derivative(j,  data[4, n, :], h, L)
                    @views v_prime_2[n, j] = FiniteDiffOrder5.compute_second_derivative(j,  data[4, n, :], h, L)
                    @views v_prime_3[n, j] = FiniteDiffOrder5.compute_third_derivative(j,  data[4, n, :], h, L)
                    @views eps_prime_1[n, j] = FiniteDiffOrder5.compute_first_derivative(j,  data[3, n, :], h, L)
                    @views eps_prime_2[n, j] = FiniteDiffOrder5.compute_second_derivative(j,  data[3, n, :], h, L)
                    @views eps_prime_3[n, j] = FiniteDiffOrder5.compute_third_derivative(j,  data[3, n, :], h, L)

                    if j == 1
                        v1m = data[4, n, j]; v1 = data[4, n, j]; v1p = data[4, n, j+1]; ε1m = data[3, n, j]; ε1 = data[3, n, j]; ε1p = data[3, n, j+1];
                        V1m = data[2, n, j]; V1 = data[2, n, j]; V1p = data[2, n, j+1]; E1m = data[1, n, j]; E1 = data[1, n, j]; E1p = data[1, n, j+1];
                    elseif j==L
                        v1m = data[4, n, j-1]; v1 = data[4, n, j]; v1p = data[4, n, j]; ε1m = data[3, n, j-1]; ε1 = data[3, n, j]; ε1p = data[3, n, j];
                        V1m = data[2, n, j-1]; V1 = data[2, n, j]; V1p = data[2, n, j]; E1m = data[1, n, j-1]; E1 = data[1, n, j]; E1p = data[1, n, j];
                    else
                        v1m = data[4, n, j-1]; v1 = data[4, n, j]; v1p = data[4, n, j+1]; ε1m = data[3, n, j-1]; ε1 = data[3, n, j]; ε1p = data[3, n, j+1];
                        V1m = data[2, n, j-1]; V1 = data[2, n, j]; V1p = data[2, n, j+1]; E1m = data[1, n, j-1]; E1 = data[1, n, j]; E1p = data[1, n, j+1];
                    end
            
                    
                    sim_diagnostics[1, n, j] = BDNKConformalTensor.tt_zeroth_order(v1, ε1)
                    sim_diagnostics[2, n, j] = BDNKConformalTensor.xx_zeroth_order(v1, ε1)
                    sim_diagnostics[3, n, j] = BDNKConformalTensor.tt_first_order(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
                    sim_diagnostics[4, n, j] = BDNKConformalTensor.xx_first_order(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
                    sim_diagnostics[5, n, j] = BDNKConformalTensor.four_velocity_inner_prod(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
                end
            elseif n > 1
                for j=1:L
                    # compute spatial derivatives
                    @views v_prime_1[n, j] = FiniteDiffOrder5.compute_first_derivative(j,  data[4, n, :], h, L)
                    @views v_prime_2[n, j] = FiniteDiffOrder5.compute_second_derivative(j,  data[4, n, :], h, L)
                    @views v_prime_3[n, j] = FiniteDiffOrder5.compute_third_derivative(j,  data[4, n, :], h, L)
                    @views eps_prime_1[n, j] = FiniteDiffOrder5.compute_first_derivative(j,  data[3, n, :], h, L)
                    @views eps_prime_2[n, j] = FiniteDiffOrder5.compute_second_derivative(j,  data[3, n, :], h, L)
                    @views eps_prime_3[n, j] = FiniteDiffOrder5.compute_third_derivative(j,  data[3, n, :], h, L)

                    v0 = data[4, n-1, j]; ε0 = data[3, n-1, j];
                    V0 = data[2, n-1, j]; E0 = data[1, n-1, j];
                    V2 = data[2, n+1, j]; E2 = data[1, n+1, j];
                    if j == 1
                        v1m = data[4, n, j]; v1 = data[4, n, j]; v1p = data[4, n, j+1]; ε1m = data[3, n, j]; ε1 = data[3, n, j]; ε1p = data[3, n, j+1];
                        V1m = data[2, n, j]; V1 = data[2, n, j]; V1p = data[2, n, j+1]; E1m = data[1, n, j]; E1 = data[1, n, j]; E1p = data[1, n, j+1];
                    elseif j==L
                        v1m = data[4, n, j-1]; v1 = data[4, n, j]; v1p = data[4, n, j]; ε1m = data[3, n, j-1]; ε1 = data[3, n, j]; ε1p = data[3, n, j];
                        V1m = data[2, n, j-1]; V1 = data[2, n, j]; V1p = data[2, n, j]; E1m = data[1, n, j-1]; E1 = data[1, n, j]; E1p = data[1, n, j];
                    else
                        v1m = data[4, n, j-1]; v1 = data[4, n, j]; v1p = data[4, n, j+1]; ε1m = data[3, n, j-1]; ε1 = data[3, n, j]; ε1p = data[3, n, j+1];
                        V1m = data[2, n, j-1]; V1 = data[2, n, j]; V1p = data[2, n, j+1]; E1m = data[1, n, j-1]; E1 = data[1, n, j]; E1p = data[1, n, j+1];
                    end
                    
                    resid_t_component[n-1, j] = abs(StressEnergyGradients.BDNK.stress_energy_gradient_t(v1m, v1, v1p, ε1m, ε1, ε1p, V0, V1m, V1, V1p, E0, E1m, E1, E1p, V2, E2, k, h, η0, λ0, χ0))
                    resid_x_component[n-1, j] = abs(StressEnergyGradients.BDNK.stress_energy_gradient_x(v1m, v1, v1p, ε1m, ε1, ε1p, V0, V1m, V1, V1p, E0, E1m, E1, E1p, V2, E2, k, h, η0, λ0, χ0))

                    sim_diagnostics[1, n, j] = BDNKConformalTensor.tt_zeroth_order(v1, ε1)
                    sim_diagnostics[2, n, j] = BDNKConformalTensor.xx_zeroth_order(v1, ε1)
                    sim_diagnostics[3, n, j] = BDNKConformalTensor.tt_first_order(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
                    sim_diagnostics[4, n, j] = BDNKConformalTensor.xx_first_order(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
                    sim_diagnostics[5, n, j] = BDNKConformalTensor.four_velocity_inner_prod(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
                end
            end
        end
    catch e
        println("\nERROR OCCURRED. SAVING SOLUTION AND RETRHOWING ERROR.\n")
        save_sol(data, sim_diagnostics, newton_residuals, resid_t_component, resid_x_component, v_prime_1, v_prime_2, v_prime_3, eps_prime_1, eps_prime_2, eps_prime_3,
            num_iterations, L, k, h, η0, λ0, χ0, kreiss_coef, xmax, total_time, path)
        rethrow(e)
    end

    @inbounds for j=1:L
        # compute spatial derivatives
        @views v_prime_1[N, j] = FiniteDiffOrder5.compute_first_derivative(j,  data[4, N, :], h, L)
        @views v_prime_2[N, j] = FiniteDiffOrder5.compute_second_derivative(j,  data[4, N, :], h, L)
        @views v_prime_3[N, j] = FiniteDiffOrder5.compute_third_derivative(j,  data[4, N, :], h, L)
        @views eps_prime_1[N, j] = FiniteDiffOrder5.compute_first_derivative(j,  data[3, N, :], h, L)
        @views eps_prime_2[N, j] = FiniteDiffOrder5.compute_second_derivative(j,  data[3, N, :], h, L)
        @views eps_prime_3[N, j] = FiniteDiffOrder5.compute_third_derivative(j,  data[3, N, :], h, L)

        if j == 1
            v1m = data[4, N, end]; v1 = data[4, N, j]; v1p = data[4, N, j+1]; ε1m = data[3, N, end]; ε1 = data[3, N, j]; ε1p = data[3, N, j+1];
            V1m = data[2, N, end]; V1 = data[2, N, j]; V1p = data[2, N, j+1]; E1m = data[1, N, end]; E1 = data[1, N, j]; E1p = data[1, N, j+1];
        elseif j==L
            v1m = data[4, N, j-1]; v1 = data[4, N, j]; v1p = data[4, N, 1]; ε1m = data[3, N, j-1]; ε1 = data[3, N, j]; ε1p = data[3, N, 1];
            V1m = data[2, N, j-1]; V1 = data[2, N, j]; V1p = data[2, N, 1]; E1m = data[1, N, j-1]; E1 = data[1, N, j]; E1p = data[1, N, 1];
        else
            v1m = data[4, N, j-1]; v1 = data[4, N, j]; v1p = data[4, N, j+1]; ε1m = data[3, N, j-1]; ε1 = data[3, N, j]; ε1p = data[3, N, j+1];
            V1m = data[2, N, j-1]; V1 = data[2, N, j]; V1p = data[2, N, j+1]; E1m = data[1, N, j-1]; E1 = data[1, N, j]; E1p = data[1, N, j+1];
        end

        
        sim_diagnostics[1, N, j] = BDNKConformalTensor.tt_zeroth_order(v1, ε1)
        sim_diagnostics[2, N, j] = BDNKConformalTensor.xx_zeroth_order(v1, ε1)
        sim_diagnostics[3, N, j] = BDNKConformalTensor.tt_first_order(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
        sim_diagnostics[4, N, j] = BDNKConformalTensor.xx_first_order(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
        sim_diagnostics[5, N, j] = BDNKConformalTensor.four_velocity_inner_prod(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
    end

    save_sol(data, sim_diagnostics, newton_residuals, resid_t_component, resid_x_component, v_prime_1, v_prime_2, v_prime_3, eps_prime_1, eps_prime_2, eps_prime_3,
    num_iterations, L, k, h, η0, λ0, χ0, kreiss_coef, xmax, total_time, path)
end

end

function solve!(type::String, L::Int, k::Float64, h::Float64, ε0::Vector{Float64}, εdot0::Vector{Float64}, v0::Vector{Float64}, vdot0::Vector{Float64}, η0::Float64, λ0::Float64, χ0::Float64, kreiss_coef::Float64, xmax::Float64, total_time::Float64, target_tol::Float64, max_tol::Float64, max_iter::Int, min_iter::Int, path::String)
    if isequal(type, "PERIODIC")
        BDNKSystem.Periodic.solve!(L, k, h, ε0, εdot0, v0, vdot0, η0, λ0, χ0, kreiss_coef, xmax, total_time, target_tol, max_tol, max_iter, min_iter, path);
    elseif isequal(type, "APERIODIC")
        BDNKSystem.Aperiodic.solve!(L, k, h, ε0, εdot0, v0, vdot0, η0, λ0, χ0, kreiss_coef, xmax, total_time, target_tol, max_tol, max_iter, min_iter, path);
    else
        throw("Invalid type. Choose either PERIODIC or APERIODIC.")
    end
end


end