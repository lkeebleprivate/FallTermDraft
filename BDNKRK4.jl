# include("../RK4BDNKConformalTensor.jl")
include("BDNKConformalTensor.jl")
include("RK4StressEnergyGradients.jl")
include("FiniteDiff.jl")
module BDNKRK4
using ...RK4StressEnergyGradients
using ...FiniteDiffOrder5
using ...BDNKConformalTensor
using LinearAlgebra
using Printf
using SparseArrays
using HDF5

one_norm(x::AbstractArray) = sum(abs, x) / length(x)

function load_sol(type::String, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, kreiss_coef::Float64, xmax::Float64, total_time::Float64, path::String)
    if isequal(type, "Periodic")
        fname = BDNKRK4.sol_fname(type, k, h, η0, λ0, χ0, kreiss_coef, xmax,  total_time, path)
    elseif isequal(type, "Ghost")
        fname = BDNKRK4.sol_fname(type, k, h, η0, λ0, χ0, kreiss_coef, xmax,  total_time, path)
    else
        throw("Invalid type. Choose either Periodic or Ghost.")
    end

    file = h5open(fname, "r")
    finalizer(file) do f
        close(f)  # Automatically close when the file object is garbage collected
    end
    return file 
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

function sol_fname(type::String, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, kreiss_coef::Float64, xmax::Float64, total_time::Float64, path::String)
    if type == "Periodic"
        return path * "BDNK_periodic_conformal_RK4_sol_k_$(k)_h_$(h)_eta0_$(η0)_lambda0_$(λ0)_chi0_$(χ0)_kreiss_$(kreiss_coef)_xmax_$(xmax)_T_$(total_time).h5"
    elseif type == "Ghost"
        return path * "BDNK_ghost_conformal_RK4_sol_k_$(k)_h_$(h)_eta0_$(η0)_lambda0_$(λ0)_chi0_$(χ0)_kreiss_$(kreiss_coef)_xmax_$(xmax)_T_$(total_time).h5"
    else
        throw("Invalid type. Choose either Periodic or Ghost.")
    end
end

function save_sol(type::String, data::AbstractArray, diagnostics::AbstractArray, ind_resid_t::AbstractArray, ind_resid_x::AbstractArray, v_prime_1::AbstractArray, v_prime_2::AbstractArray, v_prime_3::AbstractArray, eps_prime_1::AbstractArray,
    eps_prime_2::AbstractArray, eps_prime_3::AbstractArray, L::Int, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, kreiss_coef::Float64, xmax::Float64, total_time::Float64, path::String)
    fname = sol_fname(type, k, h, η0, λ0, χ0, kreiss_coef, xmax,  total_time, path)
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
        file["ind_resid_t"] = ind_resid_t
        file["ind_resid_x"] = ind_resid_x
    end
    println("File created: " * fname)
end

function kreiss_oliger(sol_access::Function, sol::AbstractArray, size::Int, var::Int, time::Int, j::Int, coef::Float64)
    u_minus_3 = sol_access(sol, size, var, time, j-3)
    u_minus_2 = sol_access(sol, size, var, time, j-2)
    u_minus_1 = sol_access(sol, size, var, time, j-1)
    u         = sol_access(sol, size, var, time, j)
    u_plus_1  = sol_access(sol, size, var, time, j+1)
    u_plus_2  = sol_access(sol, size, var, time, j+2)
    u_plus_3  = sol_access(sol, size, var, time, j+3)
    return coef * ( - u_plus_3 + 6 * u_plus_2 - 15 * u_plus_1 + 20 * u - 15 * u_minus_1 + 6 * u_minus_2 - u_minus_3 ) / 64.0
end

function F1(h::Float64, η0::Float64, λ0::Float64, χ0::Float64, vm2::Float64, vm1::Float64, vj::Float64, vp1::Float64, vp2::Float64, εm2::Float64, εm1::Float64, εj::Float64, εp1::Float64, εp2::Float64, Vm2::Float64, Vm1::Float64, Vj::Float64, Vp1::Float64, Vp2::Float64, Em2::Float64, Em1::Float64, Ej::Float64, Ep1::Float64, Ep2::Float64)
    return ((1 - vj^2)^5*(432*h^2*Ej^2*vj^2*(-1 + vj^2)^2*(λ0 + 4*χ0 + λ0*vj^2)*(3*(2*λ0 + χ0) + (-4*η0 + χ0)*vj^2) - 432*h^2*Ej^2*(-1 + vj^2)^2*(3*χ0 + (2*λ0 + χ0)*vj^2)*(3*λ0 + (-4*η0 + 3*λ0 + 4*χ0)*vj^2) + 144*h*(Em2 - 8*Em1 + 8*Ep1 - Ep2)*vj*(-1 + vj^2)^2*(3*λ0 + 7*χ0 + (λ0 + χ0)*vj^2)*(3*λ0 + (-4*η0 + 3*λ0 + 4*χ0)*vj^2)*εj + 160*(2*η0 - 3*λ0 - 2*χ0)*vj^4*(-3*λ0 + (4*η0 - 3*λ0 - 4*χ0)*vj^2)*(vm2 - 8*vm1 + 8*vp1 - vp2)^2*εj^2 + 64*η0*(-1 + vj^2)^2*(-3*λ0 + (4*η0 - 3*λ0 - 4*χ0)*vj^2)*(vm2 - 8*vm1 + 8*vp1 - vp2)^2*εj^2 - 160*(2*η0 - 3*λ0 - 2*χ0)*vj^6*(-3*(2*λ0 + χ0) + (4*η0 - χ0)*vj^2)*(vm2 - 8*vm1 + 8*vp1 - vp2)^2*εj^2 + 48*λ0*(-1 + vj^2)^2*(3*λ0 + (-4*η0 + 3*λ0 + 4*χ0)*vj^2)*(vm2 - 8*vm1 + 8*vp1 - vp2)^2*εj^2 + 64*χ0*(-1 + vj^2)^2*(3*λ0 + (-4*η0 + 3*λ0 + 4*χ0)*vj^2)*(vm2 - 8*vm1 + 8*vp1 - vp2)^2*εj^2 - 192*(4*η0 - 3*λ0 - 4*χ0)*vj*(-1 + vj^2)^2*(-3*λ0 + (4*η0 - 3*λ0 - 4*χ0)*vj^2)*(vm2 - 16*vm1 + 30*vj - 16*vp1 + vp2)*εj^2 + 2304*h^2*vj^2*(-1 + vj^2)*(8*η0 - 21*λ0 - 8*χ0 + 9*η0*vj^2)*(-3*(2*λ0 + χ0) + (4*η0 - χ0)*vj^2)*Vj^2*εj^2 + 2304*h^2*(3*λ0 + (-4*η0 + 3*λ0 + 4*χ0)*vj^2)*(3*(2*λ0 + χ0) - 3*(4*η0 - 8*λ0 - 5*χ0)*vj^2 + (-8*η0 + 2*χ0)*vj^4)*Vj^2*εj^2 - 16*vj^2*(-1 + vj^2)^2*(-3*(2*λ0 + χ0) + (4*η0 - χ0)*vj^2)*((20*η0 - 12*λ0 - 11*χ0)*(vm2 - 8*vm1 + 8*vp1 - vp2)^2 + 12*h*(8*η0 - 9*λ0 - 5*χ0)*(Vm2 - 8*Vm1 + 8*Vp1 - Vp2))*εj^2 + 192*h*η0*(-1 + vj^2)^2*(-3*λ0 + (4*η0 - 3*λ0 - 4*χ0)*vj^2)*(Vm2 - 8*Vm1 + 8*Vp1 - Vp2)*εj^2 + 192*h*η0*(-1 + vj^2)^3*(-3*λ0 + (4*η0 - 3*λ0 - 4*χ0)*vj^2)*(Vm2 - 8*Vm1 + 8*Vp1 - Vp2)*εj^2 + 576*h*λ0*(-1 + vj^2)^2*(3*λ0 + (-4*η0 + 3*λ0 + 4*χ0)*vj^2)*(Vm2 - 8*Vm1 + 8*Vp1 - Vp2)*εj^2 + 768*h*χ0*(-1 + vj^2)^2*(3*λ0 + (-4*η0 + 3*λ0 + 4*χ0)*vj^2)*(Vm2 - 8*Vm1 + 8*Vp1 - Vp2)*εj^2 + 192*h*χ0*(-1 + vj^2)^3*(3*λ0 + (-4*η0 + 3*λ0 + 4*χ0)*vj^2)*(Vm2 - 8*Vm1 + 8*Vp1 - Vp2)*εj^2 - 1536*h*vj^2*(1 - vj^2)^2.5*(3*(2*λ0 + χ0) + (-4*η0 + χ0)*vj^2)*(vm2 - 8*vm1 + 8*vp1 - vp2)*εj^2.25 + 768*h*(1 - vj^2)^2.5*(3*λ0 + (-4*η0 + 3*λ0 + 4*χ0)*vj^2)*(vm2 - 8*vm1 + 8*vp1 - vp2)*εj^2.25 - 36*h*Ej*(1 - vj^2)*(-3*λ0 + (4*η0 - 3*λ0 - 4*χ0)*vj^2)*(-4*(-1 + vj^2)*(λ0 + 7*χ0 + (2*η0 + λ0 + χ0)*vj^2)*(vm2 - 8*vm1 + 8*vp1 - vp2)*εj + 4*vj^2*(-6*η0 + 11*λ0 + 15*χ0 + (2*η0 + λ0 + χ0)*vj^2)*(vm2 - 8*vm1 + 8*vp1 - vp2)*εj + 64*h*(1 - vj^2)^1.5*(3 + vj^2)*εj^1.25 + vj*(96*h*(5*λ0 + 7*χ0 + (-2*η0 + λ0 + χ0)*vj^2)*Vj*εj + (-1 + vj^2)*(3*λ0 + 7*χ0 + (λ0 + χ0)*vj^2)*(εm2 - 8*εm1 + 8*εp1 - εp2))) - 36*h*Ej*vj*(1 - vj^2)*(-3*(2*λ0 + χ0) + (4*η0 - χ0)*vj^2)*(-192*h*(λ0 + χ0 + (-η0 + 2*λ0 + 3*χ0)*vj^2)*Vj*εj + 4*vj*((4*η0 - 7*λ0 - 13*χ0 - (5*λ0 + 3*χ0)*vj^2)*(vm2 - 8*vm1 + 8*vp1 - vp2) - 64*h*(1 - vj^2)^1.5*εj^0.25)*εj - (-1 + vj^2)*(λ0 + χ0 + (3*λ0 + 7*χ0)*vj^2)*(εm2 - 8*εm1 + 8*εp1 - εp2)) + 48*h*(-3*λ0 + (4*η0 - 3*λ0 - 4*χ0)*vj^2)*Vj*εj*(4*vj*((16*η0 - 33*λ0 - 25*χ0 + 3*(8*η0 - 9*λ0 - 5*χ0)*vj^2)*(vm2 - 8*vm1 + 8*vp1 - vp2) - 96*h*(1 - vj^2)^1.5*εj^0.25)*εj + 3*(-1 + vj^2)*(5*λ0 + 3*χ0 + (-4*η0 + 7*λ0 + 13*χ0)*vj^2)*(εm2 - 8*εm1 + 8*εp1 - εp2)) - 48*vj^3*(1 - vj^2)*(-3*λ0 + (4*η0 - 3*λ0 - 4*χ0)*vj^2)*εj*(8*(2*η0 - 3*λ0 - 2*χ0)*(vm2 - 16*vm1 + 30*vj - 16*vp1 + vp2)*εj - (η0 - 3*λ0 - 4*χ0)*(vm2 - 8*vm1 + 8*vp1 - vp2)*(εm2 - 8*εm1 + 8*εp1 - εp2)) + 48*vj^5*(-3*(2*λ0 + χ0) + (4*η0 - χ0)*vj^2)*εj*(-8*(2*η0 - 3*λ0 - 2*χ0)*(-1 + vj^2)*(vm2 - 16*vm1 + 30*vj - 16*vp1 + vp2)*εj - 80*h*(2*η0 - 3*λ0 - 2*χ0)*(vm2 - 8*vm1 + 8*vp1 - vp2)*Vj*εj + (η0 - 3*λ0 - 4*χ0)*(-1 + vj^2)*(vm2 - 8*vm1 + 8*vp1 - vp2)*(εm2 - 8*εm1 + 8*εp1 - εp2)) - 24*vj^3*(1 - vj^2)*(-3*(2*λ0 + χ0) + (4*η0 - χ0)*vj^2)*εj*(24*h*Vj*((16*η0 - 21*λ0 - 13*χ0)*(vm2 - 8*vm1 + 8*vp1 - vp2) - 32*h*sqrt(1 - vj^2)*εj^0.25)*εj - (1 - vj^2)*(24*h*(λ0 + 2*χ0)*(Em2 - 8*Em1 + 8*Ep1 - Ep2) + 8*(8*η0 - 6*λ0 - 5*χ0)*(vm2 - 16*vm1 + 30*vj - 16*vp1 + vp2)*εj - (4*η0 - 7*λ0 - 9*χ0)*(vm2 - 8*vm1 + 8*vp1 - vp2)*(εm2 - 8*εm1 + 8*εp1 - εp2)) - 32*h*(1 - vj^2)^1.5*εj^0.25*(εm2 - 8*εm1 + 8*εp1 - εp2)) - 24*vj*(-1 + vj^2)^2*(-3*(2*λ0 + χ0) + (4*η0 - χ0)*vj^2)*εj*(8*h*Vj*((8*η0 - 9*λ0 - 5*χ0)*(vm2 - 8*vm1 + 8*vp1 - vp2) - 48*h*sqrt(1 - vj^2)*εj^0.25)*εj - (1 - vj^2)*(6*h*(λ0 + χ0)*(Em2 - 8*Em1 + 8*Ep1 - Ep2) + 8*(4*η0 - χ0)*(vm2 - 16*vm1 + 30*vj - 16*vp1 + vp2)*εj - (2*η0 - λ0 - χ0)*(vm2 - 8*vm1 + 8*vp1 - vp2)*(εm2 - 8*εm1 + 8*εp1 - εp2)) - 8*h*(1 - vj^2)^1.5*εj^0.25*(εm2 - 8*εm1 + 8*εp1 - εp2)) + 48*(η0 - 2*λ0 - 3*χ0)*vj*(-1 + vj^2)^2*(-3*λ0 + (4*η0 - 3*λ0 - 4*χ0)*vj^2)*(vm2 - 8*vm1 + 8*vp1 - vp2)*εj*(εm2 - 8*εm1 + 8*εp1 - εp2) - 144*h*(4*η0 - 11*λ0 - 9*χ0)*vj^2*(-1 + vj^2)^2*(-3*(2*λ0 + χ0) + (4*η0 - χ0)*vj^2)*Vj*εj*(εm2 - 8*εm1 + 8*εp1 - εp2) + 768*h*vj*(1 - vj^2)^2.5*(3*λ0 + (-4*η0 + 3*λ0 + 4*χ0)*vj^2)*εj^1.25*(εm2 - 8*εm1 + 8*εp1 - εp2) + 3*λ0*(-1 + vj^2)^3*(3*λ0 + (-4*η0 + 3*λ0 + 4*χ0)*vj^2)*(εm2 - 8*εm1 + 8*εp1 - εp2)^2 + 144*λ0*(-1 + vj^2)^3*(3*λ0 + (-4*η0 + 3*λ0 + 4*χ0)*vj^2)*εj*(εm2 - 16*εm1 + 30*εj - 16*εp1 + εp2) - 3*(2*λ0 + χ0)*vj^2*(1 - vj^2)^3*(-3*(2*λ0 + χ0) + (4*η0 - χ0)*vj^2)*((εm2 - 8*εm1 + 8*εp1 - εp2)^2 + 48*εj*(εm2 - 16*εm1 + 30*εj - 16*εp1 + εp2)) + 6*vj^2*(1 - vj^2)*(-3*λ0 + (4*η0 - 3*λ0 - 4*χ0)*vj^2)*(8*((8*η0 - 9*λ0 - 8*χ0)*(vm2 - 8*vm1 + 8*vp1 - vp2)^2 + 16*h*(2*η0 - 3*λ0 - 2*χ0)*(Vm2 - 8*Vm1 + 8*Vp1 - Vp2))*εj^2 + 32*h*(η0 - χ0)*(-1 + vj^2)*(Vm2 - 8*Vm1 + 8*Vp1 - Vp2)*εj^2 - 256*h*sqrt(1 - vj^2)*(vm2 - 8*vm1 + 8*vp1 - vp2)*εj^2.25 - (λ0 + 2*χ0)*(-1 + vj^2)*(εm2 - 8*εm1 + 8*εp1 - εp2)^2 - 48*(λ0 + 2*χ0)*(-1 + vj^2)*εj*(εm2 - 16*εm1 + 30*εj - 16*εp1 + εp2)) - 2*vj^4*(-3*(2*λ0 + χ0) + (4*η0 - χ0)*vj^2)*(1152*h^2*(11*η0 - 30*λ0 - 20*χ0 + 9*η0*vj^2)*Vj^2*εj^2 + 8*(1 - vj^2)*((40*η0 - 42*λ0 - 31*χ0)*(vm2 - 8*vm1 + 8*vp1 - vp2)^2 + 48*h*(2*η0 - 3*λ0 - 2*χ0)*(Vm2 - 8*Vm1 + 8*Vp1 - Vp2))*εj^2 - 768*h*(1 - vj^2)^1.5*(vm2 - 8*vm1 + 8*vp1 - vp2)*εj^2.25 - 288*h*(η0 - 3*λ0 - 4*χ0)*(-1 + vj^2)*Vj*εj*(εm2 - 8*εm1 + 8*εp1 - εp2) + 3*(λ0 + 2*χ0)*(-1 + vj^2)^2*((εm2 - 8*εm1 + 8*εp1 - εp2)^2 + 48*εj*(εm2 - 16*εm1 + 30*εj - 16*εp1 + εp2)))))/(1728.0*h^2*(-1 + vj^2)^8*(-9*λ0*χ0 + 6*(2*η0 + λ0)*χ0*vj^2 + λ0*(4*η0 - χ0)*vj^4)*εj)
end

function F2(h::Float64, η0::Float64, λ0::Float64, χ0::Float64, vm2::Float64, vm1::Float64, vj::Float64, vp1::Float64, vp2::Float64, εm2::Float64, εm1::Float64, εj::Float64, εp1::Float64, εp2::Float64, Vm2::Float64, Vm1::Float64, Vj::Float64, Vp1::Float64, Vp2::Float64, Em2::Float64, Em1::Float64, Ej::Float64, Ep1::Float64, Ep2::Float64)
    return (-160*(2*η0 - 3*λ0 - 2*χ0)*(λ0 + χ0)*vj^5*(vm2 - 8*vm1 + 8*vp1 - vp2)^2*εj^2 + 48*vj^4*εj*(4*h*(-37*η0*λ0 + 60*λ0^2 - 28*η0*χ0 + 97*λ0*χ0 + 28*χ0^2 + 3*λ0*(η0 - χ0)*vj^2)*(vm2 - 8*vm1 + 8*vp1 - vp2)*Vj*εj + (λ0 + χ0)*(1 - vj^2)*(8*(2*η0 - 3*λ0 - 2*χ0)*(vm2 - 16*vm1 + 30*vj - 16*vp1 + vp2)*εj - (η0 - 3*λ0 - 4*χ0)*(vm2 - 8*vm1 + 8*vp1 - vp2)*(εm2 - 8*εm1 + 8*εp1 - εp2))) + 24*vj^2*εj*(-6*h*(λ0 + χ0)*(Em2 - 8*Em1 + 8*Ep1 - Ep2)*(-1 + vj^2)^2*(-3*λ0 - 4*χ0 + λ0*vj^2) + 8*h*Vj*((33*λ0^2 - 16*η0*(λ0 - 5*χ0) - 32*λ0*χ0 - 17*χ0^2 + (77*η0*λ0 - 93*λ0^2 + 36*η0*χ0 - 113*λ0*χ0 - 27*χ0^2)*vj^2 + 3*λ0*(-η0 + χ0)*vj^4)*(vm2 - 8*vm1 + 8*vp1 - vp2) + 96*h*(λ0 + χ0)*(1 - vj^2)^1.5*εj^0.25)*εj - (1 - vj^2)*(8*(-4*η0*λ0 + 3*λ0^2 + 8*η0*χ0 - 2*λ0*χ0 + χ0^2 + (12*η0*λ0 - 9*λ0^2 + 8*η0*χ0 - 12*λ0*χ0 - 5*χ0^2)*vj^2)*(vm2 - 16*vm1 + 30*vj - 16*vp1 + vp2)*εj - ((-2*η0*λ0 + 4*λ0^2 + 4*η0*χ0 + λ0*χ0 - 3*χ0^2 + (6*η0*λ0 - 10*λ0^2 + 4*η0*χ0 - 19*λ0*χ0 - 9*χ0^2)*vj^2)*(vm2 - 8*vm1 + 8*vp1 - vp2) + 32*h*(λ0 + χ0)*(1 - vj^2)^1.5*εj^0.25)*(εm2 - 8*εm1 + 8*εp1 - εp2))) - 36*h*Ej*(1 - vj^2)*(4*vj^3*(4*η0*(2*λ0 + 3*χ0) - 2*(λ0 + χ0)*(5*λ0 + 6*χ0) + λ0*(2*η0 + λ0 + χ0)*(-1 + vj^2))*(vm2 - 8*vm1 + 8*vp1 - vp2)*εj + 96*h*(6*χ0*(λ0 + χ0) - (λ0^2 + 3*λ0*χ0 + 2*χ0*(3*η0 + χ0))*vj^2 + λ0*(-2*η0 + λ0 + χ0)*vj^4)*Vj*εj - 4*vj*((λ0^2 - 10*λ0*χ0 + (12*η0 - 11*χ0)*χ0 + (2*η0*(5*λ0 + 6*χ0) - 3*(4*λ0^2 + 7*λ0*χ0 + 3*χ0^2))*vj^2 + λ0*(2*η0 + λ0 + χ0)*vj^4)*(vm2 - 8*vm1 + 8*vp1 - vp2) + 16*h*λ0*(1 - vj^2)^1.5*(3 - vj^2)*εj^0.25)*εj + (λ0 + χ0)*vj^2*(-1 + vj^2)*(-3*λ0 - 4*χ0 + λ0*vj^2)*(εm2 - 8*εm1 + 8*εp1 - εp2) + (λ0 + χ0)*(-1 + vj^2)*(3*χ0 + (2*λ0 + χ0)*vj^2)*(εm2 - 8*εm1 + 8*εp1 - εp2)) + 24*(1 - vj^2)*(3*χ0 + (2*λ0 + χ0)*vj^2)*εj*(8*h*Vj*((8*η0 - 9*λ0 - 5*χ0)*(vm2 - 8*vm1 + 8*vp1 - vp2) - 48*h*sqrt(1 - vj^2)*εj^0.25)*εj - (1 - vj^2)*(6*h*(λ0 + χ0)*(Em2 - 8*Em1 + 8*Ep1 - Ep2) + 8*(4*η0 - χ0)*(vm2 - 16*vm1 + 30*vj - 16*vp1 + vp2)*εj - (2*η0 - λ0 - χ0)*(vm2 - 8*vm1 + 8*vp1 - vp2)*(εm2 - 8*εm1 + 8*εp1 - εp2)) - 8*h*(1 - vj^2)^1.5*εj^0.25*(εm2 - 8*εm1 + 8*εp1 - εp2)) + 2*vj^3*(16*(16*η0 - 15*λ0 - 7*χ0)*(λ0 + 2*χ0)*(vm2 - 8*vm1 + 8*vp1 - vp2)^2*εj^2 - 1152*h^2*(8*η0*λ0 - 30*λ0^2 - η0*χ0 - 47*λ0*χ0 - 8*χ0^2 + 3*(2*η0*λ0 + 3*η0*χ0 + λ0*χ0)*vj^2)*Vj^2*εj^2 - 8*(1 - vj^2)*(((56*η0 - 57*λ0)*λ0 + 40*(η0 - 2*λ0)*χ0 - 31*χ0^2)*(vm2 - 8*vm1 + 8*vp1 - vp2)^2 + 24*h*(3*η0*λ0 - 6*λ0^2 + 2*η0*χ0 - 9*λ0*χ0 - 2*χ0^2)*(Vm2 - 8*Vm1 + 8*Vp1 - Vp2))*εj^2 - 96*h*η0*λ0*(-1 + vj^2)^2*(Vm2 - 8*Vm1 + 8*Vp1 - Vp2)*εj^2 + 96*h*λ0*χ0*(-1 + vj^2)^2*(Vm2 - 8*Vm1 + 8*Vp1 - Vp2)*εj^2 + 768*h*(λ0 + χ0)*(1 - vj^2)^1.5*(vm2 - 8*vm1 + 8*vp1 - vp2)*εj^2.25 - 72*h*(-1 + vj^2)*(-(η0*λ0) + 12*λ0^2 + 8*η0*χ0 + 27*λ0*χ0 + 12*χ0^2 + λ0*(3*η0 - χ0)*vj^2)*Vj*εj*(εm2 - 8*εm1 + 8*εp1 - εp2) - 3*λ0^2*(-1 + vj^2)^2*(εm2 - 8*εm1 + 8*εp1 - εp2)^2 - 9*λ0*χ0*(-1 + vj^2)^2*(εm2 - 8*εm1 + 8*εp1 - εp2)^2 - 6*χ0^2*(-1 + vj^2)^2*(εm2 - 8*εm1 + 8*εp1 - εp2)^2 - 144*(λ0 + χ0)*(λ0 + 2*χ0)*(-1 + vj^2)^2*εj*(εm2 - 16*εm1 + 30*εj - 16*εp1 + εp2)) + vj*(2304*h^2*(6*(λ0^2 + 4*η0*χ0 - 6*λ0*χ0 - 2*χ0^2) + (12*η0*λ0 - 36*λ0^2 + 19*η0*χ0 - 33*λ0*χ0 - 4*χ0^2)*vj^2 + (14*η0*λ0 + 9*η0*χ0 + λ0*χ0)*vj^4)*Vj^2*εj^2 + 32*(λ0 + 2*χ0)*(1 - vj^2)*((16*η0 - 9*λ0 - 7*χ0)*(vm2 - 8*vm1 + 8*vp1 - vp2)^2 + 12*h*(7*η0 - 6*λ0 - χ0)*(Vm2 - 8*Vm1 + 8*Vp1 - Vp2))*εj^2 + 192*h*η0*λ0*(-1 + vj^2)^3*(Vm2 - 8*Vm1 + 8*Vp1 - Vp2)*εj^2 - 192*h*λ0*χ0*(-1 + vj^2)^3*(Vm2 - 8*Vm1 + 8*Vp1 - Vp2)*εj^2 - 1536*h*(λ0 + 2*χ0)*(1 - vj^2)^1.5*(vm2 - 8*vm1 + 8*vp1 - vp2)*εj^2.25 + 768*h*(3*λ0 + 2*χ0)*(1 - vj^2)^2.5*(vm2 - 8*vm1 + 8*vp1 - vp2)*εj^2.25 + 144*h*(1 - vj^2)*(5*λ0^2 - 10*λ0*χ0 + 3*(4*η0 - 5*χ0)*χ0 + (5*η0*λ0 - 17*λ0^2 - 8*η0*χ0 - 25*λ0*χ0 - 5*χ0^2)*vj^2 + λ0*(-3*η0 + χ0)*vj^4)*Vj*εj*(εm2 - 8*εm1 + 8*εp1 - εp2) + 9*λ0^2*(-1 + vj^2)^3*(εm2 - 8*εm1 + 8*εp1 - εp2)^2 + 12*λ0*χ0*(-1 + vj^2)^3*(εm2 - 8*εm1 + 8*εp1 - εp2)^2 + 3*χ0^2*(-1 + vj^2)^3*(εm2 - 8*εm1 + 8*εp1 - εp2)^2 + 144*(λ0 + χ0)*(3*λ0 + χ0)*(-1 + vj^2)^3*εj*(εm2 - 16*εm1 + 30*εj - 16*εp1 + εp2) + 2*(-1 + vj^2)^2*(8*((21*λ0^2 + 30*λ0*χ0 + 11*χ0^2 - 4*η0*(9*λ0 + 5*χ0))*(vm2 - 8*vm1 + 8*vp1 - vp2)^2 - 12*h*(13*η0*λ0 - 15*λ0^2 + 4*η0*χ0 - 13*λ0*χ0 - χ0^2)*(Vm2 - 8*Vm1 + 8*Vp1 - Vp2))*εj^2 + 3*(λ0 + χ0)*(λ0 + 2*χ0)*(εm2 - 8*εm1 + 8*εp1 - εp2)^2 + 144*(λ0 + χ0)*(λ0 + 2*χ0)*εj*(εm2 - 16*εm1 + 30*εj - 16*εp1 + εp2))))/(2304.0*h^2*(1 - vj^2)*(9*λ0*χ0 - 6*(2*η0 + λ0)*χ0*vj^2 + λ0*(-4*η0 + χ0)*vj^4)*εj^2)
end

function F3(h::Float64, η0::Float64, λ0::Float64, χ0::Float64, vm2::Float64, vm1::Float64, vj::Float64, vp1::Float64, vp2::Float64, εm2::Float64, εm1::Float64, εj::Float64, εp1::Float64, εp2::Float64, Vm2::Float64, Vm1::Float64, Vj::Float64, Vp1::Float64, Vp2::Float64, Em2::Float64, Em1::Float64, Ej::Float64, Ep1::Float64, Ep2::Float64)
    return Ej
end

function F4(h::Float64, η0::Float64, λ0::Float64, χ0::Float64, vm2::Float64, vm1::Float64, vj::Float64, vp1::Float64, vp2::Float64, εm2::Float64, εm1::Float64, εj::Float64, εp1::Float64, εp2::Float64, Vm2::Float64, Vm1::Float64, Vj::Float64, Vp1::Float64, Vp2::Float64, Em2::Float64, Em1::Float64, Ej::Float64, Ep1::Float64, Ep2::Float64)
    return Vj
end

function compute_grid_points(array_access::Function, i::Int, size::Int, ε::Vector{Float64}, v::Vector{Float64}, E::Vector{Float64}, V::Vector{Float64})
    return array_access(v, size, i-2), array_access(v, size, i-1), array_access(v, size, i), array_access(v, size, i+1), array_access(v, size, i+2), array_access(ε, size, i-2), array_access(ε, size, i-1), array_access(ε, size, i), array_access(ε, size, i+1), array_access(ε, size, i+2), array_access(V, size, i-2), array_access(V, size, i-1), array_access(V, size, i), array_access(V, size, i+1), array_access(V, size, i+2), array_access(E, size, i-2), array_access(E, size, i-1), array_access(E, size, i), array_access(E, size, i+1), array_access(E, size, i+2)
end


function compute_k1!(array_access::Function, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, L::Int64, εn::Vector{Float64}, vn::Vector{Float64}, En::Vector{Float64}, Vn::Vector{Float64}, εn_k1::Vector{Float64}, vn_k1::Vector{Float64}, En_k1::Vector{Float64}, Vn_k1::Vector{Float64}, k1_ε::Vector{Float64}, k1_v::Vector{Float64}, k1_E::Vector{Float64}, k1_V::Vector{Float64})
    @inbounds for i = 1:L
        vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2 = compute_grid_points(array_access, i, L, εn, vn, En, Vn)
        k1_E[i] = k * F1(h, η0, λ0, χ0, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2)
        k1_V[i] = k * F2(h, η0, λ0, χ0, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2)
        k1_ε[i] = k * F3(h, η0, λ0, χ0, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2)
        k1_v[i] = k * F4(h, η0, λ0, χ0, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2)

        En_k1[i] = En[i] + 0.5 * k1_E[i]
        Vn_k1[i] = Vn[i] + 0.5 * k1_V[i]
        εn_k1[i] = εn[i] + 0.5 * k1_ε[i]
        vn_k1[i] = vn[i] + 0.5 * k1_v[i]
    end
end

function compute_k2!(array_access::Function, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, L::Int64, εn::Vector{Float64}, vn::Vector{Float64}, En::Vector{Float64}, Vn::Vector{Float64}, εn_k1::Vector{Float64}, vn_k1::Vector{Float64}, En_k1::Vector{Float64}, Vn_k1::Vector{Float64}, εn_k2::Vector{Float64}, vn_k2::Vector{Float64}, En_k2::Vector{Float64}, Vn_k2::Vector{Float64}, k2_ε::Vector{Float64}, k2_v::Vector{Float64}, k2_E::Vector{Float64}, k2_V::Vector{Float64})
    @inbounds for i = 1:L
        vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2 = compute_grid_points(array_access, i, L, εn_k1, vn_k1, En_k1, Vn_k1)
        k2_E[i] = k * F1(h, η0, λ0, χ0, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2)
        k2_V[i] = k * F2(h, η0, λ0, χ0, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2)
        k2_ε[i] = k * F3(h, η0, λ0, χ0, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2)
        k2_v[i] = k * F4(h, η0, λ0, χ0, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2)

        En_k2[i] = En[i] + 0.5 * k2_E[i]
        Vn_k2[i] = Vn[i] + 0.5 * k2_V[i]
        εn_k2[i] = εn[i] + 0.5 * k2_ε[i]
        vn_k2[i] = vn[i] + 0.5 * k2_v[i]
    end
end

function compute_k3!(array_access::Function, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, L::Int64, εn::Vector{Float64}, vn::Vector{Float64}, En::Vector{Float64}, Vn::Vector{Float64}, εn_k2::Vector{Float64}, vn_k2::Vector{Float64}, En_k2::Vector{Float64}, Vn_k2::Vector{Float64}, εn_k3::Vector{Float64}, vn_k3::Vector{Float64}, En_k3::Vector{Float64}, Vn_k3::Vector{Float64}, k3_ε::Vector{Float64}, k3_v::Vector{Float64}, k3_E::Vector{Float64}, k3_V::Vector{Float64})
    @inbounds for i = 1:L
        vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2 = compute_grid_points(array_access, i, L, εn_k2, vn_k2, En_k2, Vn_k2)
        k3_E[i] = k * F1(h, η0, λ0, χ0, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2)
        k3_V[i] = k * F2(h, η0, λ0, χ0, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2)
        k3_ε[i] = k * F3(h, η0, λ0, χ0, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2)
        k3_v[i] = k * F4(h, η0, λ0, χ0, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2)

        En_k3[i] = En[i] + k3_E[i]
        Vn_k3[i] = Vn[i] + k3_V[i]
        εn_k3[i] = εn[i] + k3_ε[i]
        vn_k3[i] = vn[i] + k3_v[i]
    end
end

function compute_k4!(array_access::Function, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, L::Int64, εn_k3::Vector{Float64}, vn_k3::Vector{Float64}, En_k3::Vector{Float64}, Vn_k3::Vector{Float64}, k4_ε::Vector{Float64}, k4_v::Vector{Float64}, k4_E::Vector{Float64}, k4_V::Vector{Float64})
    @inbounds for i = 1:L
        vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2 = compute_grid_points(array_access, i, L, εn_k3, vn_k3, En_k3, Vn_k3)
        k4_E[i] = k * F1(h, η0, λ0, χ0, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2)
        k4_V[i] = k * F2(h, η0, λ0, χ0, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2)
        k4_ε[i] = k * F3(h, η0, λ0, χ0, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2)
        k4_v[i] = k * F4(h, η0, λ0, χ0, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2, Vm2, Vm1, Vj, Vp1, Vp2, Em2, Em1, Ej, Ep1, Ep2)
    end
end


function RK4_step!(sol_access::Function, array_access::Function, k::Float64, h::Float64, η0::Float64, λ0::Float64, χ0::Float64, kreiss_coef::Float64, n::Int, L::Int64, data::AbstractArray, εn::Vector{Float64}, vn::Vector{Float64}, En::Vector{Float64}, Vn::Vector{Float64}, εn_k1::Vector{Float64}, vn_k1::Vector{Float64}, En_k1::Vector{Float64}, Vn_k1::Vector{Float64}, εn_k2::Vector{Float64}, vn_k2::Vector{Float64}, En_k2::Vector{Float64}, Vn_k2::Vector{Float64}, εn_k3::Vector{Float64}, vn_k3::Vector{Float64}, En_k3::Vector{Float64}, Vn_k3::Vector{Float64}, k1_ε::Vector{Float64}, k1_v::Vector{Float64}, k1_E::Vector{Float64}, k1_V::Vector{Float64}, k2_ε::Vector{Float64}, k2_v::Vector{Float64}, k2_E::Vector{Float64}, k2_V::Vector{Float64}, k3_ε::Vector{Float64}, k3_v::Vector{Float64}, k3_E::Vector{Float64}, k3_V::Vector{Float64}, k4_ε::Vector{Float64}, k4_v::Vector{Float64}, k4_E::Vector{Float64}, k4_V::Vector{Float64})

    En[:] = data[1, n, :]; Vn[:] = data[2, n, :]; εn[:] = data[3, n, :]; vn[:] = data[4, n, :];

    for i=1:L
        En[i] += -kreiss_oliger(sol_access, data, L, 1, n, i, kreiss_coef)
        Vn[i] += -kreiss_oliger(sol_access, data, L, 2, n, i, kreiss_coef)
        εn[i] += -kreiss_oliger(sol_access, data, L, 3, n, i, kreiss_coef)
        vn[i] += -kreiss_oliger(sol_access, data, L, 4, n, i, kreiss_coef)
    end

    compute_k1!(array_access, k, h, η0, λ0, χ0, L, εn, vn, En, Vn, εn_k1, vn_k1, En_k1, Vn_k1, k1_ε, k1_v, k1_E, k1_V)
    compute_k2!(array_access, k, h, η0, λ0, χ0, L, εn, vn, En, Vn, εn_k1, vn_k1, En_k1, Vn_k1, εn_k2, vn_k2, En_k2, Vn_k2, k2_ε, k2_v, k2_E, k2_V)
    compute_k3!(array_access, k, h, η0, λ0, χ0, L, εn, vn, En, Vn, εn_k2, vn_k2, En_k2, Vn_k2, εn_k3, vn_k3, En_k3, Vn_k3, k3_ε, k3_v, k3_E, k3_V)
    compute_k4!(array_access, k, h, η0, λ0, χ0, L, εn_k3, vn_k3, En_k3, Vn_k3, k4_ε, k4_v, k4_E, k4_V)

    data[1, n+1, :] = En[:] + (1/6.0) * (k1_E[:] + 2*k2_E[:] + 2*k3_E[:] + k4_E[:])
    data[2, n+1, :] = Vn[:] + (1/6.0) * (k1_V[:] + 2*k2_V[:] + 2*k3_V[:] + k4_V[:])
    data[3, n+1, :] = εn[:] + (1/6.0) * (k1_ε[:] + 2*k2_ε[:] + 2*k3_ε[:] + k4_ε[:])
    data[4, n+1, :] = vn[:] + (1/6.0) * (k1_v[:] + 2*k2_v[:] + 2*k3_v[:] + k4_v[:])
end


# for now use separate loops but eventually merge
function solve_periodic!(L::Int, k::Float64, h::Float64, ε0::Vector{Float64}, εdot0::Vector{Float64}, v0::Vector{Float64}, vdot0::Vector{Float64}, η0::Float64, λ0::Float64, χ0::Float64, kreiss_coef::Float64, xmax::Float64, total_time::Float64, path::String)
    type = "Periodic"

    sol_access(u::AbstractArray, size::Int, var::Int, time::Int, i::Int)::Float64 = u[var, time, mod(i-1, size) + 1]
    array_access(u::AbstractArray, size::Int, i::Int)::Float64 = u[mod(i-1, size) + 1]

    # allocate memory
    N = Int(ceil(total_time / k))
    data = zeros(4, N, L)

    En = zeros(L);
    En_k1 = zeros(L);
    En_k2 = zeros(L);
    En_k3 = zeros(L);

    k1_E = zeros(L);
    k2_E = zeros(L);
    k3_E = zeros(L);
    k4_E = zeros(L);

    Vn = zeros(L);
    Vn_k1 = zeros(L);
    Vn_k2 = zeros(L);
    Vn_k3 = zeros(L);

    k1_V = zeros(L);
    k2_V = zeros(L);
    k3_V = zeros(L);
    k4_V = zeros(L);

    εn = zeros(L);
    εn_k1 = zeros(L);
    εn_k2 = zeros(L);
    εn_k3 = zeros(L);

    k1_ε = zeros(L);
    k2_ε = zeros(L);
    k3_ε = zeros(L);
    k4_ε = zeros(L);

    vn = zeros(L);
    vn_k1 = zeros(L);
    vn_k2 = zeros(L);
    vn_k3 = zeros(L);

    k1_v = zeros(L);
    k2_v = zeros(L);
    k3_v = zeros(L);
    k4_v = zeros(L);

    # spatial derivatives
    eps_prime_1 = zeros(N, L)
    eps_prime_2 = zeros(N, L)
    eps_prime_3 = zeros(N, L)
    v_prime_1 = zeros(N, L)
    v_prime_2 = zeros(N, L)
    v_prime_3 = zeros(N, L)

    # periodic BCs
    data[1, 1, :] = εdot0[:];
    data[2, 1, :] = vdot0[:];
    data[3, 1, :] = ε0[:];
    data[4, 1, :] = v0[:];

    # residual diagnostics
    resid_t_component = fill(NaN, N, L)
    resid_x_component = fill(NaN, N, L)
    sim_diagnostics = zeros(5, N, L)
    
    try
        for n = 1:N-1
            print_string = "Time Level: $n, Time: $(round(k * n; digits=4)), Completion: $(round(100 * n/(N-1); digits=5))%\r"
            print(print_string)

            RK4_step!(sol_access, array_access, k, h, η0, λ0, χ0, kreiss_coef, n, L, data, εn, vn, En, Vn, εn_k1, vn_k1, En_k1, Vn_k1, εn_k2, vn_k2, En_k2, Vn_k2, εn_k3, vn_k3, En_k3, Vn_k3, k1_ε, k1_v, k1_E, k1_V, k2_ε, k2_v, k2_E, k2_V, k3_ε, k3_v, k3_E, k3_V, k4_ε, k4_v, k4_E, k4_V)

            @inbounds for j=1:L
                # compute spatial derivatives
                @views v_prime_1[n, j] = FiniteDiffOrder5.compute_first_derivative(j,  data[4, n, :], h, L)
                @views v_prime_2[n, j] = FiniteDiffOrder5.compute_second_derivative(j,  data[4, n, :], h, L)
                @views v_prime_3[n, j] = FiniteDiffOrder5.compute_third_derivative(j,  data[4, n, :], h, L)
                @views eps_prime_1[n, j] = FiniteDiffOrder5.compute_first_derivative(j,  data[3, n, :], h, L)
                @views eps_prime_2[n, j] = FiniteDiffOrder5.compute_second_derivative(j,  data[3, n, :], h, L)
                @views eps_prime_3[n, j] = FiniteDiffOrder5.compute_third_derivative(j,  data[3, n, :], h, L)

                V1 = sol_access(data, L, 2, n, j); E1 = sol_access(data, L, 1, n, j);
                v1m = sol_access(data, L, 4, n, j-1); v1 = sol_access(data, L, 4, n, j); v1p = sol_access(data, L, 4, n, j+1);
                ε1m = sol_access(data, L, 3, n, j-1); ε1 = sol_access(data, L, 3, n, j); ε1p = sol_access(data, L, 3, n, j+1);
        
                
                sim_diagnostics[1, n, j] = BDNKConformalTensor.tt_zeroth_order(v1, ε1)
                sim_diagnostics[2, n, j] = BDNKConformalTensor.xx_zeroth_order(v1, ε1)
                sim_diagnostics[3, n, j] = BDNKConformalTensor.tt_first_order(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
                sim_diagnostics[4, n, j] = BDNKConformalTensor.xx_first_order(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
                sim_diagnostics[5, n, j] = BDNKConformalTensor.four_velocity_inner_prod(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)

                if n > 3
                    Vm2j = data[2, n-3, j]; Vm1j = data[2, n-2, j]; V0j = data[2, n-1, j]; Vp1j = data[2, n, j]; Vp2j = data[2, n+1, j];
                    Em2j = data[1, n-3, j]; Em1j = data[1, n-2, j]; E0j = data[1, n-1, j]; Ep1j = data[1, n, j]; Ep2j = data[1, n+1, j];

                    V0m2 = sol_access(data, L, 2, n-1, j-2); V0m1 = sol_access(data, L, 2, n-1, j-1); V0j = sol_access(data, L, 2, n-1, j);
                    V0p1 = sol_access(data, L, 2, n-1, j+1); V0p2 = sol_access(data, L, 2, n-1, j+2);
                    E0m2 = sol_access(data, L, 1, n-1, j-2); E0m1 = sol_access(data, L, 1, n-1, j-1); E0j = sol_access(data, L, 1, n-1, j);
                    E0p1 = sol_access(data, L, 1, n-1, j+1); E0p2 = sol_access(data, L, 1, n-1, j+2);

                    v0m2 = sol_access(data, L, 4, n-1, j-2); v0m1 = sol_access(data, L, 4, n-1, j-1); v0j = sol_access(data, L, 4, n-1, j);
                    v0p1 = sol_access(data, L, 4, n-1, j+1); v0p2 = sol_access(data, L, 4, n-1, j+2); 
                    
                    ε0m2 = sol_access(data, L, 3, n-1, j-2); ε0m1 = sol_access(data, L, 3, n-1, j-1); ε0j = sol_access(data, L, 3, n-1, j);
                    ε0p1 = sol_access(data, L, 3, n-1, j+1); ε0p2 = sol_access(data, L, 3, n-1, j+2);

                    
                    resid_t_component[n-1, j] = abs(RK4StressEnergyGradients.BDNK.stress_energy_gradient_t(v0m2, v0m1, v0j, v0p1, v0p2, ε0m2, ε0m1, ε0j, ε0p1, ε0p2, V0m2, V0m1, V0j, V0p1, V0p2, E0m2, E0m1, E0j, E0p1, E0p2, Vm2j, Vm1j, Vp1j, Vp2j, Em2j, Em1j, Ep1j, Ep2j, η0, λ0, χ0, k, h))

                    resid_x_component[n-1, j] = abs(RK4StressEnergyGradients.BDNK.stress_energy_gradient_x(v0m2, v0m1, v0j, v0p1, v0p2, ε0m2, ε0m1, ε0j, ε0p1, ε0p2, V0m2, V0m1, V0j, V0p1, V0p2, E0m2, E0m1, E0j, E0p1, E0p2, Vm2j, Vm1j, Vp1j, Vp2j, Em2j, Em1j, Ep1j, Ep2j, η0, λ0, χ0, k, h))
                end
            end
        end
    catch e
        println("\nERROR OCCURRED. SAVING SOLUTION AND RETRHOWING ERROR.\n")
        save_sol(type, data, sim_diagnostics, resid_t_component, resid_x_component, v_prime_1, v_prime_2, v_prime_3, eps_prime_1, eps_prime_2, eps_prime_3, L, k, h, η0, λ0, χ0, kreiss_coef, xmax, total_time, path)
        rethrow(e)
    end

    @inbounds for j=1:L
        # compute spatial derivatives
        n = N
        @views v_prime_1[n, j] = FiniteDiffOrder5.compute_first_derivative(j,  data[4, n, :], h, L)
        @views v_prime_2[n, j] = FiniteDiffOrder5.compute_second_derivative(j,  data[4, n, :], h, L)
        @views v_prime_3[n, j] = FiniteDiffOrder5.compute_third_derivative(j,  data[4, n, :], h, L)
        @views eps_prime_1[n, j] = FiniteDiffOrder5.compute_first_derivative(j,  data[3, n, :], h, L)
        @views eps_prime_2[n, j] = FiniteDiffOrder5.compute_second_derivative(j,  data[3, n, :], h, L)
        @views eps_prime_3[n, j] = FiniteDiffOrder5.compute_third_derivative(j,  data[3, n, :], h, L)

        V1 = sol_access(data, L, 2, n, j); E1 = sol_access(data, L, 1, n, j);
        v1m = sol_access(data, L, 4, n, j-1); v1 = sol_access(data, L, 4, n, j); v1p = sol_access(data, L, 4, n, j+1);
        ε1m = sol_access(data, L, 3, n, j-1); ε1 = sol_access(data, L, 3, n, j); ε1p = sol_access(data, L, 3, n, j+1);

        
        sim_diagnostics[1, n, j] = BDNKConformalTensor.tt_zeroth_order(v1, ε1)
        sim_diagnostics[2, n, j] = BDNKConformalTensor.xx_zeroth_order(v1, ε1)
        sim_diagnostics[3, n, j] = BDNKConformalTensor.tt_first_order(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
        sim_diagnostics[4, n, j] = BDNKConformalTensor.xx_first_order(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
        sim_diagnostics[5, n, j] = BDNKConformalTensor.four_velocity_inner_prod(v1m, v1, v1p, ε1m, ε1, ε1p, V1, E1, h, η0, λ0, χ0)
    end

    save_sol(type, data, sim_diagnostics, resid_t_component, resid_x_component, v_prime_1, v_prime_2, v_prime_3, eps_prime_1, eps_prime_2, eps_prime_3, L, k, h, η0, λ0, χ0, kreiss_coef, xmax, total_time, path)
end

# for now use separate loops but eventually merge
function solve!(type::String, L::Int, k::Float64, h::Float64, ε0::Vector{Float64}, εdot0::Vector{Float64}, v0::Vector{Float64}, vdot0::Vector{Float64}, η0::Float64, λ0::Float64, χ0::Float64, kreiss_coef::Float64, xmax::Float64, total_time::Float64, path::String)
    if type == "Periodic"
        solve_periodic!(L, k, h, ε0, εdot0, v0, vdot0, η0, λ0, χ0, kreiss_coef, xmax, total_time, path)
    elseif type == "Ghost"
        nothing
    else
        throw(ArgumentError("Invalid type. Choose either 'Periodic' or 'Ghost'."))
    end
end
end