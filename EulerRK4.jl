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
include("RK4StressEnergyGradients.jl")
module EulerRK4
using LinearAlgebra, Printf, SparseArrays
using HDF5
using ..RK4StressEnergyGradients

one_norm(x::AbstractArray) = sum(abs, x) / length(x)

sol_fname(k::Float64, h::Float64, xmax::Float64, total_time::Float64, path::String) = path* "Euler_conformal_sol_RK4_h_$(h)_k_$(k)_xmax_$(xmax)_T_$(total_time).h5"

function save_sol(data::AbstractArray, ind_resid_t::AbstractArray, ind_resid_x::AbstractArray, L::Int, k::Float64, h::Float64, xmax::Float64, total_time::Float64, path::String)
    fname = sol_fname(k, h, xmax,  total_time, path)

    h5open(fname, "w") do file
        file["t"] = range(start = 0.0, step = k, length = size(data, 2)) |> collect
        file["x"] = range(start = -xmax, stop = xmax, length = L) |> collect
        file["v"] = data[2, :, 5:L+4]
        file["eps"] = data[1, :, 5:L+4]
        file["ind_resid_t"] = ind_resid_t
        file["ind_resid_x"] = ind_resid_x
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


function F(h::Float64, vm2::Float64, vm1::Float64, vj::Float64, vp1::Float64, vp2::Float64, εm2::Float64, εm1::Float64, εj::Float64, εp1::Float64, εp2::Float64)
    return  (2*((εm2 - 8*εm1 + 8*εp1 - εp2)/(12.0*h)*vj + 2*(vm2 - 8*vm1 + 8*vp1 - vp2)/(12.0*h)*εj))/(-3 + vj^2)
end

function G(h::Float64, vm2::Float64, vm1::Float64, vj::Float64, vp1::Float64, vp2::Float64, εm2::Float64, εm1::Float64, εj::Float64, εp1::Float64, εp2::Float64)
    return (3*(εm2 - 8*εm1 + 8*εp1 - εp2)/(12.0*h)*(-1 + vj^2)^2 + 8*(vm2 - 8*vm1 + 8*vp1 - vp2)/(12.0*h)*vj*εj)/(4.0*(-3 + vj^2)*εj)
end

function compute_grid_points(i::Int, ε::Vector{Float64}, v::Vector{Float64})
    return v[i-2], v[i-1], v[i], v[i+1], v[i+2], ε[i-2], ε[i-1], ε[i], ε[i+1], ε[i+2]
end

function compute_k1!(k::Float64, h::Float64, L::Int64, εn::Vector{Float64}, vn::Vector{Float64}, εn_k1::Vector{Float64}, vn_k1::Vector{Float64}, k1_ε::Vector{Float64}, k1_v::Vector{Float64})
    @inbounds for i = 5:L+4
        vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2 = compute_grid_points(i, εn, vn)
        k1_ε[i] = k * F(h, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2)
        k1_v[i] = k * G(h, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2)

        εn_k1[i] = εn[i] + 0.5 * k1_ε[i]
        vn_k1[i] = vn[i] + 0.5 * k1_v[i]
    end
    εn_k1[1:4] = reverse(εn_k1[L+1:L+4]); εn_k1[L+5:L+8] = εn_k1[5:8];
    vn_k1[1:4] = reverse(vn_k1[L+1:L+4]); vn_k1[L+5:L+8] = vn_k1[5:8];
end

function compute_k2!(k::Float64, h::Float64, L::Int64, εn::Vector{Float64}, vn::Vector{Float64}, εn_k1::Vector{Float64}, vn_k1::Vector{Float64}, εn_k2::Vector{Float64}, vn_k2::Vector{Float64}, k2_ε::Vector{Float64}, k2_v::Vector{Float64})
    @inbounds for i = 5:L+4
        vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2 = compute_grid_points(i, εn_k1, vn_k1)
        k2_ε[i] = k * F(h, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2)
        k2_v[i] = k * G(h, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2)

        εn_k2[i] = εn[i] + 0.5 * k2_ε[i]
        vn_k2[i] = vn[i] + 0.5 * k2_v[i]
    end
    εn_k2[1:4] = reverse(εn_k2[L+1:L+4]); εn_k2[L+5:L+8] = εn_k2[5:8];
    vn_k2[1:4] = reverse(vn_k2[L+1:L+4]); vn_k2[L+5:L+8] = vn_k2[5:8];
end

function compute_k3!(k::Float64, h::Float64, L::Int64, εn::Vector{Float64}, vn::Vector{Float64}, εn_k2::Vector{Float64}, vn_k2::Vector{Float64}, εn_k3::Vector{Float64}, vn_k3::Vector{Float64}, k3_ε::Vector{Float64}, k3_v::Vector{Float64})
    @inbounds for i = 5:L+4
        vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2 = compute_grid_points(i, εn_k2, vn_k2)
        k3_ε[i] = k * F(h, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2)
        k3_v[i] = k * G(h, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2)

        εn_k3[i] = εn[i] + k3_ε[i]
        vn_k3[i] = vn[i] + k3_v[i]
    end
    εn_k3[1:4] = reverse(εn_k3[L+1:L+4]); εn_k3[L+5:L+8] = εn_k3[5:8];
    vn_k3[1:4] = reverse(vn_k3[L+1:L+4]); vn_k3[L+5:L+8] = vn_k3[5:8];
end

function compute_k4!(k::Float64, h::Float64, L::Int64, εn_k3::Vector{Float64}, vn_k3::Vector{Float64}, k4_ε::Vector{Float64}, k4_v::Vector{Float64})
    @inbounds for i = 5:L+4
        vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2 = compute_grid_points(i, εn_k3, vn_k3)
        k4_ε[i] = k * F(h, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2)
        k4_v[i] = k * G(h, vm2, vm1, vj, vp1, vp2, εm2, εm1, εj, εp1, εp2)
    end
end


function kreiss_oliger(data::AbstractArray, size::Int, var::Int, time::Int, j::Int, coef::Float64)
    u_minus_3 = data[var, time, j-3]
    u_minus_2 = data[var, time, j-2]
    u_minus_1 = data[var, time, j-1]
    u         = data[var, time, j]
    u_plus_1  = data[var, time, j+1]
    u_plus_2  = data[var, time, j+2]
    u_plus_3  = data[var, time, j+3]
    return coef * ( - u_plus_3 + 6 * u_plus_2 - 15 * u_plus_1 + 20 * u - 15 * u_minus_1 + 6 * u_minus_2 - u_minus_3 ) / 64.0
end


function RK4_step!(k::Float64, h::Float64, kreiss_coef::Float64, n::Int, L::Int64, data::AbstractArray, εn::Vector{Float64}, vn::Vector{Float64}, εn_k1::Vector{Float64}, vn_k1::Vector{Float64}, εn_k2::Vector{Float64}, vn_k2::Vector{Float64}, εn_k3::Vector{Float64}, vn_k3::Vector{Float64}, k1_ε::Vector{Float64}, k1_v::Vector{Float64}, k2_ε::Vector{Float64}, k2_v::Vector{Float64}, k3_ε::Vector{Float64}, k3_v::Vector{Float64}, k4_ε::Vector{Float64}, k4_v::Vector{Float64})

    εn[:] = data[1, n, :]; vn[:] = data[2, n, :];

    for i=5:L+4
        # εn[i] += -kreiss_oliger(data, L, 1, n, i, kreiss_coef) - kreiss_oliger_low_wavelength(data, L, 1, n, i, kreiss_coef, spacing)
        # vn[i] += -kreiss_oliger(data, L, 2, n, i, kreiss_coef) - kreiss_oliger_low_wavelength(data, L, 2, n, i, kreiss_coef, spacing)
        εn[i] += kreiss_oliger(data, L, 1, n, i, kreiss_coef)
        vn[i] += kreiss_oliger(data, L, 2, n, i, kreiss_coef)
    end

    compute_k1!(k, h, L, εn, vn, εn_k1, vn_k1, k1_ε, k1_v)
    compute_k2!(k, h, L, εn, vn, εn_k1, vn_k1, εn_k2, vn_k2, k2_ε, k2_v)
    compute_k3!(k, h, L, εn, vn, εn_k2, vn_k2, εn_k3, vn_k3, k3_ε, k3_v)
    compute_k4!(k, h, L, εn_k3, vn_k3, k4_ε, k4_v)

    data[1, n+1, :] = εn[:] + (1/6.0) * (k1_ε[:] + 2*k2_ε[:] + 2*k3_ε[:] + k4_ε[:])
    data[2, n+1, :] = vn[:] + (1/6.0) * (k1_v[:] + 2*k2_v[:] + 2*k3_v[:] + k4_v[:])

    data[1, n+1, 1:4] = reverse(data[1, n+1, L+1:L+4]); data[1, n+1, L+5:L+8] = data[1, n+1, 5:8];
    data[2, n+1, 1:4] = reverse(data[2, n+1, L+1:L+4]); data[2, n+1, L+5:L+8] = data[2, n+1, 5:8];
end

# for now use separate loops but eventually merge
@views function solve!(L::Int, k::Float64, h::Float64, ε0::Vector{Float64}, v0::Vector{Float64}, xmax::Float64, total_time::Float64, path::String)
    # allocate memory
    N = Int(ceil(total_time / k))
    data = zeros(2, N, L+8)

    εn = zeros(L+8);
    εn_k1 = zeros(L+8);
    εn_k2 = zeros(L+8);
    εn_k3 = zeros(L+8);

    k1_ε = zeros(L+8);
    k2_ε = zeros(L+8);
    k3_ε = zeros(L+8);
    k4_ε = zeros(L+8);

    vn = zeros(L+8);
    vn_k1 = zeros(L+8);
    vn_k2 = zeros(L+8);
    vn_k3 = zeros(L+8);

    k1_v = zeros(L+8);
    k2_v = zeros(L+8);
    k3_v = zeros(L+8);
    k4_v = zeros(L+8);

    # periodic BCs
    data[1, 1, 5:L+4] = ε0[:]; 
    data[2, 1, 5:L+4] = v0[:]; 

    data[1, 1, 1:4] = reverse(data[1, 1, L+1:L+4]); data[1, 1, L+5:L+8] = data[1, 1, 5:8];
    data[2, 1, 1:4] = reverse(data[2, 1, L+1:L+4]); data[2, 1, L+5:L+8] = data[2, 1, 5:8];
    
    # residual diagnostics
    resid_t_component = fill(NaN, N, L)
    resid_x_component = fill(NaN, N, L)

    kreiss_coef = 0.000

    try
        for n = 1:N-1
            print_string = "Time Level: $n, Completion: $(round(100 * n/(N-1); digits=5))%\r"
            print(print_string)

            RK4_step!(k, h, kreiss_coef, n, L, data, εn, vn, εn_k1, vn_k1, εn_k2, vn_k2, εn_k3, vn_k3, k1_ε, k1_v, k2_ε, k2_v, k3_ε, k3_v, k4_ε, k4_v)

            # compute independent residual
            if n > 3
                @inbounds for j=5:L+4
                    vm2j = data[2, n-3, j]; vm1j = data[2, n-2, j]; v0j = data[2, n-1, j]; vp1j = data[2, n, j]; vp2j = data[2, n+1, j];
                    εm2j = data[1, n-3, j]; εm1j = data[1, n-2, j]; ε0j = data[1, n-1, j]; εp1j = data[1, n, j]; εp2j = data[1, n+1, j];

                    v0m2 = data[2, n-1, j-2]; v0m1 = data[2, n-1, j-1]; v0 = data[2, n-1, j]; v0p1 = data[2, n-1, j+1]; v0p2 = data[2, n-1, j+2]; ε0m2 = data[1, n-1, j-2]; ε0m1 = data[1, n-1, j-1]; ε0 = data[1, n-1, j]; ε0p1 = data[1, n-1, j+1]; ε0p2 = data[1, n-1, j+2];
                    
                    resid_t_component[n-1, j-4] = abs(RK4StressEnergyGradients.Euler.stress_energy_gradient_t(v0m2, v0m1, v0j, v0p1, v0p2, ε0m2, ε0m1, ε0j, ε0p1, ε0p2, vm2j, vm1j, vp1j, vp2j, εm2j, εm1j, εp1j, εp2j, k, h))
                    resid_x_component[n-1, j-4] = abs(RK4StressEnergyGradients.Euler.stress_energy_gradient_x(v0m2, v0m1, v0j, v0p1, v0p2, ε0m2, ε0m1, ε0j, ε0p1, ε0p2, vm2j, vm1j, vp1j, vp2j, εm2j, εm1j, εp1j, εp2j, k, h))
                end
            end
        end
    catch e
        println("\nERROR OCCURRED. SAVING SOLUTION AND RETRHOWING ERROR.\n")
        save_sol(data, resid_t_component, resid_x_component, L, k, h, xmax,  total_time, path)
        rethrow(e)
    end
    save_sol(data, resid_t_component, resid_x_component, L, k, h, xmax,  total_time, path)
end

end