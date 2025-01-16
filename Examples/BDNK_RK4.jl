#=

    Examples for BDNK RK4 solver.

=#

include("../BDNKRK4.jl")
include("../ConvergenceTests.jl")
include("../Diagnostics.jl")
using Plots, Plots.Measures, LaTeXStrings, LinearAlgebra, DelimitedFiles

path = "./Results/BDNK/"
mkpath(path)

#################### PARAMS ####################
type = "Periodic"
L0 = 1025;
λ = 0.1;
total_time = 400.0
xmax = 400.0
h0 = 2 * xmax / (L0 - 1)
k0 = λ * h0
h_array = [h0, 2.0h0, 4.0h0, 8.0h0, 16.0h0];
k_array = [k0, 2.0k0, 4.0k0, 8.0k0, 16.0k0];
L_array = [Int(ceil(2 * xmax / h_array[i]) + 1) for i in eachindex(h_array)];

η_over_s = 20 / 4π
e0 = 10.0
η0 = 4 * (e0)^(1/4) * η_over_s / 3
println("η0 = $(η0)")
frameA = [η0, 25 * η0 / 3, 25 * η0 / 2]
frameB = [η0, 25 * η0 / 7, 25 * η0 / 4]
frame = frameA
max_iter = 20; min_iter = 1; target_tol = 1e-6; max_tol = 1e-6; kreiss_coef = 0.5;

#################### INITIAL DATA ####################
# initial conditions
A = 0.01
w = 5.0
delta = 0.1 
shift_ε = 0.0
shift_v = 0.0
# shift_ε = 50.0
# shift_v = -50.0

ε0_func(x::Float64)::Float64 = A * exp(-(x-shift_ε)^2 / (w^2)) + delta
ε0_prime_func(x::Float64)::Float64 = -2.0 * A * exp(-(x-shift_ε)^2 / (w^2)) * (x-shift_ε) / w^2
v0_func(x::Float64)::Float64 = 0.0
v0_prime_func(x::Float64)::Float64 = 0.0

#################### SIMULATION ####################
i = 1
v0, ε0, vdot0, εdot0 = ConvergenceTests.BDNK.compute_initial_data(L_array[i], ε0_func, v0_func, ε0_prime_func, v0_prime_func, frame, kreiss_coef, xmax);

# include("../BDNKRK4.jl")
@time BDNKRK4.solve!(type, L_array[i], k_array[i], h_array[i], ε0, εdot0, v0, vdot0, frame..., kreiss_coef, xmax, total_time, path);

#################### DIAGNOSTIC CHECKS ###################
k = k_array[1]
h = h_array[1]
sol = BDNKRK4.load_sol(type, k, h, frame..., kreiss_coef, xmax,  total_time, path);

start_time = 0.0
stop_time = total_time
len = 10
times = range(start = start_time, stop = stop_time, length = len) |> collect

# residual
# Diagnostics.BDNK.plot_independent_resid(sol, times)

Diagnostics.BDNK.plot_one_var(sol, times; eps=true, v=false, time=false, first=false, second=false, third=false, xlims=(-xmax, xmax), ylims=(0.0, 0.0), size=(800, 600))
close(sol)
