#=

    Examples for Euler RK4 solver.

=#

include("../EulerRK4.jl")
include("../Diagnostics.jl")
using Plots, Plots.Measures, LaTeXStrings

path = "./Results/RK4/"
mkpath(path)

#################### PARAMS ####################
# simulation params
L0 = 2049;
λ = 0.1;
total_time = 30.0
xmax = 50.0
h0 = 2 * xmax / (L0 - 1)
k0 = λ * h0
h_array = [h0, 2.0h0, 4.0h0, 8.0h0, 16.0h0];
k_array = [k0, 2.0k0, 4.0k0, 8.0k0, 16.0k0];
L_array = [Int(ceil(2 * xmax / h_array[i]) + 1) for i in eachindex(h_array)];

# initial conditions
A = 0.4
w = 5.0
delta = 0.1
shift = 0.0

ε0_func(x::Float64)::Float64 = A * exp(-(x-shift)^2 / (w^2)) + delta
v0_func(x::Float64)::Float64 = 0.0

x_h = range(start=-xmax, stop=xmax, length=L0) |> collect;
v0 = v0_func.(x_h)
ε0 = ε0_func.(x_h)

#################### SIMULATION ####################
# latest run took ~ 4 secs
@time EulerRK4.solve!(L0, k0, h0, ε0, v0, xmax, total_time, path);

#################### PLOTTING ####################
sol = EulerRK4.load_sol(k0, h0, xmax, total_time, path)

start_time = 0.0
stop_time = total_time
len = 10
times = range(start = start_time, stop = stop_time, length = len) |> collect

# solution -- with and without limits
Diagnostics.Euler.plot_solution(sol, times)

eps_lims = (0.05, 0.5); v_lims = (-0.5, 0.5); eps_dot_lims = (-0.01, 0.01); vdot_lims = (-0.01, 0.01)
Diagnostics.Euler.plot_solution(sol, times, lims=true, eps_lims=eps_lims, v_lims=v_lims)

Diagnostics.BDNK.plot_one_var(sol, times; eps=true, v=false, time=false, first=false, second=false, third=false, xlims=(-80, 80.0), ylims=(0.0, 0.0), size=(800, 600))

close(sol)