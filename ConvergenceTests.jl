include("EulerSystem.jl")
include("BDNKSystem.jl")
module ConvergenceTests

module Euler
using Plots, Plots.Measures, LaTeXStrings
using ...EulerSystem

# plot attributes
left_margin=10mm; right_margin=10mm; bottom_margin=10mm; top_margin=10mm; 
xtickfontsize=15; ytickfontsize=15; guidefontsize=20; legendfontsize=15; ms=1.5; markerstrokewidth=0;
text_y_pos = 1e6; annotate_fs = 15;# annotation y_pos and fontsize
Plots.reset_defaults()
default(legendfontsize=legendfontsize, framestyle=:box, yminorticks=1, minorgrid=true, xtickfontsize = xtickfontsize,
ytickfontsize = ytickfontsize, dpi=600, guidefontsize = guidefontsize, markerstrokewidth=markerstrokewidth, markersize=ms, left_margin=10mm, right_margin=10mm, bottom_margin=10mm, top_margin=10mm);

function compute_solutions(ε0_func::Function, v0_func::Function, L_array::Vector{Int64}, h_array::Vector{Float64}, k_array::Vector{Float64}, total_time::Float64, xmax::Float64, kreiss_coef::Float64, target_tol::Float64, max_tol::Float64, max_iter::Int,
    min_iter::Int, path::String)
    mkpath(path)
    for i in eachindex(h_array)
        x_h = range(start=-xmax, stop=xmax, length=L_array[i]) |> collect;
        v0 = v0_func.(x_h)
        ε0 = ε0_func.(x_h)
        EulerSystem.solve!(L_array[i], k_array[i], h_array[i], ε0, v0, xmax,  total_time, kreiss_coef, target_tol, max_tol, max_iter, min_iter, path);
    end
end

function compute_convergence_ratio(h::Float64, k::Float64, xmax::Float64, total_time::Float64, path::String)
    sol_h = EulerSystem.load_sol(k, h, xmax,  total_time, path)
    v_h = sol_h["v"][:, :]
    eps_h = sol_h["eps"][:, :]

    sol_2h =EulerSystem.load_sol(2.0k, 2.0h, xmax,  total_time, path)
    v_2h = sol_2h["v"][:, :]
    eps_2h = sol_2h["eps"][:, :]

    sol_4h = EulerSystem.load_sol(4.0k, 4.0h, xmax,  total_time, path)
    t_4h = sol_4h["t"][:]
    v_4h = sol_4h["v"][:, :]
    eps_4h = sol_4h["eps"][:, :]

    x_h = sol_h["x"][:]
    x_2h = sol_2h["x"][:]
    x_4h = sol_4h["x"][:]

    close(sol_h)
    close(sol_2h)
    close(sol_4h)

    Q_v = zeros(length(t_4h))
    Q_eps = zeros(length(t_4h))
    for i=1:length(t_4h)
        time_idx_2h = 1 + 2 * (i - 1)
        time_idx_h = 1 + 4 * (i - 1)    
        # compute ||u^{4h} - u^{2h}||
        if x_4h != x_2h[1:2:end]
            throw(DomainError("x values are not equal for (2h, 4h) pair"))
        end
        abs_2h_4h_v = EulerSystem.one_norm(v_4h[i, :] - v_2h[time_idx_2h, 1:2:end])
        abs_2h_4h_eps = EulerSystem.one_norm(eps_4h[i, :] - eps_2h[time_idx_2h, 1:2:end])

        # compute ||u^{2h} - u^{h}||
        if x_2h != x_h[1:2:end] || x_4h != x_h[1:4:end]
            throw(DomainError("x values are not equal for (h, 2h) pair"))
        end

        abs_h_2h_v = EulerSystem.one_norm(v_2h[time_idx_2h, :] - v_h[time_idx_h, 1:2:end])
        abs_h_2h_eps = EulerSystem.one_norm(eps_2h[time_idx_2h, :] - eps_h[time_idx_h, 1:2:end])
        
        Q_v[i] = abs_2h_4h_v / abs_h_2h_v
        Q_eps[i] = abs_2h_4h_eps / abs_h_2h_eps
    end
    return t_4h, Q_v, Q_eps
end

function plot_convergence_factor(L_array::Vector{Int64}, h_array::Vector{Float64}, k_array::Vector{Float64}, xmax::Float64, total_time::Float64, path::String)
    if length(L_array) != 5
        throw(DomainError("Require five grid sizes"))
    end
    time = []
    Qv = [];
    Qeps = [];

    for i=1:length(h_array)-2
        t_i, Q_v, Q_eps = compute_convergence_ratio(h_array[i], k_array[i], xmax, total_time, path)
        push!(Qv, Q_v)
        push!(Qeps, Q_eps)
        push!(time, t_i)
    end

    text_x_pos = 1; text_y_pos = 7.9; annotate_fs = 20; # annotation y_pos and fontsize

    Qv_plot = plot(time, Qv, xlabel = "Time", ylabel = L"\mathrm{Q_{N}(t)}", size = (800, 400), color = [:blue :red :green], ylims = (0, 8), legend = false)
    annotate!(Qv_plot, text_x_pos, text_y_pos, text(L"\frac{||v^{4h}-v^{2h}||_{1}}{||v^{2h}-v^{h}||_{1}}", annotate_fs, color = :black, valign = :top, halign = :left))

    Qeps_plot = plot(time, Qeps, xlabel = "Time", size = (800, 400), color = [:blue :red :green],label = [L"N=%$(L_array[1])" L"N=%$(L_array[2])" L"N=%$(L_array[3])"], legend = :topright, ylims = (0, 8), left_margin=-10mm, yformatter=:none)
    annotate!(Qeps_plot, text_x_pos, text_y_pos, text(L"\frac{||\epsilon^{4h}-\epsilon^{2h}||_{1}}{||\epsilon^{2h}-\epsilon^{h}||_{1}}", annotate_fs, color = :black, valign = :top, halign = :left))

    p = plot(Qv_plot, Qeps_plot, layout = (1, 2), size = (1200, 400))
    display(p)
end

function compute_independent_residual_ratio(h::Float64, k::Float64, xmax::Float64, total_time::Float64, path::String)
    sol_h = EulerSystem.load_sol(k, h, xmax,  total_time, path)
    t_h = sol_h["t"][:]
    x_h = sol_h["x"][:]
    ind_resid_t_h = sol_h["ind_resid_t"][:, :]
    ind_resid_x_h = sol_h["ind_resid_x"][:, :]

    sol_2h =EulerSystem.load_sol(2.0k, 2.0h, xmax,  total_time, path)
    t_2h = sol_2h["t"][:]
    x_2h = sol_2h["x"][:]
    ind_resid_t_2h = sol_2h["ind_resid_t"][:, :]
    ind_resid_x_2h = sol_2h["ind_resid_x"][:, :]

    close(sol_h)
    close(sol_2h)

    if t_2h != t_h[1:2:end]
        throw(DomainError("t values do not overlap for (h, 2h) pair"))
    end

    if x_2h != x_h[1:2:end]
        throw(DomainError("x values do not overlap for (h, 2h) pair"))
    end

    t_ratio = zeros(length(t_2h)-2)
    x_ratio = zeros(length(t_2h)-2)
    for i in 1:length(t_2h)
        time_idx_h = 1 + 2 * (i - 1)
        if i != 1 && i != length(t_2h)
            t_ratio[i-1] = EulerSystem.one_norm(ind_resid_t_2h[i-1, :]) / EulerSystem.one_norm(ind_resid_t_h[time_idx_h-1,:])
            x_ratio[i-1] = EulerSystem.one_norm(ind_resid_x_2h[i-1, :]) / EulerSystem.one_norm(ind_resid_x_h[time_idx_h-1,:])
        end 
    end
    return t_2h[2:end-1], t_ratio, x_ratio
end

function plot_independent_residual_ratio(L_array::Vector{Int64}, h_array::Vector{Float64}, k_array::Vector{Float64}, xmax::Float64, total_time::Float64, path::String)
    if length(L_array) != 5
        throw(DomainError("Require five grid sizes"))
    end

    time_ind_resid = [];
    Q_ind_resid_t = [];
    Q_ind_resid_x = [];

    for i=1:length(h_array)-1
        t_i, QN_t, QN_x = compute_independent_residual_ratio(h_array[i], k_array[i], xmax, total_time, path)
        push!(Q_ind_resid_t, QN_t)
        push!(Q_ind_resid_x, QN_x)
        push!(time_ind_resid, t_i)
    end

    Q_ind_resid_t_plot = plot(time_ind_resid, Q_ind_resid_t, xlabel = "Time", ylabel = L"\mathrm{Q_{N}(t)}", size = (800, 400), color = [:blue :red :green :magenta], ylims = (0, 8), legend = false, title = L"\nabla_{a}T^{at}", titlefont=20,
    top_margin = 6mm)

    Q_ind_resid_x_plot = plot(time_ind_resid, Q_ind_resid_x, xlabel = "Time", size = (800, 400), color = [:blue :red :green :magenta],label = [L"N=%$(L_array[1])" L"N=%$(L_array[2])" L"N=%$(L_array[3])" L"N=%$(L_array[4])"], 
    legend = :topright, ylims = (0, 8), left_margin=-10mm, yformatter=:none, title = L"\nabla_{a}T^{ax}", titlefont=20, top_margin = 6mm)

    p = plot(Q_ind_resid_t_plot, Q_ind_resid_x_plot, layout = (1, 2), size = (1200, 400))
    display(p)
end

end

module BDNK
using Plots, Plots.Measures, LaTeXStrings
using ...BDNKSystem

#################### plot attributes ####################
left_margin=10mm; right_margin=10mm; bottom_margin=10mm; top_margin=10mm; 
xtickfontsize=15; ytickfontsize=15; guidefontsize=20; legendfontsize=15; ms=1.5; markerstrokewidth=0;
Plots.reset_defaults()
default(yminorticks=1, minorgrid=true, xtickfontsize = xtickfontsize, legendfontsize=legendfontsize, framestyle=:box,
ytickfontsize = ytickfontsize, dpi=600, guidefontsize = guidefontsize, markerstrokewidth=markerstrokewidth, markersize=ms, left_margin=10mm, right_margin=10mm, bottom_margin=10mm, top_margin=10mm);
text_y_pos = 0.5; annotate_fs = 20; # annotation y_pos and fontsize

function compute_initial_data(L::Int, ε0_func::Function, v0_func::Function, ε0_prime_func::Function, v0_prime_func::Function, frame::Vector{Float64}, kreiss_coef::Float64, xmax::Float64)
    x_h = range(start=-xmax, stop=xmax, length=L) |> collect;
    ε0 = ε0_func.(x_h)
    v0 = v0_func.(x_h)
    ε0_prime = ε0_prime_func.(x_h)
    v0_prime = v0_prime_func.(x_h)
    pi1tt0 = zeros(L)
    pi1tx0 = zeros(L)
    vdot0, εdot0 = BDNKSystem.initial_conditions(v0, ε0, pi1tt0, pi1tx0, v0_prime, ε0_prime, frame...)
    # vdot0 = zeros(L)
    # εdot0 = zeros(L)
    return v0, ε0, vdot0, εdot0
end

function compute_solutions(type::String, ε0_func::Function, v0_func::Function, ε0_prime_func::Function, v0_prime_func::Function, L_array::Vector{Int64}, frame::Vector{Float64}, h_array::Vector{Float64}, k_array::Vector{Float64}, kreiss_coef::Float64, total_time::Float64, xmax::Float64, target_tol::Float64, max_tol::Float64, max_iter::Int,
    min_iter::Int, path::String)
    mkpath(path)
    for i in eachindex(h_array)
        v0, ε0, vdot0, εdot0 = compute_initial_data(L_array[i], ε0_func, v0_func, ε0_prime_func, v0_prime_func, frame, kreiss_coef, xmax)
        BDNKSystem.solve!(type, L_array[i], k_array[i], h_array[i], ε0, εdot0, v0, vdot0, frame..., kreiss_coef, xmax, total_time, target_tol, max_tol, max_iter, min_iter, path);
    end
end

function compute_convergence_ratio(type::String, h::Float64, k::Float64, frame::Vector{Float64}, kreiss_coef::Float64, xmax::Float64, total_time::Float64, path::String)
    sol_h = BDNKSystem.load_sol(type, k, h, frame..., kreiss_coef, xmax,  total_time, path)
    t_h = sol_h["t"][:]
    v_h = sol_h["v"][:, :]
    eps_h = sol_h["eps"][:, :]
    V_h = sol_h["vdot"][:, :]
    E_h = sol_h["epsdot"][:, :]

    
    sol_2h =BDNKSystem.load_sol(type, 2.0k, 2.0h, frame..., kreiss_coef, xmax,  total_time, path)
    t_2h = sol_2h["t"][:]
    v_2h = sol_2h["v"][:, :]
    eps_2h = sol_2h["eps"][:, :]
    V_2h = sol_2h["vdot"][:, :]
    E_2h = sol_2h["epsdot"][:, :]

    sol_4h = BDNKSystem.load_sol(type, 4.0k, 4.0h, frame..., kreiss_coef, xmax,  total_time, path)
    t_4h = sol_4h["t"][:]
    v_4h = sol_4h["v"][:, :]
    eps_4h = sol_4h["eps"][:, :]
    V_4h = sol_4h["vdot"][:, :]
    E_4h = sol_4h["epsdot"][:, :]

    x_h = sol_h["x"][:]
    x_2h = sol_2h["x"][:]
    x_4h = sol_4h["x"][:]

    close(sol_h)
    close(sol_2h)
    close(sol_4h)

    Q_v = zeros(length(t_4h))
    Q_eps = zeros(length(t_4h))
    Q_V = zeros(length(t_4h))
    Q_E = zeros(length(t_4h))

    for i=1:length(t_4h)
        time_idx_2h = 1 + 2 * (i - 1)
        idx_h = 1 + 4 * (i - 1)    
        # compute ||u^{4h} - u^{2h}||
        if x_4h != x_2h[1:2:end]
            throw(DomainError("x values are not equal for (2h, 4h) pair"))
        end
        abs_2h_4h_v = BDNKSystem.one_norm(v_4h[i, :] - v_2h[time_idx_2h, 1:2:end])
        abs_2h_4h_eps = BDNKSystem.one_norm(eps_4h[i, :] - eps_2h[time_idx_2h, 1:2:end])
        abs_2h_4h_V = BDNKSystem.one_norm(V_4h[i, :] - V_2h[time_idx_2h, 1:2:end])
        abs_2h_4h_E = BDNKSystem.one_norm(E_4h[i, :] - E_2h[time_idx_2h, 1:2:end])

        # compute ||u^{2h} - u^{h}||
        if x_2h != x_h[1:2:end] || x_4h != x_h[1:4:end]
            throw(DomainError("x values are not equal for (h, 2h) pair"))
        end

        abs_h_2h_v = BDNKSystem.one_norm(v_2h[time_idx_2h, :] - v_h[idx_h, 1:2:end])
        abs_h_2h_eps = BDNKSystem.one_norm(eps_2h[time_idx_2h, :] - eps_h[idx_h, 1:2:end])
        abs_h_2h_V = BDNKSystem.one_norm(V_2h[time_idx_2h, :] - V_h[idx_h, 1:2:end])
        abs_h_2h_E = BDNKSystem.one_norm(E_2h[time_idx_2h, :] - E_h[idx_h, 1:2:end])

        # abs_h_2h_v = BDNKSystem.one_norm(v_2h[time_idx_2h, :] - v_h[idx_h, 1:2:end])
        # abs_h_2h_eps = BDNKSystem.one_norm(eps_2h[time_idx_2h, :] - eps_h[idx_h, 1:2:end])
        # abs_h_2h_V = BDNKSystem.one_norm(V_2h[time_idx_2h, :] - V_h[idx_h, 1:2:end])
        # abs_h_2h_E = BDNKSystem.one_norm(E_2h[time_idx_2h, :] - E_h[idx_h, 1:2:end])


        Q_v[i] = abs_2h_4h_v / abs_h_2h_v
        Q_eps[i] = abs_2h_4h_eps / abs_h_2h_eps
        Q_V[i] = abs_2h_4h_V / abs_h_2h_V
        Q_E[i] = abs_2h_4h_E / abs_h_2h_E
    end
    return t_4h, Q_v, Q_eps, Q_V, Q_E
end

function plot_convergence_factor(type::String, L_array::Vector{Int64}, h_array::Vector{Float64}, k_array::Vector{Float64}, frame::Vector{Float64}, kreiss_coef::Float64, xmax::Float64, total_time::Float64, path::String)
    if length(L_array) != 5
        throw(DomainError("Require five grid sizes"))
    end

    time = []
    Q_v = []
    Q_eps = []
    Q_V = []
    Q_E = []

    for i=1:length(h_array)-2
        t_i, Q_v_i, Q_eps_i, Q_V_i, Q_E_i = compute_convergence_ratio(type, h_array[i], k_array[i], frame, kreiss_coef, xmax, total_time, path)
        push!(Q_v, Q_v_i)
        push!(Q_eps, Q_eps_i)
        push!(Q_V, Q_V_i)
        push!(Q_E, Q_E_i)
        push!(time, t_i)
    end

    Qv_plot = plot(time, Q_v, xlabel = "", ylabel = L"\mathrm{Q_{N}(t)}", size = (800, 400), color = [:blue :red :green], ylims = (0, 8), label = [L"N=%$(L_array[1])" L"N=%$(L_array[2])" L"N=%$(L_array[3])"], legend = :bottomleft, bottom_margin=-10mm, xformatter=:none)
    annotate!(Qv_plot, total_time*0.99, text_y_pos, text(L"\frac{||v^{4h}-v^{2h}||_{1}}{||v^{2h}-v^{h}||_{1}}", annotate_fs, color = :black, valign = :bottom, halign = :right))
    hline!([4], color = :black, linestyle = :dash, linewidth=1.5, label="")

    Qeps_plot = plot(time, Q_eps, xlabel = "", size = (800, 400), color = [:blue :red :green], legend=false, ylims = (0, 8), left_margin=-10mm, 
        xformatter=:none, yformatter=:none, bottom_margin=-10mm)
    annotate!(Qeps_plot, total_time*0.99, text_y_pos, text(L"\frac{||\epsilon^{4h}-\epsilon^{2h}||_{1}}{||\epsilon^{2h}-\epsilon^{h}||_{1}}", annotate_fs, color = :black, valign = :bottom, halign = :right))
    hline!([4], color = :black, linestyle = :dash, linewidth=1.5, label="")

    QV_plot = plot(time, Q_V, xlabel = "Time", ylabel = L"\mathrm{Q_{N}(t)}", size = (800, 400), color = [:blue :red :green], legend = false, ylims = (0, 8), left_margin=-10mm)
    annotate!(QV_plot, total_time*0.99, text_y_pos, text(L"\frac{||\dot{v}^{4h}-\dot{v}^{2h}||_{1}}{||\dot{v}^{2h}-\dot{v}^{h}||_{1}}", annotate_fs, color = :black, valign = :bottom, halign = :right))
    hline!([4], color = :black, linestyle = :dash, linewidth=1.5, label="")

    QE_plot = plot(time, Q_E, xlabel = "Time", size = (800, 400), color = [:blue :red :green], legend=false, ylims = (0, 8), left_margin=-10mm, yformatter=:none)
    annotate!(QE_plot, total_time*0.99, text_y_pos, text(L"\frac{||\dot{\epsilon}^{4h}-\dot{\epsilon}^{2h}||_{1}}{||\dot{\epsilon}^{2h}-\dot{\epsilon}^{h}||_{1}}", annotate_fs, color = :black, valign = :bottom, halign = :right))
    hline!([4], color = :black, linestyle = :dash, linewidth=1.5, label="")

    p = plot(Qv_plot, Qeps_plot, QV_plot, QE_plot, layout = (2, 2), size = (1200, 800))
    display(p)
end

function compute_independent_residual_ratio(type::String, h::Float64, k::Float64, frame::Vector{Float64}, kreiss_coef::Float64, xmax::Float64, total_time::Float64, path::String)
    sol_h = BDNKSystem.load_sol(type, k, h, frame..., kreiss_coef, xmax,  total_time, path)
    t_h = sol_h["t"][:]
    x_h = sol_h["x"][:]
    ind_resid_t_h = sol_h["ind_resid_t"][:, :]
    ind_resid_x_h = sol_h["ind_resid_x"][:, :]
    
    sol_2h = BDNKSystem.load_sol(type, 2.0k, 2.0h, frame..., kreiss_coef, xmax,  total_time, path)
    t_2h = sol_2h["t"][:]
    x_2h = sol_2h["x"][:]
    ind_resid_t_2h = sol_2h["ind_resid_t"][:, :]
    ind_resid_x_2h = sol_2h["ind_resid_x"][:, :]

    close(sol_h)
    close(sol_2h)

    if t_2h != t_h[1:2:end]
        throw(DomainError("t values do not overlap for (h, 2h) pair"))
    end

    if x_2h != x_h[1:2:end]
        throw(DomainError("x values do not overlap for (h, 2h) pair"))
    end

    t_ratio = zeros(length(t_2h)-2)
    x_ratio = zeros(length(t_2h)-2)
    for i in 1:length(t_2h)
        idx_h = 1 + 2 * (i - 1)
        if i != 1 && i != length(t_2h)
            t_ratio[i-1] = BDNKSystem.one_norm(ind_resid_t_2h[i-1, :]) / BDNKSystem.one_norm(ind_resid_t_h[idx_h-1, :])
            x_ratio[i-1] = BDNKSystem.one_norm(ind_resid_x_2h[i-1, :]) / BDNKSystem.one_norm(ind_resid_x_h[idx_h-1, :])
            # t_ratio[i-1] = BDNKSystem.one_norm(ind_resid_t_2h[i-1, :]) / BDNKSystem.one_norm(ind_resid_t_h[idx_h-1, :])
            # x_ratio[i-1] = BDNKSystem.one_norm(ind_resid_x_2h[i-1, :]) / BDNKSystem.one_norm(ind_resid_x_h[idx_h-1, :])
        end 
    end
    return t_2h[2:end-1], t_ratio, x_ratio
end

function plot_independent_residual_ratio(type::String, L_array::Vector{Int64}, h_array::Vector{Float64}, k_array::Vector{Float64}, frame::Vector{Float64}, kreiss_coef::Float64, xmax::Float64, total_time::Float64, path::String)
    if length(L_array) != 5
        throw(DomainError("Require five grid sizes"))
    end

    Q_ind_resid_t = [];
    Q_ind_resid_x = [];
    time_ind_resid = []

    for i=1:length(h_array)-2
        t_i, QN_t, QN_x = compute_independent_residual_ratio(type, h_array[i], k_array[i], frame, kreiss_coef, xmax, total_time, path)
        push!(Q_ind_resid_t, QN_t)
        push!(Q_ind_resid_x, QN_x)
        push!(time_ind_resid, t_i)
    end

    Q_ind_resid_t_plot = plot(time_ind_resid, Q_ind_resid_t, xlabel = "Time", ylabel = L"\mathrm{Q_{N}(t)}", size = (800, 400), color = [:blue :red :green], ylims = (0, 12), legend = false)
    annotate!(Q_ind_resid_t_plot, total_time*0.2, 11.9, text(L"\nabla_{a}T^{at}", annotate_fs, color = :black, valign = :top, halign = :right))

    Q_ind_resid_x_plot = plot(time_ind_resid, Q_ind_resid_x, xlabel = "Time", size = (800, 400), color = [:blue :red :green],label = [L"N=%$(L_array[1])" L"N=%$(L_array[2])" L"N=%$(L_array[3])"], legend = :bottomleft, ylims = (0, 12), left_margin=-10mm, yformatter=:none)
    annotate!(Q_ind_resid_x_plot, total_time*0.2, 11.9, text(L"\nabla_{a}T^{ax}", annotate_fs, color = :black, valign = :top, halign = :right))

    p = plot(Q_ind_resid_t_plot, Q_ind_resid_x_plot, layout = (1, 2), size = (1200, 400))
    display(p)
end


function plot_independent_resid(type::String, h::Float64, k::Float64, frame::Vector{Float64}, kreiss_coef::Float64, xmax::Float64, total_time::Float64, times::Vector{Float64}, path::String)
    sol_h = BDNKSystem.load_sol(type, k, h, frame..., kreiss_coef, xmax,  total_time, path)
    t_h = sol_h["t"][:]
    x_h = sol_h["x"][:]
    ind_resid_t_h = sol_h["ind_resid_t"][:, :]
    ind_resid_x_h = sol_h["ind_resid_x"][:, :]
    
    sol_2h = BDNKSystem.load_sol(type, 2.0k, 2.0h, frame..., kreiss_coef, xmax,  total_time, path)
    t_2h = sol_2h["t"][:]
    x_2h = sol_2h["x"][:]
    ind_resid_t_2h = sol_2h["ind_resid_t"][:, :]
    ind_resid_x_2h = sol_2h["ind_resid_x"][:, :]

    close(sol_h)
    close(sol_2h)
    ii = 1
    for time in times
        idx_h = argmin(abs.(t_h[2:end-1] .- time))
        idx_2h = argmin(abs.(t_2h[2:end-1] .- time))
        
        p1 = plot(x_h, ind_resid_t_h[idx_h, :], ylabel = L"|\mathcal{L}^{h}u^{h}|", size = (800, 400), xlabel = "",
                yscale=:log10, yticks = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0], ylims = (1e-10, 1e0), xformatter=:none, label=L"N=%$(length(x_h))", color=:blue, legend = :topright, title="Time = $(round(t_h[2:end-1][idx_h]))", titlefont=20, top_margin = 2mm, bottom_margin=-5mm)
        
#         ymax = maximum([maximum(ind_resid_t_h[idx_h, :]), maximum(ind_resid_t_2h[idx_2h, :])])
        ymax = 1e0
        
        annotate!(p1, x_h[1] + 0.02 * (x_h[end] - x_h[1]), 1e-1 * ymax, text(L"\nabla_{a}T^{at}", annotate_fs, color = :black, valign = :top, halign = :left))
        
        plot!(p1, x_2h, ind_resid_t_2h[idx_2h, :], label=L"N=%$(length(x_2h))", color=:red)

        p2 = plot(x_h, ind_resid_x_h[idx_h, :], ylabel = L"|\mathcal{L}^{h}u^{h}|", size = (800, 400), xlabel = L"x",
            yscale=:log10, yticks = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0], ylims = (1e-10, 1e0), color=:blue, legend = false, titlefont=20)
        
        plot!(p2, x_2h, ind_resid_x_2h[idx_2h, :], color=:red)
            
#         ymax = maximum([maximum(ind_resid_x_h[idx_h, :]), maximum(ind_resid_x_2h[idx_2h, :])])
        ymax = 1e0
        annotate!(p2, x_2h[1] + 0.02 * (x_2h[end] - x_2h[1]), 1e-1 * ymax, text(L"\nabla_{a}T^{ax}", annotate_fs, color = :black, valign = :top, halign = :left))

        p = plot(p1, p2, layout=(2, 1), size = (800, 800))
        savefig("plot_$ii).png")
        ii += 1
        display(p)
    end
end


end
end