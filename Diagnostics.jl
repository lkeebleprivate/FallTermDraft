module Diagnostics

module Euler
using Plots, Plots.Measures, LaTeXStrings

# plot attributes
left_margin=10mm; right_margin=10mm; bottom_margin=10mm; top_margin=6mm; 
xtickfontsize=12; ytickfontsize=12; guidefontsize=15; legendfontsize=12; ms=1.5; markerstrokewidth=0;
text_y_pos = 1e6; annotate_fs = 15;# annotation y_pos and fontsize
Plots.reset_defaults()
default(legendfontsize=legendfontsize, framestyle=:box, yminorticks=1, minorgrid=true, xtickfontsize = xtickfontsize,
ytickfontsize = ytickfontsize, dpi=600, guidefontsize = guidefontsize, markerstrokewidth=markerstrokewidth, markersize=ms, left_margin=10mm, right_margin=10mm, bottom_margin=10mm, top_margin=10mm);

plot11_width = 400; plot11_height = 300;
plot12_width = 800; plot12_height = 300;
plot21_width = 400; plo21_height = 600;
plot22_width = 800; plot22_height = 600;

function plot_independent_resid(sol, times::Vector{Float64}; save=false, xlims = (0.0, 0.0))
    t = sol["t"][:]
    x = sol["x"][:]
    ind_resid_t = sol["ind_resid_t"][:, :]
    ind_resid_x = sol["ind_resid_x"][:, :]
    for time in times
        idx = argmin(abs.(t[2:end-1] .- time))
        p = plot(x, ind_resid_t[idx, :], ylabel = L"|\mathcal{L}^{h}u^{h}|", xlims = xlims, size = (800, 400), xlabel = L"x", label=L"\nabla_{a}T^{at}", color=:blue, legend = :topright, title="Time = $(round(t[2:end-1][idx]))", titlefont=20,
        top_margin = 2mm)

        plot!(p, x, ind_resid_x[idx, :], label=L"\nabla_{a}T^{ax}", color= :red)
        if save
            savefig(p, "ind_resid_t_$(time).png")
        else
            display(p)
        end
    end
end


function plot_one_var(sol, times::Vector{Float64}; eps::Bool=false, v::Bool=false, time::Bool=false, first::Bool=false, second::Bool=false, third::Bool=false, xlims=(0.0, 0.0), ylims=(0.0, 0.0), size=(plot11_width, plot11_height), save=false)
    t = sol["t"][:]
    x = sol["x"][:]    
    if eps
        f = sol["eps"][:, :]
        f_label = L"\epsilon"
    elseif v
        f = sol["v"][:, :]
        f_label = L"v"
    else
        throw("Must set one of eps or v to true")
    end

    for T in times
        idx = argmin(abs.(t .- T))
        p1 = plot(x, f[idx, :], ylabel = f_label, size = size, xlabel = L"x", color=:blue, legend=false, right_margin = -1mm, top_margin = 1mm, suptitle = "Time = $(round(t[idx]))")
        
        if !isequal(xlims, (0.0, 0.0))
            plot!(p1, xlims = xlims)
        end
        
        if !isequal(ylims, (0.0, 0.0))
            plot!(p1, ylims = ylims)
        end
        
        if save
            savefig(p1, "plot_t_$(T).png")
        else
            display(p1)
        end
    end
end

function plot_solution(sol, times::Vector{Float64}; lims=false, eps_lims=(0.0, 0.0), v_lims = (0.0, 0.0))
    t = sol["t"][:]
    x = sol["x"][:]
    v = sol["v"][:, :]
    eps = sol["eps"][:, :]
   
    for time in times
        idx = argmin(abs.(t .- time))
        if lims
            p1 = plot(x, eps[idx, :], ylabel = L"\epsilon", size = (800, 400), xlabel = L"x", color=:blue, legend=false, right_margin = -5mm, top_margin = 1mm,
            ylims = eps_lims)
            p2 = plot(x, v[idx, :], ylabel = L"v", size = (800, 400), xlabel = L"x", color=:blue, legend=false, top_margin = 1mm,
            ylims = v_lims)
            
            p3 = plot(p1, p2, layout = (1, 2), size = (plot12_width, plot12_height), suptitle = "Time = $(round(t[idx]))")
            display(p3)
        else
            p1 = plot(x, eps[idx, :], ylabel = L"\epsilon", size = (800, 400), xlabel = L"x", color=:blue, legend=false, right_margin = -5mm, top_margin = 1mm)
            p2 = plot(x, v[idx, :], ylabel = L"v", size = (800, 400), xlabel = L"x", color=:blue, legend=false, top_margin = 1mm)
            
            p3 = plot(p1, p2, layout = (1, 2), size = (plot12_width, plot12_height), suptitle = "Time = $(round(t[idx]))")
            display(p3)
        end
    end
end

end
module BDNK
using Plots, Plots.Measures, LaTeXStrings

# plot attributes
left_margin=10mm; right_margin=10mm; bottom_margin=10mm; top_margin=6mm; 
xtickfontsize=12; ytickfontsize=12; guidefontsize=15; legendfontsize=12; ms=1.5; markerstrokewidth=0;
text_y_pos = 1e6; annotate_fs = 15;# annotation y_pos and fontsize
Plots.reset_defaults()
default(legendfontsize=legendfontsize, framestyle=:box, yminorticks=1, minorgrid=true, xtickfontsize = xtickfontsize,
ytickfontsize = ytickfontsize, dpi=600, guidefontsize = guidefontsize, markerstrokewidth=markerstrokewidth, markersize=ms, left_margin=10mm, right_margin=10mm, bottom_margin=10mm, top_margin=10mm);

plot11_width = 400; plot11_height = 300;
plot12_width = 800; plot12_height = 300;
plot21_width = 400; plo21_height = 600;
plot22_width = 800; plot22_height = 600;


function load_data(type::String, k::Float64, h::Float64, frame::Vector{Float64}, kreiss_coef::Float64, xmax::Float64,  total_times::Vector{Float64}, path::String)
    sol = BDNKSystem.load_sol(type, k, h, frame..., kreiss_coef, xmax,  total_time, path)
    t = sol["t"][:]
    x = sol["x"][:]
    v = sol["v"][:, :]
    eps = sol["eps"][:, :]
    V = sol["vdot"][:, :]
    E = sol["epsdot"][:, :]
    T_tt_0 = sol["T_tt_0"][:, :]
    T_tt_1 = sol["T_tt_1"][:, :]
    T_xx_0 = sol["T_xx_0"][:, :]
    T_xx_1 = sol["T_xx_1"][:, :]
    vel_dot_T = sol["vel_dot_T"][:, :]    # inner production u_{a} u_{b} T^{ab} to check weak energy condition
    newton_residuals = sol["residuals"][:]
    ind_resid_t = sol["ind_resid_t"][:, :]
    ind_resid_x = sol["ind_resid_x"][:, :]
    num_iterations = sol["num_iterations"][:]
    close(sol)
    return t, x, v, eps, V, E, T_tt_0, T_tt_1, T_xx_0, T_xx_1, vel_dot_T, newton_residuals, ind_resid_t, ind_resid_x, num_iterations
end

function plot_independent_resid(sol, times::Vector{Float64}; save=false, xlims = (0.0, 0.0))
    t = sol["t"][:]
    x = sol["x"][:]
    ind_resid_t = sol["ind_resid_t"][:, :]
    ind_resid_x = sol["ind_resid_x"][:, :]
    for time in times
        idx = argmin(abs.(t[2:end-1] .- time))
        p = plot(x, ind_resid_t[idx, :], ylabel = L"|\mathcal{L}^{h}u^{h}|", xlims = xlims, size = (800, 400), xlabel = L"x", label=L"\nabla_{a}T^{at}", color=:blue, legend = :topright, title="Time = $(round(t[2:end-1][idx]))", titlefont=20,
        top_margin = 2mm)

        plot!(p, x, ind_resid_x[idx, :], label=L"\nabla_{a}T^{ax}", color= :red)
        if save
            savefig(p, "ind_resid_t_$(time).png")
        else
            display(p)
        end
    end
end


function plot_one_var(sol, times::Vector{Float64}; eps::Bool=false, v::Bool=false, time::Bool=false, first::Bool=false, second::Bool=false, third::Bool=false, xlims=(0.0, 0.0), ylims=(0.0, 0.0), size=(plot11_width, plot11_height), save=false)
    t = sol["t"][:]
    x = sol["x"][:]    
    if eps
        if time
            f = sol["epsdot"][:, :]
            f_label = L"\dot{\epsilon}"
        elseif first
            f = sol["eps_prime_1"][:, :]
            f_label = L"\epsilon^{\prime}"
        elseif second
            f = sol["eps_prime_2"][:, :]
            f_label = L"\epsilon^{\prime\prime}"
        elseif third
            f = sol["eps_prime_3"][:, :]
            f_label = L"\epsilon^{\prime\prime\prime}"
        else
            f = sol["eps"][:, :]
            f_label = L"\epsilon"
        end
    elseif v
        if time
            f = sol["vdot"][:, :]
            f_label = L"\dot{v}"
        elseif first
            f = sol["v_prime_1"][:, :]
            f_label = L"v^{\prime}"
        elseif second
            f = sol["v_prime_2"][:, :]
            f_label = L"v^{\prime\prime}"
        elseif third
            f = sol["v_prime_3"][:, :]
            f_label = L"v^{\prime\prime\prime}"
        else 
            f = sol["v"][:, :]
            f_label = L"v"
        end
    else
        throw("Must set one of eps or v to true")
    end

    for T in times
        idx = argmin(abs.(t .- T))
        p1 = plot(x, f[idx, :], ylabel = f_label, size = size, xlabel = L"x", color=:blue, legend=false, right_margin = -1mm, top_margin = 1mm, suptitle = "Time = $(round(t[idx]))")
        
        if !isequal(xlims, (0.0, 0.0))
            plot!(p1, xlims = xlims)
        end
        
        if !isequal(ylims, (0.0, 0.0))
            plot!(p1, ylims = ylims)
        end
        
        if save
            savefig(p1, "plot_t_$(T).png")
        else
            display(p1)
        end
    end
end

function plot_eps_rescaled(sol, times::Vector{Float64}, A::Float64, delta::Float64; xlims=(0.0, 0.0), ylims=(0.0, 0.0), size=(plot11_width, plot11_height), save=false)
    t = sol["t"][:]
    x = sol["x"][:]    

    f = sol["eps"][:, :]
    f_label = L"\frac{1}{A}(\epsilon-\delta)"

    for T in times
        idx = argmin(abs.(t .- T))

        y = @. ((f[idx, :] - delta) / A)

        p1 = plot(x, y, ylabel = f_label, size = size, xlabel = L"x", color=:blue, legend=false, right_margin = -1mm, top_margin = 1mm, suptitle = "Time = $(round(t[idx]))")   

        if !isequal(xlims, (0.0, 0.0))
            plot!(p1, xlims = xlims)
        end
        
        if !isequal(ylims, (0.0, 0.0))
            plot!(p1, ylims = ylims)
        end
        
        if save
            savefig(p1, "plot_t_$(T).png")
        else
            display(p1)
        end
    end
end

function plot_solution(sol, times::Vector{Float64}; lims=false, eps_lims=(0.0, 0.0), v_lims = (0.0, 0.0), epsdot_lims=(0.0,0.0), vdot_lims=(0.0,0.0))
    t = sol["t"][:]
    x = sol["x"][:]
    v = sol["v"][:, :]
    eps = sol["eps"][:, :]
    V = sol["vdot"][:, :]
    E = sol["epsdot"][:, :]

    for time in times
        idx = argmin(abs.(t .- time))
        if lims
            p1 = plot(x, eps[idx, :], ylabel = L"\epsilon", size = (800, 400), xlabel = "", color=:blue, legend=false, bottom_margin = -5mm, right_margin = -5mm, top_margin = 1mm, xformatter=:none,
            ylims = eps_lims)
            p2 = plot(x, v[idx, :], ylabel = L"v", size = (800, 400), xlabel = "", color=:blue, legend=false, bottom_margin = -5mm, top_margin = 1mm, xformatter=:none,
            ylims = v_lims)
            p3 = plot(x, E[idx, :], ylabel = L"\dot{\epsilon}", size = (800, 400), xlabel = L"x", color=:blue, legend=false, right_margin = -5mm, ylims = epsdot_lims)
            p4 = plot(x, V[idx, :], ylabel = L"\dot{v}", size = (800, 400), xlabel = L"x", color=:blue, legend=false, ylims = vdot_lims)

            p5 = plot(p1, p2, p3, p4, layout = (2, 2), size = (plot22_width, plot22_height), suptitle = "Time = $(round(t[idx]))")
            display(p5)
        else
            p1 = plot(x, eps[idx, :], ylabel = L"\epsilon", size = (800, 400), xlabel = "", color=:blue, legend=false, bottom_margin = -5mm, right_margin = -5mm, top_margin = 1mm, xformatter=:none)
            p2 = plot(x, v[idx, :], ylabel = L"v", size = (800, 400), xlabel = "", color=:blue, legend=false, bottom_margin = -5mm, top_margin = 1mm, xformatter=:none)
            p3 = plot(x, E[idx, :], ylabel = L"\dot{\epsilon}", size = (800, 400), xlabel = L"x", color=:blue, legend=false, right_margin = -5mm)
            p4 = plot(x, V[idx, :], ylabel = L"\dot{v}", size = (800, 400), xlabel = L"x", color=:blue, legend=false)

            p5 = plot(p1, p2, p3, p4, layout = (2, 2), size = (plot22_width, plot22_height), suptitle = "Time = $(round(t[idx]))")
            display(p5)
        end
    end
end

function plot_derivs(sol, times::Vector{Float64}; eps::Bool=false, v::Bool=false)
    t = sol["t"][:]
    x = sol["x"][:]
    if eps
        f = sol["eps"][:, :]
        f_prime_1 = sol["eps_prime_1"][:, :]
        f_prime_2 = sol["eps_prime_2"][:, :]
        f_prime_3 = sol["eps_prime_3"][:, :]
        f_labels = [L"\epsilon", L"\epsilon^{\prime}", L"\epsilon^{\prime\prime}", L"\epsilon^{\prime\prime\prime}"]
    else
        f = sol["v"][:, :]
        f_prime_1 = sol["v_prime_1"][:, :]
        f_prime_2 = sol["v_prime_2"][:, :]
        f_prime_3 = sol["v_prime_3"][:, :]
        f_labels = [L"v", L"v^{\prime}", L"v^{\prime\prime}", L"v^{\prime\prime\prime}"]
    end

    for time in times
        idx = argmin(abs.(t .- time))
        p1 = plot(x, f[idx, :], ylabel = f_labels[1], size = (800, 400), xlabel = "", color=:blue, legend=false, bottom_margin = -5mm, right_margin = -5mm, top_margin = 1mm, xformatter=:none)
        p2 = plot(x, f_prime_1[idx, :], ylabel = f_labels[2], size = (800, 400), xlabel = "", color=:blue, legend=false, bottom_margin = -5mm, top_margin = 1mm, xformatter=:none)
        p3 = plot(x, f_prime_2[idx, :], ylabel = f_labels[3], size = (800, 400), xlabel = L"x", color=:blue, legend=false, right_margin = -5mm)
        p4 = plot(x, f_prime_3[idx, :], ylabel = f_labels[4], size = (800, 400), xlabel = L"x", color=:blue, legend=false)
        p5 = plot(p1, p2, p3, p4, layout = (2, 2), size = (plot22_width, plot22_height), suptitle = "Time = $(round(t[idx]))")
        display(p5)
    end
end

function plot_one_deriv(sol, times::Vector{Float64}; eps::Bool=false, v::Bool=false, first::Bool=false, second::Bool=false, third::Bool=false)
    t = sol["t"][:]
    x = sol["x"][:]
    if !first & !second & !third
        throw("Choose one of first, second, or third to be true")
    elseif first & second || first & third || second & third
        throw("Choose only one of first, second, or third to be true")
    end
    
    if eps
        f = sol["eps"][:, :]
        if first
            f_prime = sol["eps_prime_1"][:, :]
            f_labels = [L"\epsilon", L"\epsilon^{\prime}"]
        elseif second
            f_prime = sol["eps_prime_2"][:, :]
            f_labels = [L"\epsilon", L"\epsilon^{\prime\prime}"]
        elseif third
            f_prime = sol["eps_prime_3"][:, :]
            f_labels = [L"\epsilon", L"\epsilon^{\prime\prime\prime}"]
        end
    else
        f = sol["v"][:, :]
        if first
            f_prime = sol["v_prime_1"][:, :]
            f_labels = [L"v", L"v^{\prime}"]
        elseif second
            f_prime = sol["v_prime_2"][:, :]
            f_labels = [L"v", L"v^{\prime\prime}"]
        elseif third
            f_prime = sol["v_prime_3"][:, :]
            f_labels = [L"v", L"v^{\prime\prime\prime}"]
        end
    end

    for time in times
        idx = argmin(abs.(t .- time))
        p1 = plot(x, f[idx, :], ylabel = f_labels[1], size = (800, 400), xlabel = L"x", color=:blue, legend=false, right_margin = -1mm, top_margin = 1mm)
        p2 = plot(x, f_prime[idx, :], ylabel = f_labels[2], size = (800, 400), xlabel = L"x", color=:blue, legend=false, top_margin = 1mm)
        p3 = plot(p1, p2, layout = (1, 2), size = (plot12_width, plot12_height), suptitle = "Time = $(round(t[idx]))")
        display(p3)
    end
end

function plot_SE_tensor(sol, times::Vector{Float64})
    t = sol["t"][:]
    x = sol["x"][:]
    T_tt_0 = sol["T_tt_0"][:, :]
    T_tt_1 = sol["T_tt_1"][:, :]
    T_xx_0 = sol["T_xx_0"][:, :]
    T_xx_1 = sol["T_xx_1"][:, :]

    for time in times
        idx = argmin(abs.(t .- time))
        # tt component
        p1 = plot(x, T_tt_0[idx, :], ylabel = L"T^{tt}", size = (800, 400), xlabel = "", label=L"T^{tt}_{(0)}", color=:blue, legend = :topright)
        plot!(p1, x, T_tt_1[idx, :], label=L"T^{tt}_{(1)}", color=:red, xformatter=:none, bottom_margin = -5mm, right_margin = -5mm, top_margin=1mm)
        p2 = plot(x, T_tt_0[idx, :] .+ T_tt_1[idx, :], ylabel = "", size = (800, 400), xlabel = "", color=:black, label=L"T^{tt}_{(0)} + T^{tt}_{(1)}", legend = :topright, 
        xformatter=:none, bottom_margin = -5mm, top_margin=1mm)

        # xx component
        p3 = plot(x, T_xx_0[idx, :], ylabel = L"T^{xx}", size = (800, 400), xlabel = L"x", label=L"T^{xx}_{(0)}", color=:blue, legend = :topright, right_margin = -5mm)
        plot!(p3, x, T_xx_1[idx, :], label=L"T^{xx}_{(1)}", color=:red)
        p4 = plot(x, T_xx_0[idx, :] .+ T_xx_1[idx, :], ylabel = "", size = (800, 400), xlabel = L"x", color=:black, label=L"T^{xx}_{(0)} + T^{xx}_{(1)}", legend = :topright)

        p5 = plot(p1, p2, p3, p4, layout = (2, 2), size = (plot22_width, plot22_height), suptitle = "Time = $(round(t[idx]))")
        display(p5)
    end
end

function plot_validity_checks(sol, times::Vector{Float64})
    t = sol["t"][:]
    x = sol["x"][:]
    T_tt_0 = sol["T_tt_0"][:, :]
    T_tt_1 = sol["T_tt_1"][:, :]
    vel_dot_T = sol["vel_dot_T"][:, :]

    for time in times
        idx = argmin(abs.(t .- time))
        SE_ratio = @. abs(T_tt_1[idx, :] / T_tt_0[idx, :]) + 1e-18 # add 1e-18 since log scale ill defined when ratio is zero
        p1 = plot(x, SE_ratio,  yscale=:log10, ylabel = "", size = (800, 400), xlabel = L"x", legend = false, color=:blue,
        title = L"T^{tt}_{1} / T^{tt}_{0}", titlefont=20, right_margin = -5mm, ylims = (1e-14, 1e2), yticks=[1e-14, 1e-10, 1e-6, 1e-2, 1e2])

        # weak energy check: u_{a} u_{b} T^{ab}
        p2 = plot(x, vel_dot_T[idx, :], ylabel = "", title = L"u_{a}u_{b}T^{ab}", titlefont=20, size = (800, 400), xlabel = L"x", legend = false, color=:blue, ylims = (-1, 1))

        p3 = plot(p1, p2, layout = (1, 2), size = (plot12_width, plot12_height), suptitle = "Time = $(round(t[idx]))")
        display(p3)
    end
end

function plot_all(sol, time::Float64)
    plot_solution(sol, [time])
    plot_SE_tensor(sol, [time])
    plot_validity_checks(sol, [time])
    plot_independent_resid(sol, [time])
end

end
end