module BDNKODEs
using StaticArrays
using DifferentialEquations

# Sqrt(x) = abs(x) < 1e-8 ? 0.0 : sqrt(x) 
Sqrt(x) = sqrt(x)

function right_states(vL::Float64, εL::Float64)
    εR = εL * (9.0 * vL^2 - 1.0) / 3.0 / (1.0 - vL^2)
    vR = 1.0 / 3.0 / vL
    return vR, εR
end

tt_zeroth_order(ε::Float64, v::Float64) = -0.3333333333333333*((3 + v^2)*ε)/(-1 + v^2)
tx_zeroth_order(ε::Float64, v::Float64) = (4*v*ε)/(3 - 3*v^2)
xx_zeroth_order(ε::Float64, v::Float64) = ((1 + 3*v^2)*ε)/(3 - 3*v^2)

tt_first_order(ε::Float64, v::Float64, εprime::Float64, vprime::Float64, η0::Float64, λ0::Float64, χ0::Float64) = (-3*εprime*v*(-1 + v^2)*(2*λ0 + 3*χ0 + χ0*v^2) + 4*(3*χ0 + (-4*η0 + 6*λ0 + χ0)*v^2)*vprime*ε)/(12.0*(1 - v^2)^2.5*ε^0.25)
tx_first_order(ε::Float64, v::Float64, εprime::Float64, vprime::Float64, η0::Float64, λ0::Float64, χ0::Float64) = (-3*εprime*(-1 + v^2)*(λ0 + (λ0 + 4*χ0)*v^2) + 4*v*(-4*η0 + 3*λ0 + 4*χ0 + 3*λ0*v^2)*vprime*ε)/(12.0*(1 - v^2)^2.5*ε^0.25)
xx_first_order(ε::Float64, v::Float64, εprime::Float64, vprime::Float64, η0::Float64, λ0::Float64, χ0::Float64) = (-3*εprime*v*(-1 + v^2)*(2*λ0 + χ0 + 3*χ0*v^2) + 4*(-4*η0 + χ0 + 3*(2*λ0 + χ0)*v^2)*vprime*ε)/(12.0*(1 - v^2)^2.5*ε^0.25)

c1(η0::Float64, λ0::Float64, χ0::Float64) = Sqrt(((2*η0 + λ0)*χ0 + 2*Sqrt(η0*χ0*(λ0^2 + (η0 + λ0)*χ0)))/(3*λ0*χ0))
c2(η0::Float64, λ0::Float64, χ0::Float64) = Sqrt(((2*η0 + λ0)*χ0 - 2*Sqrt(η0*χ0*(λ0^2 + (η0 + λ0)*χ0)))/(3*λ0*χ0))
c3(η0::Float64, λ0::Float64, χ0::Float64) = -Sqrt(((2*η0 + λ0)*χ0 + 2*Sqrt(η0*χ0*(λ0^2 + (η0 + λ0)*χ0)))/(3*λ0*χ0))
c4(η0::Float64, λ0::Float64, χ0::Float64) = -Sqrt(((2*η0 + λ0)*χ0 - 2*Sqrt(η0*χ0*(λ0^2 + (η0 + λ0)*χ0)))/(3*λ0*χ0))

εprime(ε, v, c1::Float64, c2::Float64, c3::Float64, c4::Float64, C1::Float64, C2::Float64, η0::Float64, λ0::Float64, χ0::Float64) = (4*Sqrt(1 - v^2)*ε^0.25*(C1*(4*η0 - χ0) + v*(C2*(-4*η0 + 3*λ0 + 4*χ0) -
3*C1*(2*λ0 + χ0)*v - (4*η0 + λ0)*ε + 3*λ0*v^2*(C2 + ε))))/(9.0*λ0*χ0*(c1 - v)*(-c2 + v)*(-c3 + v)*(-c4 + v))

vprime(ε, v, c1::Float64, c2::Float64, c3::Float64, c4::Float64, C1::Float64, C2::Float64, η0::Float64, λ0::Float64, χ0::Float64) = ((1 - v^2)^1.5*(3*C1*(2*λ0 + χ0)*v + 9*C1*χ0*v^3 + λ0*(-3*C2 + ε) - 3*v^2*(C2*(λ0 + 4*χ0) +
λ0*ε)))/(9.0*λ0*χ0*(c1 - v)*(-c2 + v)*(-c3 + v)*(-c4 + v)*ε^0.75)

# initial conditions for bound kerr orbits starting in equatorial plane
function ics(εL::Float64, vL::Float64)
    return @SArray [εL, vL]
end

# equation for ODE solver
function BDNK_Eqns(u, params, x)
    @SArray [εprime(u..., params...), vprime(u..., params...)]
end

function compute_ODE_params(εL::Float64, vL::Float64, η0::Float64, λ0::Float64, χ0::Float64)
    c1 = BDNKODEs.c1(η0, λ0, χ0)
    c2 = BDNKODEs.c2(η0, λ0, χ0)
    c3 = BDNKODEs.c3(η0, λ0, χ0)
    c4 = BDNKODEs.c4(η0, λ0, χ0)

    # initialize BDNK tensor to zero
    C1 = tx_zeroth_order(εL, vL)
    C2 = xx_zeroth_order(εL, vL)
    return @SArray [c1, c2, c3, c4, C1, C2, η0, λ0, χ0]
end

function compute_Txx(sol, εL::Float64, vL::Float64, η0::Float64, λ0::Float64, χ0::Float64)
    eps = sol[1, :];
    v = sol[2, :];
    ode_params = compute_ODE_params(εL, vL, η0, λ0, χ0);

    Txx0 = zeros(length(eps))
    Txx1 = zeros(length(eps))
    Ttx0 = zeros(length(eps))
    Ttx1 = zeros(length(eps))

    for i in eachindex(eps)
        Txx0[i] = xx_zeroth_order(eps[i], v[i]);
        Ttx0[i] = tx_zeroth_order(eps[i], v[i]);

        εprime_i = εprime(eps[i], v[i], ode_params...)
        vprime_i = vprime(eps[i], v[i], ode_params...)
        Txx1[i] = xx_first_order(eps[i], v[i], εprime_i, vprime_i, η0, λ0, χ0);
        Ttx1[i] = tx_first_order(eps[i], v[i], εprime_i, vprime_i, η0, λ0, χ0);
    end
    return Txx0, Txx1, Ttx0, Ttx1
end

function compute_frame(η_over_s::Float64, frame::String)
    e0 = 10.0
    η0 = 4 * (e0)^(1/4) * η_over_s / 3
    if frame == "A"
        frameA = [η0, 25 * η0 / 3, 25 * η0 / 2]
        return frameA
    elseif frame == "B"
        frameB = [η0, 25 * η0 / 7, 25 * η0 / 4]
        return frameB
    else
        throw(ArgumentError("frame must be either A or B"))
    end
end

function RK4_solve(εL::Float64, vL::Float64, x0::Float64, x1::Float64, N::Int64, η0::Float64, λ0::Float64, χ0::Float64)
    ode_params = compute_ODE_params(εL, vL, η0, λ0, χ0)
    u0 = ics(εL, vL)
    x = zeros(N) * NaN
    sol = zeros(2, N) * NaN
    sol[:, 1] .= u0
    h = (x1 - x0) / (N - 1)
    x_n = x0
    try
        for i in 1:N-1
            x[i] = x_n

            k1 = h * BDNK_Eqns(sol[:, i], ode_params, x_n)
            k2 = h * BDNK_Eqns(sol[:, i] + 0.5 * k1, ode_params, x_n + 0.5 * h)
            k3 = h * BDNK_Eqns(sol[:, i] + 0.5 * k2, ode_params, x_n + 0.5 * h)
            k4 = h * BDNK_Eqns(sol[:, i] + k3, ode_params, x_n + h)
            sol[:, i + 1] .= sol[:, i] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            x_n += h
        end
        x[N] = x_n
    catch e
        println("Error: $e")
        return x, sol
    end

    return x, sol
end


# Condition tested at each solver step
function stop_condition(u, t, integrator)
    (u[1] < 0.001) || (u[2] > 0.9999999)
end

# Action when condition is true
function stop_affect!(integrator)
    terminate!(integrator)
end


function compute_solution(εL::Float64, vL::Float64, η_over_s::Float64, L0::Int64, x0::Float64, x1::Float64, frame::String, type::String; maxiters::Int64 = Int(1e5), ode_method = Rodas4P(), target_tol::Float64 = 1e-8, max_tol::Float64 = 1e-8, max_iter::Int = 10, min_iter::Int = 1, warn::Bool=true)
    #################### PARAMETERS ###################
    frame = compute_frame(η_over_s, frame)
    vR, εR = right_states(vL, εL)

    #################### SOLVE TIME-INDEPENDENT ODE ###################
    if type == "Julia DE" # inbuilt julia DifferentialEquations
        Δxi = 1e-5; reltol = 1e-12; abstol = 1e-12;
        params = BDNKODEs.compute_ODE_params(εL, vL, frame...);
        ics = BDNKODEs.ics(εL, vL);
        xspan = (x0, x1); saveat_x = range(start=xspan[1], length=L0, stop=xspan[2]);
        prob = ODEProblem(BDNKODEs.BDNK_Eqns, ics, xspan, params);
        cb = DiscreteCallback(stop_condition, stop_affect!)
        sol = solve(prob, ode_method, adaptive=true, reltol = reltol, abstol = abstol, saveat=saveat_x, maxiters=maxiters, callback = cb); # dt = Δxi

        # deconstruct solution
        x = sol.t;
    elseif type == "RK4" # custom RK4 solver
        x, sol = RK4_solve(εL, vL, x0, x1, L0, frame...);
    elseif type == "CN" # Crank Nicholson solver
        x, sol = CN_solve(εL, vL, x0, x1, L0, frame..., target_tol, max_tol, max_iter, min_iter);
    else
        throw(ArgumentError("type must be either Julia DE, RK4 or CN"))
    end

    isapprox(x[end], x1, atol=1e-5) ? success = true : success = false
    eps = sol[1, :];
    v = sol[2, :];
    Txx0, Txx1, Ttx0, Ttx1 = BDNKODEs.compute_Txx(sol, εL, vL, frame...);

    if warn && !success
        @warn "Domain boundary v ≤ 1, ε ≥ 0 reached at x=$(x[end]). Stopping solver."
    end

    return x, eps, v, Txx0, Txx1, Ttx0, Ttx1, εR, vR, success
end


@views function RK4_convergence(εL::Float64, vL::Float64, x0::Float64, x1::Float64, N_MAX::Int64, η_over_s::Float64, frame::String)
    η0, λ0, χ0 = compute_frame(η_over_s, frame)
    N1 = (N_MAX - 1) ÷ 4 + 1
    N2 = (N_MAX - 1) ÷ 2 + 1
    N3 = N_MAX

    x_1, sol_1 = RK4_solve(εL, vL, x0, x1, N1, η0, λ0, χ0)
    x_2, sol_2 = RK4_solve(εL, vL, x0, x1, N2, η0, λ0, χ0)
    x_3, sol_3 = RK4_solve(εL, vL, x0, x1, N3, η0, λ0, χ0)

    # desample to lowest resolution grid
    x_2 = x_2[1:2:end]; x_3 = x_3[1:4:end];

    if !(x_1 == x_2 == x_3)
        throw(ArgumentError("x_1, x_2, x_3 must be equal"))
    end

    sol_2 = sol_2[:, 1:2:end]; sol_3 = sol_3[:, 1:4:end];

    Q_eps = @. (abs((sol_3[1, :] - sol_2[1, :]) / (sol_2[1, :] - sol_1[1, :])))
    Q_v = @. (abs((sol_3[2, :] - sol_2[2, :]) / (sol_2[2, :] - sol_1[2, :])))
    return x_1, Q_eps, Q_v
end

### CRANK NICHOLSON SOLVER ###
function CN_solve(εL::Float64, vL::Float64, x0::Float64, x1::Float64, N::Int64, η0::Float64, λ0::Float64, χ0::Float64, target_tol::Float64, max_tol::Float64, max_iter::Int, min_iter::Int)
    x = zeros(N) * NaN
    ode_params = compute_ODE_params(εL, vL, η0, λ0, χ0)[5:end]
    u0 = ics(εL, vL)
    sol = zeros(2, N) * NaN
    sol[:, 1] .= u0

    h = (x1 - x0) / (N - 1)
    jac = @MArray zeros(2, 2)
    Func = zeros(2)
    stop = false
    try
        for n in 1:N-1
            x[n] = x0 + (n - 1) * h
            ε0 = sol[1, n]
            v0 = sol[2, n]
            ε1 = ε0
            v1 = v0
            compute_function_vector!(Func, h, v0, v1, ε0, ε1, ode_params...)
            for step in 1:max_iter
                compute_jacobian!(jac, h, v0, v1, ε0, ε1, ode_params...)
                Δu = jac \ -Func
                ε1 += Δu[1]
                v1 += Δu[2]
                compute_function_vector!(Func, h, v0, v1, ε0, ε1, ode_params...)

                resid = maximum(abs.(Func))
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

            sol[1, n + 1] = ε1
            sol[2, n + 1] = v1
        end
    catch e
        println("Error: $e")
        return x, sol
    end

    x[end] = x1

    return x, sol
end

@views function CN_convergence(εL::Float64, vL::Float64, x0::Float64, x1::Float64, N_MAX::Int64, η_over_s::Float64, frame::String, target_tol::Float64, max_tol::Float64, max_iter::Int, min_iter::Int)
    η0, λ0, χ0 = compute_frame(η_over_s, frame)
    N1 = (N_MAX - 1) ÷ 4 + 1
    N2 = (N_MAX - 1) ÷ 2 + 1
    N3 = N_MAX

    x_1, sol_1 = CN_solve(εL, vL, x0, x1, N1, η0, λ0, χ0, target_tol, max_tol, max_iter, min_iter)
    x_2, sol_2 = CN_solve(εL, vL, x0, x1, N2, η0, λ0, χ0, target_tol, max_tol, max_iter, min_iter)
    x_3, sol_3 = CN_solve(εL, vL, x0, x1, N3, η0, λ0, χ0, target_tol, max_tol, max_iter, min_iter)

    # desample to lowest resolution grid
    x_2 = x_2[1:2:end]; x_3 = x_3[1:4:end];

    if !(x_1 == x_2 == x_3)
        throw(ArgumentError("x_1, x_2, x_3 must be equal"))
    end

    sol_2 = sol_2[:, 1:2:end]; sol_3 = sol_3[:, 1:4:end];

    Q_eps = @. (abs((sol_3[1, :] - sol_2[1, :]) / (sol_2[1, :] - sol_1[1, :])))
    Q_v = @. (abs((sol_3[2, :] - sol_2[2, :]) / (sol_2[2, :] - sol_1[2, :])))
    return x_1, Q_eps, Q_v
end

function compute_function_vector!(Func::Vector{Float64}, h::Float64, v0::Float64, v1::Float64, ε0::Float64, ε1::Float64, C1::Float64, C2::Float64, η0::Float64, λ0::Float64, χ0::Float64)
    Func[1] = BDNKODEs.F(h, v0, v1, ε0, ε1, C1, C2, η0, λ0, χ0)
    Func[2] = BDNKODEs.G(h, v0, v1, ε0, ε1, C1, C2, η0, λ0, χ0)
end


function compute_jacobian!(J::MMatrix{2, 2}, h::Float64, v0::Float64, v1::Float64, ε0::Float64, ε1::Float64, C1::Float64, C2::Float64, η0::Float64, λ0::Float64, χ0::Float64)
    J[1, 1] = BDNKODEs.Jacobian_F_eps(h, v0, v1, ε0, ε1, C1, C2, η0, λ0, χ0)
    J[1, 2] = BDNKODEs.Jacobian_F_v(h, v0, v1, ε0, ε1, C1, C2, η0, λ0, χ0)
    J[2, 1] = BDNKODEs.Jacobian_G_eps(h, v0, v1, ε0, ε1, C1, C2, η0, λ0, χ0)
    J[2, 2] = BDNKODEs.Jacobian_G_v(h, v0, v1, ε0, ε1, C1, C2, η0, λ0, χ0)
end

@inline function F(h::Float64, v0::Float64, v1::Float64, ε0::Float64, ε1::Float64, C1::Float64, C2::Float64, η0::Float64, λ0::Float64, χ0::Float64)
    return (-ε0 + ε1)/h + (2*2^0.75*Sqrt(1 - (v0 + v1)^2/4.)*(ε0 + ε1)^0.25*(C1*(4*η0 - χ0) + ((v0 + v1)*(C2*(-4*η0 + 3*λ0 + 4*χ0) - (3*C1*(2*λ0 + χ0)*(v0 + v1))/2. - ((4*η0 + λ0)*(ε0 + ε1))/2. + (3*λ0*(v0 + v1)^2*(2*C2 + ε0 + ε1))/8.))/2.))/(λ0*(-4*η0 + χ0) - (3*(2*η0 + λ0)*χ0*(v0 + v1)^2)/2. + (9*λ0*χ0*(v0 + v1)^4)/16.)
end

@inline function G(h::Float64, v0::Float64, v1::Float64, ε0::Float64, ε1::Float64, C1::Float64, C2::Float64, η0::Float64, λ0::Float64, χ0::Float64)
    return (-v0 + v1)/h + ((1 - (v0 + v1)^2/4.)^1.5*(12*C1*(2*λ0 + χ0)*(v0 + v1) + 9*C1*χ0*(v0 + v1)^3 + 4*λ0*(-6*C2 + ε0 + ε1) - 3*(v0 + v1)^2*(2*C2*(λ0 + 4*χ0) + λ0*(ε0 + ε1))))/(4.0*2^0.25*(λ0*(-4*η0 + χ0) - (3*(2*η0 + λ0)*χ0*(v0 + v1)^2)/2. + (9*λ0*χ0*(v0 + v1)^4)/16.)*(ε0 + ε1)^0.75)
end

@inline function Jacobian_F_eps(h::Float64, v0::Float64, v1::Float64, ε0::Float64, ε1::Float64, C1::Float64, C2::Float64, η0::Float64, λ0::Float64, χ0::Float64)
    return 1/h + (2^0.75*(v0 + v1)*Sqrt(1 - (v0 + v1)^2/4.)*(-2*η0 - λ0/2. + (3*λ0*(v0 + v1)^2)/8.)*(ε0 + ε1)^0.25)/(λ0*(-4*η0 + χ0) - (3*(2*η0 + λ0)*χ0*(v0 + v1)^2)/2. + (9*λ0*χ0*(v0 + v1)^4)/16.) + (Sqrt(1 - (v0 + v1)^2/4.)*(C1*(4*η0 - χ0) + ((v0 + v1)*(C2*(-4*η0 + 3*λ0 + 4*χ0) - (3*C1*(2*λ0 + χ0)*(v0 + v1))/2. - ((4*η0 + λ0)*(ε0 + ε1))/2. + (3*λ0*(v0 + v1)^2*(2*C2 + ε0 + ε1))/8.))/2.))/(2^0.25*(λ0*(-4*η0 + χ0) - (3*(2*η0 + λ0)*χ0*(v0 + v1)^2)/2. + (9*λ0*χ0*(v0 + v1)^4)/16.)*(ε0 + ε1)^0.75)
end

@inline function Jacobian_F_v(h::Float64, v0::Float64, v1::Float64, ε0::Float64, ε1::Float64, C1::Float64, C2::Float64, η0::Float64, λ0::Float64, χ0::Float64)
    return (2^0.75*(ε0 + ε1)^0.25*(((4 - v0^2 - 2*v0*v1 - v1^2)*(λ0*(-4*η0 + χ0) - (3*(2*η0 + λ0)*χ0*(v0 + v1)^2)/2. + (9*λ0*χ0*(v0 + v1)^4)/16.)*(8*C2*(-4*η0 + 3*λ0 + 4*χ0) - 12*C1*(2*λ0 + χ0)*(v0 + v1) - 4*(4*η0 + λ0)*(ε0 + ε1) + 3*λ0*(v0 + v1)^2*(2*C2 + ε0 + ε1) + 6*(v0 + v1)*(-2*C1*(2*λ0 + χ0) + λ0*(v0 + v1)*(2*C2 + ε0 + ε1))))/16. + (-4 + v0^2 + 2*v0*v1 + v1^2)*(-3*(2*η0 + λ0)*χ0*(v0 + v1) + (9*λ0*χ0*(v0 + v1)^3)/4.)*(C1*(4*η0 - χ0) + ((v0 + v1)*(C2*(-4*η0 + 3*λ0 + 4*χ0) - (3*C1*(2*λ0 + χ0)*(v0 + v1))/2. - ((4*η0 + λ0)*(ε0 + ε1))/2. + (3*λ0*(v0 + v1)^2*(2*C2 + ε0 + ε1))/8.))/2.) + (-v0 - v1)*(λ0*(-4*η0 + χ0) - (3*(2*η0 + λ0)*χ0*(v0 + v1)^2)/2. + (9*λ0*χ0*(v0 + v1)^4)/16.)*(C1*(4*η0 - χ0) + ((v0 + v1)*(C2*(-4*η0 + 3*λ0 + 4*χ0) - (3*C1*(2*λ0 + χ0)*(v0 + v1))/2. - ((4*η0 + λ0)*(ε0 + ε1))/2. + (3*λ0*(v0 + v1)^2*(2*C2 + ε0 + ε1))/8.))/2.)))/(Sqrt(4 - v0^2 - 2*v0*v1 - v1^2)*(λ0*(-4*η0 + χ0) - (3*(2*η0 + λ0)*χ0*(v0 + v1)^2)/2. + (9*λ0*χ0*(v0 + v1)^4)/16.)^2)
end

@inline function Jacobian_G_eps(h::Float64, v0::Float64, v1::Float64, ε0::Float64, ε1::Float64, C1::Float64, C2::Float64, η0::Float64, λ0::Float64, χ0::Float64)
    return ((4 - v0^2 - 2*v0*v1 - v1^2)^1.5*(4*λ0*(4 - 3*(v0 + v1)^2)*(ε0 + ε1) - 3*(12*C1*(2*λ0 + χ0)*(v0 + v1) + 9*C1*χ0*(v0 + v1)^3 + 4*λ0*(-6*C2 + ε0 + ε1) - 3*(v0 + v1)^2*(2*C2*(λ0 + 4*χ0) + λ0*(ε0 + ε1)))))/(128.0*2^0.25*(λ0*(-4*η0 + χ0) - (3*(2*η0 + λ0)*χ0*(v0 + v1)^2)/2. + (9*λ0*χ0*(v0 + v1)^4)/16.)*(ε0 + ε1)^1.75)
end

@inline function Jacobian_G_v(h::Float64, v0::Float64, v1::Float64, ε0::Float64, ε1::Float64, C1::Float64, C2::Float64, η0::Float64, λ0::Float64, χ0::Float64)
    return 1/h + ((1 - (v0 + v1)^2/4.)^1.5*(12*C1*(2*λ0 + χ0) + 27*C1*χ0*(v0 + v1)^2 - 6*(v0 + v1)*(2*C2*(λ0 + 4*χ0) + λ0*(ε0 + ε1))))/(4.0*2^0.25*(λ0*(-4*η0 + χ0) - (3*(2*η0 + λ0)*χ0*(v0 + v1)^2)/2. + (9*λ0*χ0*(v0 + v1)^4)/16.)*(ε0 + ε1)^0.75) - ((1 - (v0 + v1)^2/4.)^1.5*(-3*(2*η0 + λ0)*χ0*(v0 + v1) + (9*λ0*χ0*(v0 + v1)^3)/4.)*(12*C1*(2*λ0 + χ0)*(v0 + v1) + 9*C1*χ0*(v0 + v1)^3 + 4*λ0*(-6*C2 + ε0 + ε1) - 3*(v0 + v1)^2*(2*C2*(λ0 + 4*χ0) + λ0*(ε0 + ε1))))/(4.0*2^0.25*(λ0*(-4*η0 + χ0) - (3*(2*η0 + λ0)*χ0*(v0 + v1)^2)/2. + (9*λ0*χ0*(v0 + v1)^4)/16.)^2*(ε0 + ε1)^0.75) + (3*(-v0 - v1)*Sqrt(1 - (v0 + v1)^2/4.)*(12*C1*(2*λ0 + χ0)*(v0 + v1) + 9*C1*χ0*(v0 + v1)^3 + 4*λ0*(-6*C2 + ε0 + ε1) - 3*(v0 + v1)^2*(2*C2*(λ0 + 4*χ0) + λ0*(ε0 + ε1))))/(16.0*2^0.25*(λ0*(-4*η0 + χ0) - (3*(2*η0 + λ0)*χ0*(v0 + v1)^2)/2. + (9*λ0*χ0*(v0 + v1)^4)/16.)*(ε0 + ε1)^0.75)
end


end

# ε, v, εprime, vprime, η0, λ0, χ0, c1, c2, c3, c4, C1, C2 = 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0

# BDNKODEs.tt_zeroth_order(ε, v)
# BDNKODEs.tx_zeroth_order(ε, v)
# BDNKODEs.xx_zeroth_order(ε, v)

# BDNKODEs.tt_first_order(ε, v, εprime, vprime, η0, λ0, χ0)
# BDNKODEs.tx_first_order(ε, v, εprime, vprime, η0, λ0, χ0)
# BDNKODEs.xx_first_order(ε, v, εprime, vprime, η0, λ0, χ0)

# BDNKODEs.c1(η0, λ0, χ0)
# BDNKODEs.c2(η0, λ0, χ0)
# BDNKODEs.c3(η0, λ0, χ0)
# BDNKODEs.c4(η0, λ0, χ0)

# BDNKODEs.εprime(ε, v, c1, c2, c3, c4, C1, C2, η0, λ0, χ0)
# BDNKODEs.vprime(ε, v, c1, c2, c3, c4, C1, C2, η0, λ0, χ0) 