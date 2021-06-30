@testset "Utility & Objective" begin
    D = 10
    N = 15
    lookback = 30

    financial_data = get_financial_data(D=D)

    lower_bound = 0.0
    upper_bound = 25.0
    wealth = 100.0
    max_risk = 500.0

    Δ_init = ones(D) 
    Γ_init = 1.0

    μs = []
    sqrtΣs = []
    samples = []

    for n = 1:N
        d = Matrix(financial_data[n:N+n-1, 1:D])
        push!(μs, mean(Matrix(d), dims=1)[:])
        push!(sqrtΣs, Matrix(cholesky(cov(Matrix(d), dims=1)).U))
        push!(samples, Vector(financial_data[N+n, 1:D]))
    end

    POs = []
    for n = 1:N
        weights, constraints = DiffBasicWeightConvexSet(
            D,
            wealth,
            lower_bound,
            upper_bound,
            max_risk,
            μs[n],
            sqrtΣs[n]
        )
        push!(POs, DiffBertsimas(D, weights, constraints, μs[n], sqrtΣs[n]))
    end

    utility_function = utility(POs, samples, expected_return())

    function f(params::Vector{Float64})
        return utility_function.value(params)[1]
    end

    function g!(storage::Vector, params::Vector{Float64})
        storage[:] .= utility_function.grad(params)
    end

    result = Optim.maximize(
        f,
        g!,
        vcat(Δ_init, Γ_init),
        LBFGS(alphaguess = LineSearches.InitialStatic(scaled=true), linesearch = LineSearches.BackTracking()),
        Optim.Options(
            x_tol = 1.0e-12,
            f_tol = 1.0e-12,
            g_tol = 1.0e-2,
            f_calls_limit = 1000,
            g_calls_limit = 1000,
            iterations = 1000,
            show_trace = true,
        ),
    )

    @test norm(utility_function.grad(vcat(Δ_init, Γ_init))) > norm(utility_function.grad(result.res.minimizer)) 
end
