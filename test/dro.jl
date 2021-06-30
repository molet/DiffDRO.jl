@testset "DRO models" begin
    D = 10

    financial_data = get_financial_data(D=D)

    μ = mean(Matrix(financial_data[!, 1:D]), dims=1)[:]
    Σ = cov(Matrix(financial_data[!, 1:D]), dims=1)
    sqrtΣ = Matrix(cholesky(Σ).U)

    lower_bound = 0.0
    upper_bound = 25.0
    wealth = 100.0
    max_risk = 500.0

    solver = optimizer_with_attributes(
        SCS.Optimizer,
        "eps" => 1.0e-5,
        "max_iters" => 20000,
        "verbose" => false
    )

    function BasicWeightConvexSet(D, wealth, lower_bound, upper_bound, max_risk, μ, sqrtΣ)
        model = Model()
    
        @variable(model, w[i=1:D])
    
        @constraint(model, sum(w) <= wealth)
        @constraint(model, lower_bound .<= w .<= upper_bound)
        @constraint(model, [max_risk; sqrtΣ*w] in JuMP.SecondOrderCone())
        return model
    end

    basic_model = BasicWeightConvexSet(                                                        
        D,                                                                               
        wealth,                                                                          
        lower_bound,                                                                     
        upper_bound,                                                                     
        max_risk,                                                                        
        μ,                                                                               
        sqrtΣ                                                                            
    )

    weights, constraints = DiffBasicWeightConvexSet(
        D,
        wealth,
        lower_bound,
        upper_bound,
        max_risk,
        μ,
        sqrtΣ
    )

    @testset "Bertsimas model" begin
        Δ = collect(0.1:0.1:1.0)
        Γ = 0.8

        formulation = RobustBertsimas(
            predicted_mean=μ,
            predicted_covariance=Σ,
            uncertainty_delta=Δ,
            bertsimas_budget=Γ,
        )
        model = copy(basic_model)
        @objective(model, Max, portfolio_return!(model, model[:w], formulation))
        w_po_opt = compute_solution(model, solver) * wealth

        w_diff_dro = DiffBertsimas(D, weights, constraints, μ, sqrtΣ).get_w(vcat(Δ, Γ)).data.numpy()

        @test w_po_opt ≈ w_diff_dro atol=0.001
    end

    @testset "BenTal model" begin
        δ = 10.0

        formulation = RobustBenTal(
            predicted_mean=μ,
            predicted_covariance=Σ,
            uncertainty_delta=δ,
        )
        model = copy(basic_model)
        @objective(model, Max, portfolio_return!(model, model[:w], formulation))
        w_po_opt = compute_solution(model, solver) * wealth

        w_diff_dro = DiffBenTal(D, weights, constraints, μ, sqrtΣ).get_w([δ]).data.numpy()

        @test w_po_opt ≈ w_diff_dro atol=0.001
    end

    @testset "Delage model" begin
        γ1 = 1.5
        γ2 = 1.1

        formulation = RobustDelague(
            predicted_mean=μ,
            predicted_covariance=Σ,
            γ1=γ1,
            γ2=γ2,
            utility_coeficients=[1.0],
            utility_intercepts=[0.0]
        )
        model = copy(basic_model)
        model = po_max_utility_return(formulation, current_wealth=wealth, model=model)
        w_po_opt = compute_solution(model, solver) * wealth

        w_diff_dro = DiffDelage(D, weights, constraints, μ, sqrtΣ).get_w([γ1, γ2]).data.numpy()

        @test w_po_opt ≈ w_diff_dro atol=0.01
    end
end
