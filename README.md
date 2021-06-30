# DiffDRO
Implementation of differentiable Robust and Distributionally Robust methods using [cvxpy](https://www.cvxpy.org/index.html) and [cvxpylayers](https://github.com/cvxgrp/cvxpylayers).

## Installation

This is an unregistered package that can be installed as:

```julia
julia> ] add https://github.com/molet/DiffDRO.jl.git
```

## Example

```julia
using CSV
using DataFrames
using HTTP

# obtain financial data
financial_url = "https://raw.githubusercontent.com/scikit-learn/examples-data/master/financial-data/"
financial_dict = Dict(                                                                       
	"TOT" => "Total",                                                                        
	"XOM" => "Exxon",                                                                        
	"CVX" => "Chevron",                                                                      
	"COP" => "ConocoPhillips",                                                               
	"VLO" => "Valero Energy",
)
financial_data = DataFrame()                                                                 
for key in keys(financial_dict)                                                              
    financial_data[!, key] = DataFrame(CSV.File(HTTP.get(financial_url*key*".csv").body))[!, :close]
end

using DiffDRO
using LinearAlgebra
using LineSearches
using Optim
using Statistics

D = 5				# number of assets
N = 15				# number of training days
lookback = 30		# number of lookback days

lower_bound = 0.0	# lower bounds of weights
upper_bound = 25.0	# upper bounds of weights
wealth = 100.0		# total wealth to invest
max_risk = 500.0	# maximum risk (variance)

# initital parameters of Bertsimas model
Δ_init = ones(D)
Γ_init = 1.0

# distribution parameters for each training day
μs = []				# mean vectors
sqrtΣs = []			# cholesky decomposition of covariance matrices
samples = []		# true values of prices at each training day

# compute multivariate normal distributions based on sample mean and cov
for n = 1:N
    d = Matrix(financial_data[n:N+n-1, 1:D])
    push!(μs, mean(Matrix(d), dims=1)[:])
    push!(sqrtΣs, Matrix(cholesky(cov(Matrix(d), dims=1)).U))
    push!(samples, Vector(financial_data[N+n, 1:D]))
end

# construct differenctiable Bertsimas models for each training day
# using simple convex set for the weights
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

# construct ultimate utility function to optimize
utility_function = utility(POs, samples, expected_return())

# optimize utility function wrt. PO parameters
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
	Optim.Options(g_tol = 1.0e-2)
)

# check convergence
println(norm(utility_function.grad(vcat(Δ_init, Γ_init))))  # ≈ 48.88
println(norm(utility_function.grad(result.res.minimizer)))  # ≈ 0.003
```
