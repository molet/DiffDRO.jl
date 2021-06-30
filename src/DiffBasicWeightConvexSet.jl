py"""
def DiffBasicWeightConvexSet(D, wealth, lower_bound, upper_bound, max_risk, μ, sqrtΣ):
    w = cp.Variable(D)
    c = [
        cp.sum(w) <= wealth,
        w >= lower_bound,
        w <= upper_bound,
        cp.SOC(max_risk, sqrtΣ @ w),
    ]
    return w, c
"""

"""
    DiffBasicWeightConvexSet

Basic convex set for weights.

Atributes:
- `D::Int`: dimension (i.e. number of assets)
- `wealths::Float64`: total wealth to invest
- `lower_bound::Float64`: lower bound of weights
- `upper_bound::Float64`: upper bound of weights
- `max_risk::Float64`: maximum risk
- `μ::Array{Float64,1}`: predicted mean vector
- `sqrtΣ::Array{Float64,2}`: predicted upper triangular Cholesky factor of covariance matrix
"""
DiffBasicWeightConvexSet = py"DiffBasicWeightConvexSet"
