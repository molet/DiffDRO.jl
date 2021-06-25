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

DiffBasicWeightConvexSet = py"DiffBasicWeightConvexSet"
