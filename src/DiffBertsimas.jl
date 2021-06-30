py"""
class DiffBertsimas(torch.nn.Module):
    def __init__(
        self,
        D,
        w,
        c,
        μ,
        sqrtΣ,
        solver_args={'use_indirect': True, 'eps': 1e-5, 'max_iters': 20000}
    ):
        super(DiffBertsimas, self).__init__()
        self.D = D
        self.μ = μ
        Σ = sqrtΣ.T @ sqrtΣ
        self.Σ = Σ
        self.solver_args=solver_args

        λ = cp.Variable()
        π_neg = cp.Variable(D)
        π_pos = cp.Variable(D)
        θ = cp.Variable(D)

        params = cp.Parameter(D+1)

        constraints = [
            λ >= 0,
            π_neg >= 0,
            π_pos >= 0,
            θ >= 0,
            w == π_pos - π_neg,
            cp.multiply(params[0:D], (π_pos + π_neg)) - θ <= λ,
        ]

        objective = cp.Maximize(μ.T @ w - cp.sum(θ) - λ * params[D])
        problem = cp.Problem(objective, constraints + c)

        self.cvxpylayer = CvxpyLayer(
            problem,
            parameters=[params],
            variables=[w, λ, π_neg, π_pos, θ]
        )

    def forward(self, params):
        return self.cvxpylayer(params, solver_args=self.solver_args)[0]

    def get_variables(self, params):
        params_ = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        return self.cvxpylayer(params_, solver_args=self.solver_args)

    def get_w(self, params):
        params_ = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        return self.get_variables(params_)[0]

    def get_lagrangian(self, params):
        params_ = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        D = self.D
        μ_ = torch.tensor(self.μ, dtype=torch.float64)
        Σ_ = torch.tensor(self.Σ, dtype=torch.float64)
        w_, λ_, π_neg_, π_pos_, θ_ = self.cvxpylayer(params_, solver_args=self.solver_args)
        return μ_.T @ w_ - torch.sum(θ_) - λ_ * params_[D]
"""

"""
    DiffBertsimas

Differentiable Bertsimas model.

Atributes:
- `D::Int`: dimension (i.e. number of assets)
- `w::PyObject`: differentiable weights
- `c::Vector{PyObject}`: convex constraints on weights
- `μ::Array{Float64,1}`: predicted mean vector
- `sqrtΣ::Array{Float64,2}`: predicted upper triangular Cholesky factor of covariance matrix
- `solver_args::Dict=Dict("use_indirect" => true, "eps" => 1e-5, "max_iters" => 20000}`: solver arguments
"""
DiffBertsimas = py"DiffBertsimas"
