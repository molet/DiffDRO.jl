py"""
class DiffDelage(torch.nn.Module):
    def __init__(
        self,
        D,
        w,
        c,
        μ,
        sqrtΣ,
        solver_args={'use_indirect': True, 'eps': 1e-5, 'max_iters': 20000}
    ):
        super(DiffDelage, self).__init__()
        self.D = D
        self.μ = μ
        Σ = sqrtΣ.T @ sqrtΣ
        self.Σ = Σ
        self.solver_args=solver_args

        P = cp.Variable((D+1, D+1), PSD=True)
        Q = cp.Variable((D+1, D+1), PSD=True)

        params = cp.Parameter(2)

        constraints = [
            P[D, 0:D] == -Q[D, 0:D] + w / 2 - Q[0:D, 0:D] @ μ,
            Q[0:D, 0:D] >> 0,
        ]
        objective = cp.Minimize(params[1] * cp.sum(cp.multiply(Σ, Q[0:D, 0:D])) - cp.quad_form(μ, Q[0:D, 0:D]) + Q[D, D] + cp.sum(cp.multiply(Σ, P[0:D, 0:D])) - 2 * μ.T @ P[D, 0:D] + params[0] * P[D, D])
        problem = cp.Problem(objective, constraints + c)

        self.cvxpylayer = CvxpyLayer(problem, parameters=[params], variables=[w, P, Q])

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
        w_, P_, Q_ = self.cvxpylayer(params_, solver_args=self.solver_args)
        return params_[1] * torch.sum(Σ_ * Q_[0:D, 0:D]) - torch.dot(μ_, Q_[0:D, 0:D] @ μ_) + Q_[D, D] + torch.sum(Σ_ * P_[0:D, 0:D]) - 2 * μ_.T @ P_[D, 0:D] + params_[0] * P_[D, D]
"""

"""
    DiffDelage

Differentiable Delage model.

Atributes:
- `D::Int`: dimension (i.e. number of assets)
- `w::PyObject`: differentiable weights
- `c::Vector{PyObject}`: convex constraints on weights
- `μ::Array{Float64,1}`: predicted mean vector
- `sqrtΣ::Array{Float64,2}`: predicted upper triangular Cholesky factor of covariance matrix
- `solver_args::Dict=Dict("use_indirect" => true, "eps" => 1e-5, "max_iters" => 20000}`: solver arguments
"""
DiffDelage = py"DiffDelage"
