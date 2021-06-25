py"""
class DiffBenTal(torch.nn.Module):
    def __init__(
        self,
        D,
        w,
        c,
        μ,
        sqrtΣ,
        solver_args={'use_indirect': True, 'eps': 1e-5, 'max_iters': 20000}
    ):
        super(DiffBenTal, self).__init__()
        self.D = D
        self.μ = μ
        Σ = sqrtΣ.T @ sqrtΣ
        self.Σ = Σ
        self.solver_args=solver_args

        θ = cp.Variable()

        params = cp.Parameter(1)

        constraints = [
            cp.SOC(θ, sqrtΣ @ w),
        ]

        objective = cp.Maximize(μ.T @ w - θ * params[0])

        problem = cp.Problem(objective, constraints + c)

        self.cvxpylayer = CvxpyLayer(
            problem,
            parameters=[params],
            variables=[w, θ]
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
        w_, θ_ = self.cvxpylayer(params_, solver_args=self.solver_args)
        return μ_.T @ w_ - θ_ * params_[0]
"""

DiffBenTal = py"DiffBenTal"
