py"""
class utility():
    def __init__(self, po_array, samples_array, objective):
        if len(po_array) is not len(samples_array):
            raise ValueError('Lenght of  po_array and samples_array must be the same.')
        self.po_array = po_array
        self.samples_array = torch.tensor(samples_array, dtype=torch.float64)
        self.objective = objective

    def value(self, params):
        params_ = torch.tensor(params, dtype=torch.float64, requires_grad=True)

        v = torch.zeros(1)
        for po, samples in zip(self.po_array, self.samples_array):
            w = po(params_)
            v += self.objective(w, samples)
        return np.array(v.data, dtype=np.float64)

    def grad(self, params):
        params_ = torch.tensor(params, dtype=torch.float64, requires_grad=True)

        v = torch.zeros(1)
        for po, samples in zip(self.po_array, self.samples_array):
            w = po(params_)
            v += self.objective(w, samples)
        v.backward()
        return np.array(params_.grad, dtype=np.float64)
"""

utility = py"utility"

py"""
def expected_return(w, samples):
    r = samples @ w
    return torch.mean(r)
"""

expected_return = py"expected_return"

py"""
def std_adjusted_expected_return(w, samples, risk_aversion):
    r = samples @ w
    return torch.mean(r) - risk_aversion * torch.std(r)
"""

std_adjusted_expected_return = py"std_adjusted_expected_return"

py"""
def var_adjusted_expected_return(w, samples, risk_aversion):
    r = samples @ w
    return torch.mean(r) - risk_aversion * torch.var(r)
"""

var_adjusted_expected_return = py"var_adjusted_expected_return"
