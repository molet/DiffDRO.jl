__precompile__(false)
module DiffDRO

using Conda
using PyCall

export DiffBasicWeightConvexSet,
    DiffBertsimas,
    DiffBenTal,
    DiffDelage,
    utility,
    expected_return,
    std_adjusted_expected_return,
    var_adjusted_expected_return

Conda.add("pip")

run(PyCall.python_cmd(`-m pip install numpy`))
run(PyCall.python_cmd(`-m pip install scs==2.1.3`))
run(PyCall.python_cmd(`-m pip install cvxpy`))
run(PyCall.python_cmd(`-m pip install torch`))
run(PyCall.python_cmd(`-m pip install cvxpylayers`))

py"""
import numpy as np
import cvxpy as cp
import torch 
from cvxpylayers.torch import CvxpyLayer
"""

include("DiffBasicWeightConvexSet.jl")
include("DiffBertsimas.jl")
include("DiffBenTal.jl")
include("DiffDelage.jl")
include("DiffObj.jl")

end
