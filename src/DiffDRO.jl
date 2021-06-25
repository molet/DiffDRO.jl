__precompile__(false)
module DiffDRO

using Conda
using PyCall

export DiffBasicWeightConvexSet,
    DiffBertsimas,
    DiffBenTal,
    DiffDelage

Conda.add("pip")

run(PyCall.python_cmd(`-m pip install numpy`))
run(PyCall.python_cmd(`-m pip install scs==2.1.3`))
run(PyCall.python_cmd(`-m pip install cvxpy`))
run(PyCall.python_cmd(`-m pip install torch`))
run(PyCall.python_cmd(`-m pip install cvxpylayers`))

#const numpy = PyNULL()
#const cvxpy = PyNULL()
#const torch = PyNULL()
#const cvxpylayers = PyNULL()

#function __init__()
#    copy!(numpy, pyimport("numpy"))
#    copy!(cvxpy, pyimport("cvxpy"))
#    copy!(torch, pyimport("torch"))
#    copy!(cvxpylayers, pyimport("cvxpylayers"))
#end

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

end
