using DataFrames
using DiffDRO
using DiffDRO.TestUtils: get_financial_data
using JuMP
using LinearAlgebra
using LineSearches
using Optim
using Random
using Pkg
using Statistics
using SCS
using Test

Pkg.add(url="https://github.com/andrewrosemberg/PortfolioOpt.jl.git")
using PortfolioOpt

@testset "DiffDRO" begin
    include("dro.jl")
    include("obj.jl")
end
