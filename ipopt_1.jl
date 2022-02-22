using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
##------------------------  ------------------------##  
using Ipopt
using LinearAlgebra, ForwardDiff, Zygote
##------------------------  ------------------------##
# Implement hs071 from IPOPT/IPOPT.jl documentation using 
# analytical and automatic differentiation.

# min x1 * x4 * (x1 + x2 + x3) + x3
# st  x1 * x2 * x3 * x4 >= 25
#     x1^2 + x2^2 + x3^2 + x4^2 = 40
#     1 <= x1, x2, x3, x4 <= 5
# Start at (1,5,5,1)
# End at (1.000..., 4.743..., 3.821..., 1.379...)

# solution/starting point = x
# objective function = f(x)
# constraints = g(x)

# number of variables
n = 4

# bounds
# 1 <= x1, x2, x3, x4 <= 5
# lower bounds
x_L = [1.0, 1.0, 1.0, 1.0]
# upper bounds
x_U = [5.0, 5.0, 5.0, 5.0]

# number of constraints, dim of g(x)
m = 2

# constraints lower bounds
# x1 * x2 * x3 * x4 >= 25
# x1^2 + x2^2 + x3^2 + x4^2 = 40
g_L = [25.0, 40.0]
# constraints upper bounds
g_U = [2.0e19, 40.0]# unbounded first term

# initial point
x_0 = [1.0, 5.0, 5.0, 1.0]

##---------------------- obj/constrain functions ----------------------##
# objective function
function eval_f(x)
    return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
end

# constraint function
function eval_g(x, g)
    g[1] = x[1] * x[2] * x[3] * x[4]
    g[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    return g
end

function eval_g_ad(x, g)
    g = eltype(x).(g)
    g[1] = x[1] * x[2] * x[3] * x[4]
    g[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    return g
end


##------------------------ grad obj function ------------------------##

# grad of obj function ∇f(x)
grad_f = zeros(n)
# analytical grad of obj function
function eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64})
    grad_f[1] = x[1] * x[4] + x[4] * (x[1] + x[2] + x[3])
    grad_f[2] = x[1] * x[4]
    grad_f[3] = x[1] * x[4] + 1
    return grad_f[4] = x[1] * (x[1] + x[2] + x[3])
end

# now with automatic differentiation
∇f = x -> ForwardDiff.gradient(eval_f, x)

# check both are the same
eval_grad_f(x_0, grad_f)
@show grad_f
@show ∇f(x_0)

function eval_grad_f_ad(x::Vector{Float64}, grad_f::Vector{Float64})
    grad_f[:] = ForwardDiff.gradient(eval_f, x)[:]
end
@show eval_grad_f_ad(x_0, grad_f)




##--------------------- jac of constraints ∇g(x)^T --------------------##
println("jacobian of constraints")
g = zeros(m)
∇g = x -> ForwardDiff.jacobian(x->eval_g_ad(x, g), x)
jac_temp = ∇g(x_0)
sparsity = jac_temp .!= 0.0
I = ((i, j) for i = 1:m, j = 1:n if sparsity[i,j] != 0.0)
rows = Int32.(getindex.(I, 1)[:])
cols = Int32.(getindex.(I, 2)[:])
values = zeros(length(rows))

# analytical version
function eval_jac_g(
    x::Vector{Float64},
    rows::Vector{Int32},
    cols::Vector{Int32},
    values::Union{Nothing,Vector{Float64}},
)
    if values === nothing
    # return the structure of the jacobian, 
    # Sparse Triplet Matrix Format 
    # i.e. the rows and cols of the non-zero entries
    # index style=FORTRAN STYLE (1-indexed)
        # Constraint (row) 1
        rows[1] = 1
        cols[1] = 1
        rows[2] = 1
        cols[2] = 2
        rows[3] = 1
        cols[3] = 3
        rows[4] = 1
        cols[4] = 4
        # Constraint (row) 2
        rows[5] = 2
        cols[5] = 1
        rows[6] = 2
        cols[6] = 2
        rows[7] = 2
        cols[7] = 3
        rows[8] = 2
        cols[8] = 4
    else
    # return the values of the jacobian of the constraints
        # Constraint (row) 1
        values[1] = x[2] * x[3] * x[4]  # 1,1
        values[2] = x[1] * x[3] * x[4]  # 1,2
        values[3] = x[1] * x[2] * x[4]  # 1,3
        values[4] = x[1] * x[2] * x[3]  # 1,4
        # Constraint (row) 2
        values[5] = 2 * x[1]  # 2,1
        values[6] = 2 * x[2]  # 2,2
        values[7] = 2 * x[3]  # 2,3
        values[8] = 2 * x[4]  # 2,4
    end
    @show values
    return nothing
end
println("analytical")
eval_jac_g(x_0, rows, cols, nothing)
@show rows; @show cols
eval_jac_g(x_0, rows, cols, values)
@show values
display(reshape(values, m, n))

println("\nAD")

function jac_structure!(
    m, n,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
  )
    I = ((i, j) for i = 1:m, j = 1:n)
    rows .= getindex.(I, 1)[:]
    cols .= getindex.(I, 2)[:]
    return rows, cols
end

function jac_structure_sparse!(
    x, m, n,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
  )
    jac_temp =  ForwardDiff.jacobian(x->eval_g_ad(x, g), x)
    sparsity = jac_temp .!= 0.0
    I = ((i, j) for i = 1:m, j = 1:n if sparsity[i,j] != 0.0)
    rows .= getindex.(I, 1)[:]
    cols .= getindex.(I, 2)[:]
    return rows, cols
end

∇g = x -> ForwardDiff.jacobian(x->eval_g_ad(x, g), x)

function eval_jac_g_ad(x, rows, cols, values)

    if values === nothing
        jac_structure_sparse!(x_0, m, n, rows, cols)
        # jac_structure!(m, n, rows, cols)
    else

        values[:] = ForwardDiff.jacobian(x->eval_g_ad(x, g), x)'[:]
    end
    return nothing
end

#jac_structure!(m, n, rows, cols)
@show jac_structure_sparse!(x_0, m, n, rows, cols)
@show rows; @show cols

eval_jac_g_ad(x_0, rows, cols, nothing)
eval_jac_g_ad(x_0, rows, cols, values)



##------------------------ Hessian of Langrangian  ------------------------##
# L = f(x) + g(x)^T * λ
# ∇L = σ_f  ∇^2 f(x_k) + Σ^m_i λ_i * ∇^2 g_i(x_k)
println("Hessian of Langrangian")
values = ones(10)

function eval_h(
    x::Vector{Float64},
    rows::Vector{Int32},
    cols::Vector{Int32},
    obj_factor::Float64,
    lambda::Vector{Float64},
    values::Union{Nothing,Vector{Float64}},
)
    if values === nothing
        # Symmetric matrix, fill the lower left triangle only
        idx = 1
        for row in 1:4
            for col in 1:row
                rows[idx] = row
                cols[idx] = col
                idx += 1
            end
        end
    else
        # Again, only lower left triangle
        # Objective
        values[1] = obj_factor * (2 * x[4])  # 1,1
        values[2] = obj_factor * (x[4])  # 2,1
        values[3] = 0                      # 2,2
        values[4] = obj_factor * (x[4])  # 3,1
        values[5] = 0                      # 3,2
        values[6] = 0                      # 3,3
        values[7] = obj_factor * (2 * x[1] + x[2] + x[3])  # 4,1
        values[8] = obj_factor * (x[1])  # 4,2
        values[9] = obj_factor * (x[1])  # 4,3
        values[10] = 0                     # 4,4

        # First constraint
        values[2] += lambda[1] * (x[3] * x[4])  # 2,1
        values[4] += lambda[1] * (x[2] * x[4])  # 3,1
        values[5] += lambda[1] * (x[1] * x[4])  # 3,2
        values[7] += lambda[1] * (x[2] * x[3])  # 4,1
        values[8] += lambda[1] * (x[1] * x[3])  # 4,2
        values[9] += lambda[1] * (x[1] * x[2])  # 4,3

        # Second constraint
        values[1] += lambda[2] * 2  # 1,1
        values[3] += lambda[2] * 2  # 2,2
        values[6] += lambda[2] * 2  # 3,3
        values[10] += lambda[2] * 2  # 4,4
    end
    return
end

println("analytical")
eval_h(x_0, Int32.(ones(10)), Int32.(ones(10)), 1., ones(2), nothing)
# Hessian of constraints
eval_h(x_0, Int32.(ones(10)), Int32.(ones(10)), 0., ones(2), values)
@show values


function L(x, λ, σ)
    λ = typeof(x)(λ)
    return σ * eval_f(x) + eval_g_ad(x,g)'*λ 
end


λ = ones(2)
σ = 0.0
L(x_0, λ, σ)
∇H = x -> ForwardDiff.hessian(x->L(x, λ, σ), x)
sparsity = LowerTriangular(∇H(x_0)) .!= 0.0
I = ((i, j) for i = 1:size(sparsity,1), j = 1:size(sparsity,2) if sparsity[i,j] != 0.0)
rows_hess = Int32.(getindex.(I, 1)[:])
cols_hess = Int32.(getindex.(I, 2)[:])
values_hess = zeros(length(rows_hess))

function hess_structure_sparse!(x, rows, cols)
    ∇H = x -> ForwardDiff.hessian(x->L(x, λ, 1), x)
    sparsity = LowerTriangular(∇H(x_0)) .!= 0.0
    I = ((i, j) for i = 1:size(sparsity,1), j = 1:size(sparsity,2) if sparsity[i,j] != 0.0)
    rows .= Int32.(getindex.(I, 1)[:])
    cols .= Int32.(getindex.(I, 2)[:])
    return rows, cols
end


function eval_h_ad(
    x::Vector{Float64},
    rows,
    cols,
    σ::Float64,
    λ::Vector{Float64},
    values::Union{Nothing,Vector{Float64}},
)
    if values === nothing
        hess_structure_sparse!(x, rows, cols)
    else
        hess = ForwardDiff.hessian(x->L(x, λ, σ), x)
        values[:] = hess[tril(trues(size(hess)))]
    end        
end
println("automatic differentiation")
eval_h_ad(x_0, rows_hess, cols_hess, σ, λ, values_hess)
@show rows_hess; @show cols_hess
eval_h_ad(x_0, rows_hess, cols_hess, σ, λ, values_hess)
@show values_hess

##------------------------ Solve problem analytical ------------------------##

n = 4
x_L = [1.0, 1.0, 1.0, 1.0]
x_U = [5.0, 5.0, 5.0, 5.0]

m = 2
g_L = [25.0, 40.0]
g_U = [2.0e19, 40.0]

prob = Ipopt.CreateIpoptProblem(
    n,
    x_L,
    x_U,
    m,
    g_L,
    g_U,
    8,
    10,
    eval_f,
    eval_g,
    eval_grad_f,
    eval_jac_g,
    eval_h,
)

prob.x = [1.0, 5.0, 5.0, 1.0]
solvestat = Ipopt.IpoptSolve(prob)
@show prob.x
##------------------------ solve with ad ------------------------##



n = 4
x_L = [1.0, 1.0, 1.0, 1.0]
x_U = [5.0, 5.0, 5.0, 5.0]

m = 2
g_L = [25.0, 40.0]
g_U = [2.0e19, 40.0]

g = zeros(m)
∇g = x -> ForwardDiff.jacobian(x->eval_g_ad(x, g), x)
jac_temp = ∇g(x_0)
sparsity = jac_temp .!= 0.0
I = ((i, j) for i = 1:m, j = 1:n if sparsity[i,j] != 0.0)
rows = Int32.(getindex.(I, 1)[:])
cols = Int32.(getindex.(I, 2)[:])
values = zeros(length(rows))
rows = nothing
cols = nothing
values = nothing


prob = Ipopt.CreateIpoptProblem(
    n,
    x_L,
    x_U,
    m,
    g_L,
    g_U,
    8,
    10,
    eval_f,
    eval_g,
    eval_grad_f_ad,
    eval_jac_g_ad,
    eval_h_ad,
    # function h() return false end
)
# Ipopt.AddIpoptStrOption(prob, "hessian_approximation", "limited-memory")
# Ipopt.AddIpoptStrOption(prob, "check_derivatives_for_naninf", "yes")
# Ipopt.AddIpoptIntOption(prob, "file_print_level", 10)
# Ipopt.AddIpoptIntOption(prob, "output_file", "ipopt.out")
# Ipopt.OpenIpoptOutputFile(prob, "blah.txt", 5)

prob.x = [1.0, 5.0, 5.0, 1.0]
solvestat = Ipopt.IpoptSolve(prob)
@show prob.x