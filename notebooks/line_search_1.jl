using CairoMakie


##

function f(x::Vector)
    return x[1]^2
end
    
##


function backtracking_ls(f::Function, x::Vector, p_k::Vector)
    α_k = 1/2
    f_k = f(x)
    τ = 0.5
    k = 1
    while f(x+α_k*p_k) >= f_k && k < 100
        α_k = τ*(1/2)^(k+1)
        k += 1
    end
    return α_k
end
p_k = [-1.0]
x = [2.0]

α_list = Float64[]
f_list = Float64[]
x_list = Float64[]

function line_search(x)
    for i in 1:10
        α_k = backtracking_ls(f, x, p_k)
        x = x.+α_k*p_k
        push!(x_list, x[1])
        push!(f_list, f(x))
    end
end
line_search(x)
fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, range(-2,2,length=100), x->x^2)
scatter!(ax, x_list, f_list, color="blue")
display(fig)

##

function backtracking_ls_armijo(f::Function, x::Vector, p_k::Vector)
    α_k = 1/2
    f_k = f(x)
    τ = 0.5
    k = 1
    β = 1
    while f(x+α_k*p_k) >= f_k + α_k*β && k < 100
        α_k = τ*(1/2)^(k+1)
        k += 1
    end
    return α_k
end
p_k = [-1.0]
x = [2.0]

α_list = Float64[]
f_list = Float64[]
x_list = Float64[]

function line_search(x)
    for i in 1:10
        α_k = backtracking_ls_armijo(f, x, p_k)
        x = x.+α_k*p_k
        push!(x_list, x[1])
        push!(f_list, f(x))
    end
end
line_search(x)
ax = Axis(fig[2, 1])
lines!(ax, range(-2,2,length=100), x->x^2)
scatter!(ax, x_list, f_list, color=:red)
fig