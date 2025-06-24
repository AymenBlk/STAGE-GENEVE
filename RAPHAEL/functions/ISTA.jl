module ISTA

using LinearAlgebra, Printf, Statistics, Random

function ista_L(x0, f, g, ∇f, L, prox;
        max_iter = 100_000, tol = 1e-9,print_freq = 1000, verbose = true
    )
    x = copy(x0)
    x_next = similar(x)
    
    step = 1/L
    cost_prev = f(x) + g(x)
    for k in 1:max_iter
        grad = ∇f(x)
        x_next .= prox(x .- step .* grad, step)
        
        cost_current = f(x_next) + g(x_next)
        if abs(cost_current - cost_prev) <tol
            x .= x_next
            if verbose
                @printf("[ISTA] END iter %5d  cost=%.3e  diff=%.3e\n", k, cost_current, abs(cost_current - cost_prev))
            end
            break
        end
        if verbose && (k == 1 || k % print_freq == 0)
            @printf("[ISTA] iter %5d  cost=%.3e  diff=%.3e\n", k, cost_current, abs(cost_current - cost_prev))
        end
    
        cost_prev = cost_current
        x .= x_next
    end
    
    return x
end

"""
    ista(x0, f, g, ∇f, L0, prox;
        eta = 2, max_iter = 100_000, tol = 1e-9, print_freq = 1000, verbose = true)
ISTA algorithm with backtracking line search for the step size.
This function solves the optimization problem:
    min_x f(x) + g(x)
where `f` is a differentiable function with gradient `∇f`, and `g` is a possibly non-smooth function.
The algorithm uses a backtracking line search to adaptively find the step size `L`.
The parameters are:
- `x0`: Initial point.
- `f`: Function to be minimized.
- `g`: Regularization function (possibly non-smooth).   
- `∇f`: Gradient of the function `f`.
- `L0`: Initial step size (choose L very small).
- `prox`: Proximal operator for the regularization function `g`.
- `eta`: Factor by which the step size is increased in each iteration (default is
    2).
- `max_iter`: Maximum number of iterations (default is 100,000).
- `tol`: Tolerance for convergence (default is 1e-9).
- `print_freq`: Frequency of printing progress (default is 1000).
- `verbose`: If true, prints progress information (default is true).
"""
function ista(x0, f, g, ∇f, L0, prox;
        eta = 2, max_iter = 100_000, tol = 1e-9, print_freq = 1000, verbose = true
    )
    xnew = x = copy(x0) # xnew is defined later; we only initialize
    F_new = F_prev = f(x) + g(x)
    L = L0
    for k in 1:max_iter
        grad = ∇f(x)
        Ltrial = L
        while true  # search the smallest i_k (power of eta)
            xnew = prox.(x.-grad.*(1/Ltrial), 1/Ltrial) 
            diff = xnew.-x
            F_new = f(xnew) + g(xnew)
            Q_val = f(x) +dot(diff, grad) +(Ltrial/2)* dot(diff, diff) +g(xnew)

            if F_new <= Q_val + eps()
                break  # Ltrial is good
            end 
            Ltrial *= eta # increase at each iteration
        end
        L = Ltrial

        if verbose && (k == 1 || k % print_freq == 0)
            @printf("[ISTA-BT] iter %6d  cost = %.3e  ΔF = %.3e  L = %.3e\n",
                    k, F_new, abs(F_new - F_prev), L)
        end

        if abs(F_new - F_prev) < tol
            verbose && @printf("[ISTA-BT] END iter %6d  cost = %.3e  ΔF = %.3e\n",
                               k, F_new, abs(F_new - F_prev))
            return xnew
        end
        x, F_prev = xnew, F_new
    end
    return x
end

end


