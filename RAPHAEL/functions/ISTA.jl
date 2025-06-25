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
        eta = 2.0, max_iter = 100_000,tol= 1e-9,print_freq = 1000,verbose = true)

    x  = copy(x0)
    x_new  = similar(x)
    F_prev = f(x) + g(x) # coût initial
    F_new = 0
    L      = L0

    for k in 1:max_iter
        grad = ∇f(x)

        L_trial = L
        while true
            step = 1/L_trial
            x_new .= prox(x .- step .* grad, step)

            diff = x_new .- x
            f_new = f(x_new)
            g_new = g(x_new)
            F_new = f_new + g_new

            Q_val = f(x) + dot(diff, grad) + (L_trial/2) * dot(diff, diff) + g_new  # majorant

            F_new ≤ Q_val + eps() && break # critère BT
            L_trial *= eta
            if L_trial > 1e12                 
                error("Backtracking diverged : L trop grand")
            end
        end
        L = L_trial                            # prochain tour part de L_trial

        if verbose && (k == 1 || k % print_freq == 0)
            @printf("[ISTA-BT] iter %6d  cost = %.3e  ΔF = %.3e  L = %.3e\n",
                    k, F_new, abs(F_prev - F_new), L)
        end
        if abs(F_prev - F_new) < tol 
            if verbose
                @printf("[ISTA-BT] END iter %5d  cost=%.3e  diff=%.3e\n", k, F_new, abs(F_prev - F_new))
            end
            return x_new
        end

        x .= x_new
        F_prev = F_new
    end
    return x
end

end


