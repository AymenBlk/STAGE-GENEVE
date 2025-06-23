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

end