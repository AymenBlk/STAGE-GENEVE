module coordinateDescent

using LinearAlgebra, Printf, Statistics, Random


function cd_L(x0, f, g, ∇f, L, prox;
        max_iter = 100_000, tol = 1e-9,print_freq = 1000, verbose = true
    )
    x = copy(x0)
    n = length(x)
    Li = (L isa AbstractVector) ? L : fill(L, n) # is an array. Each entry for each coordinate
    cost_prev = f(x) +g(x)
    epoch = 0

    for k in 1:max_iter
        i = (mod(k-1, n) + 1) # cyclic

        grad_i = ∇f(x)[i] 
        step = 1 / Li[i]
        z_i = prox(x[i] - step * grad_i, step)
        x[i] = z_i 
        if i == n # finished a full sweep (one epoch)
            epoch += 1

            cost_current = f(x) + g(x)
            diff         = abs(cost_current - cost_prev)

            if verbose && (epoch == 1 || epoch % print_freq == 0)
                @printf("[CD]  epoch %6d  cost = %.3e   diff = %.3e\n", epoch, cost_current, diff)
            end

            if diff < tol
                if verbose
                    @printf("[CD]  END   epoch %6d  cost = %.3e   diff = %.3e\n",epoch, cost_current, diff)
                end
                break
            end
            cost_prev = cost_current
        end
    end
    return x
end

end