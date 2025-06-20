module metrics

function pesr(β::AbstractVector, β̂::AbstractVector; tol::Real = 0.0)
    support     = findall(b -> abs(b) > tol, β)
    support_hat = findall(b -> abs(b) > tol, β̂)
    return support == support_hat ? 1.0 : 0.0
end

function tpr(y::AbstractVector{Bool}, ŷ::AbstractVector{Bool})
    TP = sum(y .& ŷ)
    FN = sum(y .& .!ŷ)
    return TP == 0 ? 0.0 : TP / (TP + FN)
end

function fdr(y::AbstractVector{Bool}, ŷ::AbstractVector{Bool})
    TP = sum(y .& ŷ)
    FP = sum(.!y .& ŷ)
    return (TP + FP) == 0 ? 0.0 : FP / (TP + FP)
end

function f1score(y::AbstractVector{Bool}, ŷ::AbstractVector{Bool})
    TP = sum(y .& ŷ)
    FP = sum(.!y .& ŷ)
    FN = sum(y .& .!ŷ)
    return (2TP) == 0 ? 0.0 : 2TP / (2TP + FP + FN)
end

end