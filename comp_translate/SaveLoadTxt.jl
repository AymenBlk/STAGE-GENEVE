module SaveLoadTxt
export dump_txt, load_txt

const HEADER = "# translate2compare 1.0"

import Printf: @sprintf

# ── helpers ────────────────────────────────────────────────────────────────
function _describe(obj)
    if obj isa Integer
        return "scalar", "int64", nothing
    elseif obj isa AbstractFloat
        obj isa Float64 || throw(ArgumentError("Only Float64 supported"))
        return "scalar", "float64", nothing
    elseif obj isa AbstractVector
        T = eltype(obj)
        dtype = T <: Integer ? "int64" :
                T == Float64  ? "float64" :
                throw(ArgumentError("Unsupported eltype $T"))
        return "vector", dtype, (length(obj),)
    elseif obj isa AbstractMatrix
        T = eltype(obj)
        dtype = T <: Integer ? "int64" :
                T == Float64  ? "float64" :
                throw(ArgumentError("Unsupported eltype $T"))
        return "matrix", dtype, size(obj)
    else
        throw(ArgumentError("Unsupported object $(typeof(obj))"))
    end
end

_token(x::Int)      = string(x)
_token(x::Float64)  = @sprintf("%.17g", x)

function _write_object(io, obj)
    kind, dtype, dims = _describe(obj)
    if kind == "scalar"
        println(io, "# object $kind $dtype")
        println(io, _token(obj))
    elseif kind == "vector"
        n = dims[1]; println(io, "# object $kind $dtype $n")
        println(io, join((_token(x) for x in obj), ' '))
    else  # matrix
        r, c = dims; println(io, "# object $kind $dtype $r $c")
        for i in 1:r
            println(io, join((_token(obj[i,j]) for j in 1:c), ' '))
        end
    end
end

"""
    dump_txt(path, objects; append=true)

Serialise each object in `objects` to `path` (translate2compare format).
Set `append=false` to overwrite the file, otherwise new objects are appended.
"""
function dump_txt(path::AbstractString, objects; append::Bool=true)
    isempty(objects) && return
    mode = append ? "a" : "w"
    first_write = !isfile(path) || !append
    open(path, mode) do io
        first_write && println(io, HEADER)
        for obj in objects
            _write_object(io, obj)
        end
    end
    return nothing
end

# ── reading ────────────────────────────────────────────────────────────────
function _parse_header(io)
    hdr = readline(io) |> strip
    hdr == HEADER || error("Bad header (got “$(hdr)”)")
end

function _parse_descriptor(line)
    startswith(line, "# object ") || error("Malformed descriptor")
    parts = split(strip(line[10:end]))
    length(parts) ≥ 2 || error("Descriptor too short")
    # kind, dtype = parts[1:2]...
    # dims = length(parts) > 2 ? parse.(Int, parts[3:end]) : Int[]
    # return kind, dtype, dims
    kind = parts[1]
    dtype = parts[2]
    dims = length(parts) > 2 ? parse.(Int, parts[3:end]) : Int[]
    return kind, dtype, dims
end

function _readtokens(io, n)
    toks = split(readline(io))
    length(toks) == n || error("Expected $n tokens")
    return toks
end

function _convert(toks, dtype)
    if dtype == "int64"
        return parse.(Int64, toks)
    elseif dtype == "float64"
        return parse.(Float64, toks)
    else
        error("Unsupported dtype $dtype")
    end
end

"""
    load_txt(path) → Vector{Any}

Read every object stored in `path` and return them in file order.
"""
function load_txt(path::AbstractString)
    objs = Any[]
    open(path) do io
        _parse_header(io)
        while !eof(io)
            line = strip(readline(io))
            isempty(line) && continue
            k, dt, dims = _parse_descriptor(line)
            if k == "scalar"
                push!(objs, dt == "int64" ? parse(Int64, strip(readline(io)))
                                          : parse(Float64, strip(readline(io))))
            elseif k == "vector"
                n = (length(dims) == 1) ? dims[1] : error("Vector needs 1 dim")
                push!(objs, _convert(_readtokens(io, n), dt))
            elseif k == "matrix"
                r, c = (length(dims) == 2) ? dims : error("Matrix needs 2 dims")
                M = dt == "int64" ? Array{Int64}(undef, r, c) : Array{Float64}(undef, r, c)
                for i in 1:r
                    M[i, :] = _convert(_readtokens(io, c), dt)
                end
                push!(objs, M)
            else
                error("Unknown kind $k")
            end
        end
    end
    return objs
end

end # module
