# Translate to compare  

This folder is for functions "translating" some Julia/Python objects to Julia/Python objects.  
Because we work with random matrices and between these both languages we haven't the same random generator for the same seed we can't compare on very specific task.

So the idea is from a language we call a function and an object (vector or matrix) to write in a .txt file in a format explained below. Then with the other langage we can read the .txt to get the object.

## File format
``object <kind> <dtype> [dims]``    
``<ascii-encoded data> ``   
```
| Field         | Allowed values                       | Notes  
|---------------|--------------------------------------|--------------
| `kind`        | `scalar`, `vector`, `matrix`  

| `dtype`       | `float64`, `int64`   

| `dims`        | *vector*: `N`                        | *optional*
                | *matrix*: `rows cols`
                | *scalar*: omitted
```
### Data block rules
* **Scalar** exactly one line, one token.  
* **Vector** one line, `N` tokens.  
* **Matrix** `rows` lines, each with `cols` tokens.

Tokens are the full-precision `repr` of Julia or Python values, guaranteeing **bit-for-bit round-trips** for `Float64` and `Int64`.

Multiple objects are simply appended: every block starts with its own `# object …` line; existing data never moves.

#### Minimal example

```
# object matrix float64 2 3
1.0 2.0 3.0
4.0 5.0 6.0
# object vector int64 4
1 2 3 4
# object scalar float64
3.141592653589793

````

---

## Supported objects

| Julia                                 | Python / NumPy                  |
|---------------------------------------|---------------------------------|
| `Number` (`Int`, `Float64`)           | `int`, `float`                  |
| `AbstractVector{<:Int,Float64}`       | `numpy.ndarray` (`ndim == 1`)   |
| `AbstractMatrix{<:Int,Float64}`       | `numpy.ndarray` (`ndim == 2`)   |

Anything else triggers a clear `ArgumentError` / `ValueError`.

---

## Appending new data

Both `dump_txt` functions open the file in **append** mode by default.  
Pass `append=false` / `append=False` to start a fresh file.

Appending is atomic at the block level, so a partially written object is always detectable.

---

## Quick-start

### Julia → file → Python
```julia
using SaveLoadTxt

A = rand(2,3)          # Float64 matrix
v = rand(Int, 4)       # Int64 vector
dump_txt("example.txt", [A, v, π])

# … later, in Python …
from save_load_txt import load_txt
objs = load_txt("example.txt")
````

### Python → append → Julia

```python
import numpy as np, save_load_txt as slt

slt.dump_txt("example.txt",
             [np.asarray(42),
              np.ones((2,2), dtype=np.float64)],
             append=True)

# … back in Julia …
all_objs = load_txt("example.txt")
```

## Data
Data is a folder to store all matrices used to compare. If we want compare some function between both languages we try to do in appropriates Notebook in a (sub)section named **Comparison**.  
In this (sub)section, a sub section **setup** indicates objects involved and in which languages the setup has been made.