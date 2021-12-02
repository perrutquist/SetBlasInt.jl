# SetBlasInt

Note: This is experimental code, intended as a temporary fix, and not for production use. 

This package works with Julia v1.7.0 *only*.

The SetBlasInt package exports a single function, `setblasint`, which re-defines methods inside the `LinearAlgebra.BLAS` module to use a specified type of integer in calls to linear algebra subroutines. (The redefinition is done via `eval` on code modified from julia/stdlib/LinearAlgebra/src/blas.jl)

This allows for re-directing calls to libraries with 32 bit integer arguments, loaded via libblastrampoline, even in 64-bit builds of Julia.

The simplest use is:
```julia
using SetBlasInt
setblasint(Int32, :all) 
```
which redefines all the methods that `setblasint` knows about to use 32-bit integers. (Currently this encompasses just the gemm and syrk methods.)

Alternatively, the methods to be redefined can be listed individually. For example:
```julia
setblasint(Int32, :sgemm, :ssyrk)
```
will only re-define the "s" (Float32) versions of `BLAS.gemm!` and `BLAS.syrk!`.

Of course, for this to work, an external BLAS library must first be loaded using `lbt_forward`. For example, this will load BLAS from the Accelerate framework on a Mac:
```julia
using LinearAlgebra
BLAS.lbt_forward("/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate")
```

Footnote: The re-definitions do not affect currently running functions. Typically, `setblasint` should only be called once, from the top level.
