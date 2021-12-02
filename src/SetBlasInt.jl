module SetBlasInt

using LinearAlgebra

export setblasint

"""
    setblasint(T, symbols...)

Re-define methods inside the LinearAlgebra.BLAS module to use the specified type of integer.
"""
function setblasint(T, symbs...)
    exprs = setblasint_exprs(T, symbs)
    @eval LinearAlgebra.BLAS begin
        $(exprs...)
    end
    length(exprs)
end

function setblasint_exprs(::Type{BlasInt}, symbs) where {BlasInt<:Union{Int32, Int64}}
    function blasfunc(x) 
        BlasInt === Int32 ? Expr(:quote, x) : Expr(:quote, Symbol(x, "64_"))
    end
    exprs = []
    function pushfun(fun, expr)
        if endswith(String(fun), "_")
            fun = Symbol(String(fun)[1:end-1])
        end
        if symbs == (:all,) || fun in symbs
            push!(exprs, expr)
        end
    end

    # The code that appears as arguments to `pushfun` below is taken from Julia v1.7.0
    # https://github.com/JuliaLang/julia/blob/v1.7.0/stdlib/LinearAlgebra/src/blas.jl
    # with BlasInt changed to $BlasInt and @blasfunc($x) changed to $(blasfunc(x)).
    # (This is likely to be incompatible with other versions of Julia.)

    for (gemm, elty) in ((:dgemm_,:Float64),
                        (:sgemm_,:Float32),
                        (:zgemm_,:ComplexF64),
                        (:cgemm_,:ComplexF32))
        pushfun(gemm, quote
            function gemm!(transA::AbstractChar, transB::AbstractChar,
                       alpha::Union{($elty), Bool},
                       A::AbstractVecOrMat{$elty}, B::AbstractVecOrMat{$elty},
                       beta::Union{($elty), Bool},
                       C::AbstractVecOrMat{$elty})
                require_one_based_indexing(A, B, C)
                m = size(A, transA == 'N' ? 1 : 2)
                ka = size(A, transA == 'N' ? 2 : 1)
                kb = size(B, transB == 'N' ? 1 : 2)
                n = size(B, transB == 'N' ? 2 : 1)
                if ka != kb || m != size(C,1) || n != size(C,2)
                    throw(DimensionMismatch("A has size ($m,$ka), B has size ($kb,$n), C has size $(size(C))"))
                end
                chkstride1(A)
                chkstride1(B)
                chkstride1(C)
                ccall(($(blasfunc(gemm)), libblastrampoline), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{$BlasInt}, Ref{$BlasInt},
                    Ref{$BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{$BlasInt},
                    Ptr{$elty}, Ref{$BlasInt}, Ref{$elty}, Ptr{$elty},
                    Ref{$BlasInt}, Clong, Clong),
                    transA, transB, m, n,
                    ka, alpha, A, max(1,stride(A,2)),
                    B, max(1,stride(B,2)), beta, C,
                    max(1,stride(C,2)), 1, 1)
                C
            end
        end)
    end

    for (fname, elty) in ((:dsyrk_,:Float64),
                            (:ssyrk_,:Float32),
                            (:zsyrk_,:ComplexF64),
                            (:csyrk_,:ComplexF32))
        pushfun(fname, quote
            function syrk!(uplo::AbstractChar, trans::AbstractChar,
                    alpha::Union{($elty), Bool}, A::AbstractVecOrMat{$elty},
                    beta::Union{($elty), Bool}, C::AbstractMatrix{$elty})
                require_one_based_indexing(A, C)
                n = checksquare(C)
                nn = size(A, trans == 'N' ? 1 : 2)
                if nn != n throw(DimensionMismatch("C has size ($n,$n), corresponding dimension of A is $nn")) end
                k  = size(A, trans == 'N' ? 2 : 1)
                chkstride1(A)
                chkstride1(C)
                ccall(($(blasfunc(fname)), libblastrampoline), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{$BlasInt}, Ref{$BlasInt},
                    Ref{$elty}, Ptr{$elty}, Ref{$BlasInt}, Ref{$elty},
                    Ptr{$elty}, Ref{$BlasInt}, Clong, Clong),
                uplo, trans, n, k,
                alpha, A, max(1,stride(A,2)), beta,
                C, max(1,stride(C,2)), 1, 1)
                C
            end
        end)
    end

    exprs
end

end # module

