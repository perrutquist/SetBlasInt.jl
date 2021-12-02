using SetBlasInt
using LinearAlgebra
using Test

S = rand(Float32, 10, 10)
D = rand(Float64, 10, 10)
C = rand(Complex{Float32}, 10, 10)
Z = rand(Complex{Float64}, 10, 10)

SS = S*S
DD = D*D
CC = C*C
ZZ = Z*Z

StS = S'*S
DtD = D'*D
CtC = C'*C
ZtZ = Z'*Z

setblasint(Int64, :all)

@testset "SetBlasInt 64" begin
    # GEMM
    @test SS ≈ S*S
    @test DD ≈ D*D
    @test CC ≈ C*C
    @test ZZ ≈ Z*Z
    
    # SYRK
    @test StS ≈ S'*S
    @test DtD ≈ D'*D
    @test CtC ≈ C'*C
    @test ZtZ ≈ Z'*Z
end

# TODO: Test on more systems
if Sys.isapple()
    BLAS.lbt_forward("/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate")

    setblasint(Int32, :all)

    @testset "SetBlasInt 32" begin
        # GEMM
        @test SS ≈ S*S
        @test DD ≈ D*D
        @test CC ≈ C*C
        @test ZZ ≈ Z*Z
        
        # SYRK
        @test StS ≈ S'*S
        @test DtD ≈ D'*D
        @test CtC ≈ C'*C
        @test ZtZ ≈ Z'*Z
    end
end
