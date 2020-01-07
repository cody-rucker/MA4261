# C := C + A * B
function mygemm_ijp!(C, A, B)
    M, N = size(C)
    K = size(A, 2)

    @assert size(C) == (M, N)
    @assert size(A) == (M, K)
    @assert size(B) == (K, N)

    for i = 1:M
        for j = 1:N
            for p = 1:K
                @inbounds C[i, j] = C[i,j] + A[i, p] * B[p, j]
            end
        end
    end
    return C
end

# Use loop ordering pji -> rank one update of matrix-matrix multiplication
function mygemm_pji!(C, A, B)
    M, N = size(C)
    K = size(A, 2)

    @assert size(C) == (M, N)
    @assert size(A) == (M, K)
    @assert size(B) == (K, N)

    for p = 1:K
        for j = 1:N
            for i = 1:M
                @inbounds C[i, j] = C[i,j] + A[i, p] * B[p, j]
            end
        end
    end
    return C
end

# jpi matrix-matrix multiplication
function mygemm_jpi!(C, A, B)
    M, N = size(C)
    K = size(A, 2)

    @assert size(C) == (M, N)
    @assert size(A) == (M, K)
    @assert size(B) == (K, N)

    @inbounds for j = 1:N
        for p = 1:K
            for i = 1:M
                C[i, j] = C[i,j] + A[i, p] * B[p, j]
            end
        end
    end
    return C
end

using LinearAlgebra
let
    # Compile with small matrices
    C = rand(3, 4)
    A = rand(3, 5)
    B = rand(5, 4)
    mygemm_ijp!(C, A, B)

    M, N, K = 256, 256, 256
    C = rand(M, N)
    A = rand(M, K)
    B = rand(K, N)

    nsamples = 10

    # @time C = C + A * B
    t_mul! = floatmax(Float64)
    for s = 1:nsamples
        t = @elapsed mul!(C, A, B)
        t_mul! = min(t_mul!, t)
    end
    @show t_mul!

    t_mygemm_ijp! = floatmax(Float64)
    for s = 1:nsamples
        t = @elapsed mygemm_ijp!(C, A, B)
        t_mygemm_ijp! = min(t_mygemm_ijp!, t)
    end
    @show t_mygemm_ijp!

    t_mygemm_pji! = floatmax(Float64)
    for s = 1:nsamples
        t = @elapsed mygemm_pji!(C, A, B)
        t_mygemm_pji! = min(t_mygemm_pji!, t)
    end
    @show t_mygemm_pji!

    t_mygemm_jpi! = floatmax(Float64)
    for s = 1:nsamples
        t = @elapsed mygemm_jpi!(C, A, B)
        t_mygemm_jpi! = min(t_mygemm_jpi!, t)
    end
    @show t_mygemm_jpi!
end

nothing
