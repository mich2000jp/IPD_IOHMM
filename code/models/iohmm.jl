## logsumexp helper: fast version
@inline function manual_logsumexp(log_trans, k, log_alpha, K)
    max_val = log_alpha[1] + log_trans[1, k]
    @inbounds @simd for j in 2:K
        candidate = log_alpha[j] + log_trans[j, k]
        max_val = max(max_val, candidate)
    end
    
    sum_exp = zero(eltype(log_alpha))
    @inbounds @simd for j in 1:K
        sum_exp += exp(log_alpha[j] + log_trans[j, k] - max_val)
    end
    
    return max_val + log(sum_exp)
end

## Compute transition matrix
function compute_transition_matrix(gamma0::Matrix{T}, gamma::Array{T,3}, x::Vector, K::Int) where T
    log_trans = Matrix{T}(undef, K, K)
    
    for j in 1:K
        idx = 1
        for k in 1:K
            log_trans[k, j] = 0.0
            if k == j
            else
                log_trans[k, j] = gamma0[idx, j] + dot(view(gamma, :, idx, j), x)
                idx += 1
            end
        end
        
        lse = logsumexp(view(log_trans, :, j))
        for k in 1:K
            log_trans[k, j] = exp(log_trans[k, j] - lse)
        end
    end
    
    return log_trans
end

## Input-Output HMM (Mealy)
@model function iohmm_mealy(data::ExperimentData, K::Int; track::Bool = false)
    T_len   = data.n_periods
    N       = data.n_subjects
    D       = data.d_cov

    # =====================================================
    # Priors
    mean_beta = zeros(K)
    param_init = ones(K)

    beta0 ~ MvNormal(mean_beta,2.25 * I)
    beta1 ~ MvNormal(mean_beta,2.25 * I)
    beta2 ~ MvNormal(mean_beta,2.25 * I)
    beta3 ~ MvNormal(mean_beta,2.25 * I)
    beta  = hcat(beta1, beta2, beta3)


    gamma0 ~ filldist(Normal(0, 1.5), K-1, K)
    gamma ~ filldist(Normal(0, 1.5), D, K-1, K)
    init ~ Dirichlet(param_init)
    log_init = log.(init)

    # =====================================================
    # Likelihood
    Tp = eltype(beta0)
    log_lik = Vector{Tp}(undef, N)

    log_alpha      = Vector{Tp}(undef, K)
    log_alpha_next = Vector{Tp}(undef, K)
    log_trans      = Matrix{Tp}(undef, K, K)
    logits         = Vector{Tp}(undef, K)

    for i in 1:N
        y_seq = data.observations[i]
        X_seq = data.covariates[i]

        # t = 1: 
        mul!(logits, beta, view(X_seq, 1, :))
        @inbounds for k in 1:K
            logits[k] += beta0[k]
            le = y_seq[1] == 1 ?
                -log1pexp(-logits[k]) :
                -log1pexp( logits[k])
            log_alpha[k] = log_init[k] + le
        end

        # t = 2,...,T
        for t in 2:T_len
            x_trans = view(X_seq, t-1, :)
            
            # Compute transition matrix [from, to]
            @inbounds for j in 1:K
                idx = 1
                for k in 1:K
                    log_trans[k, j] = 0.0
                    if k == j
                    else
                        log_trans[k, j] = gamma0[idx, j] + dot(view(gamma, :, idx, j), x_trans)
                        idx += 1
                    end 
                end
                
                lse = logsumexp(view(log_trans, :, j))
                @simd for k in 1:K
                    log_trans[k, j] -= lse
                end
            end
            
            mul!(logits, beta, view(X_seq, t, :))
            @simd for k in 1:K
                logits[k] += beta0[k]
            end
            
            @inbounds for k in 1:K
                lp = manual_logsumexp(log_trans, k, log_alpha, K)
                
                le = y_seq[t] == 1 ?
                    -log1pexp(-logits[k]) :
                    -log1pexp( logits[k])
                
                log_alpha_next[k] = lp + le
            end

            log_alpha, log_alpha_next = log_alpha_next, log_alpha
        end

        log_lik[i] = logsumexp(log_alpha)
        Turing.@addlogprob! log_lik[i]
    end

    # =====================================================
    # Generated quantities
    if track
        x_dd = [0.0, 0.0, 0.0]
        x_dc = [0.0, 1.0, 0.0]
        x_cd = [1.0, 0.0, 0.0]
        x_cc = [1.0, 1.0, 1.0]
        
        logit_dd = beta0 .+ beta * x_dd
        logit_dc = beta0 .+ beta * x_dc
        logit_cd = beta0 .+ beta * x_cd
        logit_cc = beta0 .+ beta * x_cc

        pdd = logistic.(logit_dd)
        pdc = logistic.(logit_dc)
        pcd = logistic.(logit_cd)
        pcc = logistic.(logit_cc)

        trans_dd = compute_transition_matrix(gamma0, gamma, x_dd, K)
        trans_dc = compute_transition_matrix(gamma0, gamma, x_dc, K)
        trans_cd = compute_transition_matrix(gamma0, gamma, x_cd, K)
        trans_cc = compute_transition_matrix(gamma0, gamma, x_cc, K)
        
        return (;
            pdd, pdc, pcd, pcc,
            trans_dd, trans_dc, trans_cd, trans_cc
        )
    else
        return (; log_lik)
    end
end

## Input-Output HMM (Moore)
@model function iohmm_moore(data::ExperimentData, K::Int; track::Bool = false)
    T_len   = data.n_periods
    N       = data.n_subjects
    D       = data.d_cov

    # =====================================================
    # Priors
    mean_beta = zeros(K)
    param_init = ones(K)
    
    beta0_raw ~ Bijectors.ordered(MvNormal(mean_beta, 2.25 * I))
    beta0 = sort(beta0_raw)

    gamma0 ~ filldist(Normal(0, 1.5), K-1, K)
    gamma ~ filldist(Normal(0, 1.5), D, K-1, K)

    init ~ Dirichlet(param_init)
    log_init = log.(init)

    # =====================================================
    # Likelihood
    Tp = eltype(beta0)
    log_lik = Vector{Tp}(undef, N)

    log_alpha      = Vector{Tp}(undef, K)
    log_alpha_next = Vector{Tp}(undef, K)
    log_trans      = Matrix{Tp}(undef, K, K)

    for i in 1:N
        y_seq = data.observations[i]
        X_seq = data.covariates[i]

        # t = 1: 
        @inbounds for k in 1:K
            le = y_seq[1] == 1 ?
                -log1pexp(-beta0[k]) :
                -log1pexp( beta0[k])
            log_alpha[k] = log_init[k] + le
        end

        # t = 2,...,T
        for t in 2:T_len
            x_trans = view(X_seq, t-1, :)
            
            # Compute transition matrix
            @inbounds for j in 1:K
                idx = 1
                for k in 1:K
                    if k == j
                        log_trans[k, j] = 0.0
                    else
                        log_trans[k, j] = gamma0[idx, j] + dot(view(gamma, :, idx, j), x_trans)
                        idx += 1
                    end 
                end
                
                lse = logsumexp(view(log_trans, :, j))
                @simd for k in 1:K
                    log_trans[k, j] -= lse
                end
            end
            
            
            @inbounds for k in 1:K
                lp = manual_logsumexp(log_trans, k, log_alpha, K)
                
                le = y_seq[t] == 1 ?
                    -log1pexp(-beta0[k]) :
                    -log1pexp( beta0[k])
                
                log_alpha_next[k] = lp + le
            end

            log_alpha, log_alpha_next = log_alpha_next, log_alpha
        end

        log_lik[i] = logsumexp(log_alpha)
        Turing.@addlogprob! log_lik[i]
    end

    # =====================================================
    # Generated quantities
    # Generated quantities
    if track
        x_dd = [0.0, 0.0, 0.0]
        x_dc = [0.0, 1.0, 0.0]
        x_cd = [1.0, 0.0, 0.0]
        x_cc = [1.0, 1.0, 1.0]

        p = logistic.(beta0)

        trans_dd = compute_transition_matrix(gamma0, gamma, x_dd, K)
        trans_dc = compute_transition_matrix(gamma0, gamma, x_dc, K)
        trans_cd = compute_transition_matrix(gamma0, gamma, x_cd, K)
        trans_cc = compute_transition_matrix(gamma0, gamma, x_cc, K)
        
        return (;
            p,
            trans_dd, trans_dc, trans_cd, trans_cc
        )
    else
        return (; log_lik)
    end
end
