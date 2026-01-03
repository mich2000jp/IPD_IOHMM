## GLM-HMM (covariate-independent transitions)
@model function glmhmm(data::ExperimentData, K::Int; track::Bool = false)
    T_len   = data.n_periods
    N       = data.n_subjects
    D       = data.d_cov

    # =====================================================
    # Priors 
    beta0 ~ filldist(Normal(0, 1.5), K)
    beta1 ~ filldist(Normal(0, 1.5), K)
    beta2 ~ filldist(Normal(0, 1.5), K)
    beta3 ~ filldist(Normal(0, 1.5), K)
    trans ~ filldist(Dirichlet(K, 1.5), K)
    init ~ Dirichlet(ones(K))

    beta = hcat(beta1, beta2, beta3)
    log_trans = log.(trans)
    log_init = log.(init)
    # =====================================================
    # Likelihood
    Tp = eltype(beta0)
    log_lik = Vector{Tp}(undef, N)

    log_alpha      = Vector{Tp}(undef, K)
    log_alpha_next = Vector{Tp}(undef, K)
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

        ll = logsumexp(log_alpha)
        log_lik[i] = ll
        Turing.@addlogprob! ll
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
        
        return (;
            pdd, pdc, pcd, pcc
        )
    else
        return (; log_lik)
    end
end
