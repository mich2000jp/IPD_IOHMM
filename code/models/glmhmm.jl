## GLM-HMM (covariate-independent transitions)
@model function glmhmm(data::ExperimentData, K::Int; track::Bool = false)
    T_len   = data.n_periods
    N       = data.n_subjects
    D       = data.d_cov

    # =====================================================
    # Priors 
    mean_beta = zeros(K)
    param_init = ones(K)

    beta0 ~ MvNormal(mean_beta,2.25 * I)
    beta1_raw ~ MvNormal(mean_beta,2.25 * I)
    beta1 = sort(beta1_raw)
    beta2 ~ MvNormal(mean_beta,2.25 * I)
    beta3 ~ MvNormal(mean_beta,2.25 * I)
    beta  = hcat(beta1, beta2, beta3)
    
    # trans[from, to] 
    trans ~ filldist(Dirichlet(K, 1.0), K)
    log_trans = log.(trans)

    init ~ Dirichlet(param_init)
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
        mul!(logits, beta', view(X_seq, 1, :))
        @inbounds for k in 1:K
            logits[k] += beta0[k]
            le = y_seq[1] == 1 ?
                -log1pexp(-logits[k]) :
                -log1pexp( logits[k])
            log_alpha[k] = log_init[k] + le
        end

        # t = 2,...,T
        for t in 2:T_len
            mul!(logits, beta', view(X_seq, t, :))
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
        x_cc = [0.0, 0.0, 0.0]
        x_cd = [0.0, 1.0, 0.0]
        x_dc = [1.0, 0.0, 0.0]
        x_dd = [1.0, 1.0, 1.0]
        
        logit_cc = beta0 .+ beta' * x_cc
        logit_cd = beta0 .+ beta' * x_cd
        logit_dc = beta0 .+ beta' * x_dc
        logit_dd = beta0 .+ beta' * x_dd

        pcc = logistic.(logit_cc)
        pcd = logistic.(logit_cd)
        pdc = logistic.(logit_dc)
        pdd = logistic.(logit_dd)
        
        return (;
            pcc, pcd, pdc, pdd,
            beta1
        )
    else
        return (; log_lik)
    end
end