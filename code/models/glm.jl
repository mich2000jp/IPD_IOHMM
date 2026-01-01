## Baseline GLM model
@model function glm(data::ExperimentData; track::Bool = false)
    T_len   = data.n_periods
    N       = data.n_subjects
    D       = data.d_cov

    # =====================================================
    # Priors
    sigma = 1.5
    beta0 ~ Normal(0, sigma)
    beta  ~ filldist(Normal(0, sigma), D)

    # =====================================================
    # Likelihood
    Tp = eltype(beta0)
    log_lik = Vector{Tp}(undef, N)

    for i in 1:N
        y_seq = data.observations[i]
        X_seq = data.covariates[i]
        ll = 0.0

        # t = 1,...,T
        @inbounds for t in 1:T_len
            logit = beta0 + dot(beta, view(X_seq, t, :))
            ll += y_seq[t] == 1 ?
                -log1pexp(-logit) : # Pr(y=1) = logistic(logit)
                -log1pexp( logit)   # Pr(y=0) = 1 - logistic(logit)
        end

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
        
        logit_cc = beta0 + dot(beta, x_cc)
        logit_cd = beta0 + dot(beta, x_cd)
        logit_dc = beta0 + dot(beta, x_dc)
        logit_dd = beta0 + dot(beta, x_dd)

        pcc = logistic(logit_cc)
        pcd = logistic(logit_cd)
        pdc = logistic(logit_dc)
        pdd = logistic(logit_dd)
        
        return (;
            pcc, pcd, pdc, pdd,
        )
    else
        return (; log_lik)
    end
end