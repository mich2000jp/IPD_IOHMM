## logsumexp helper: safer version
@inline function manual_logsumexp(log_trans, k, log_alpha, K)
    # 初期値が -Inf の場合に問題
    max_val = log_alpha[1] + log_trans[1, k]

    # より安全な実装:
    max_val = typemin(eltype(log_alpha))
    @inbounds for j in 1:K
        candidate = log_alpha[j] + log_trans[j, k]
        max_val = max(max_val, candidate)
    end

    # max_val が -Inf の場合の処理
    if isinf(max_val)
        return max_val
    end

    sum_exp = zero(eltype(log_alpha))
    @inbounds for j in 1:K
        sum_exp += exp(log_alpha[j] + log_trans[j, k] - max_val)
    end

    return max_val + log(sum_exp)
end

# logsumexp helper: using LogExpFunctions.jl (slow)
@inline function manual_logsumexp(log_trans, k, log_alpha, K)
    return logsumexp(log_alpha[j] + log_trans[j, k] for j in 1:K)
end

# Moore-Machine HMM with covariate-dependent transitions, emissions depend on state only
@model function moore_hmm4x(data::ExperimentData, K::Int; track::Bool = false)
    T_len   = data.n_periods
    N       = data.n_subjects
    D       = data.d_cov

    # =====================================================
    # Priors
    mean_beta = [0.0, 0.0]
    param_init = [1.0, 1.0]
    
    beta0 ~ Bijectors.ordered(MvNormal(mean_beta, 2.25 * I))

    gamma0 ~ filldist(Normal(0, 0.5), K-1, K)
    gamma ~ filldist(Normal(0, 0.5), D, K-1, K)

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
            x_trans = view(X_seq, t, :)
            
            # Compute transition matrix
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
    if track

        x_cc = [0.0, 0.0, 0.0]
        x_cd = [0.0, 1.0, 0.0]
        x_dc = [1.0, 0.0, 0.0]
        x_dd = [1.0, 1.0, 1.0]

        p = logistic.(beta0)

        trans_cc = compute_transition_matrix_4x(gamma0, gamma, x_cc, K)
        trans_cd = compute_transition_matrix_4x(gamma0, gamma, x_cd, K)
        trans_dc = compute_transition_matrix_4x(gamma0, gamma, x_dc, K)
        trans_dd = compute_transition_matrix_4x(gamma0, gamma, x_dd, K)
        
        return (;
            p,
            trans_cc, trans_cd, trans_dc, trans_dd,
            log_lik
        )
    else
        return (; log_lik)
    end
end

# Mealy-Machine HMM with covariate-dependent transitions
@model function mealy_hmm4x(data::ExperimentData, K::Int; track::Bool = false)
    T_len   = data.n_periods
    N       = data.n_subjects
    D       = data.d_cov

    # =====================================================
    # Priors
    mean_beta = zeros(K)
    param_init = ones(K)

    beta0 ~ MvNormal(mean_beta, 2.25 * I)
    beta  ~ filldist(Normal(0, 1.5), D, K)

    gamma0 ~ filldist(Normal(0, 1.5), K, K)
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
            x_trans = view(X_seq, t-1, :)
            
            # Compute transition matrix [from, to]
            @inbounds for j in 1:K
                idx = 1
                for k in 1:K
                    log_trans[k, j] = gamma0[k, j]  # baseline
                    if k == j
                        
                    else
                        log_trans[k, j] += dot(view(gamma, :, idx, j), x_trans)
                        idx += 1
                    end 
                end
                
                lse = logsumexp(view(log_trans, :, j))
                @simd for k in 1:K
                    log_trans[k, j] -= lse
                end
            end
            
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

        log_lik[i] = logsumexp(log_alpha)
        Turing.@addlogprob! log_lik[i]
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

        trans_cc = compute_transition_matrix_4x(gamma0, gamma, x_cc, K)
        trans_cd = compute_transition_matrix_4x(gamma0, gamma, x_cd, K)
        trans_dc = compute_transition_matrix_4x(gamma0, gamma, x_dc, K)
        trans_dd = compute_transition_matrix_4x(gamma0, gamma, x_dd, K)
        
        return (;
            pcc, pcd, pdc, pdd,
            trans_cc, trans_cd, trans_dc, trans_dd,
            log_lik
        )
    else
        return (; log_lik)
    end
end

# Compute transition matrix for Mealy/Moore-Machine
function compute_transition_matrix_4x(gamma0::Matrix{T}, gamma::Array{T,3}, x::Vector, K::Int) where T
    log_trans = Matrix{T}(undef, K, K)
    
    for j in 1:K
        idx = 1
        for k in 1:K
            if k == j
                log_trans[k, j] = gamma0[k, j]
            else
                log_trans[k, j] += dot(view(gamma, :, idx, j), x)
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

# Hierarchical IO-HMM with individual heterogeneity
@model function io_hmm_hierarchical(data::ExperimentData, K::Int; track::Bool = false)
    T_len   = data.n_periods
    N       = data.n_subjects
    D       = data.d_cov

    # =====================================================
    # Priors - Population level
    # =====================================================
    mean_beta = zeros(K)
    param_init = ones(K)

    # Shared parameters across subjects
    beta0 ~ MvNormal(mean_beta, 2.25 * I)
    beta  ~ filldist(Normal(0, 1.5), D, K)

    gamma0 ~ filldist(Normal(0, 1.5), K-1, K)
    gamma ~ filldist(Normal(0, 1.5), D, K-1, K)

    init ~ Dirichlet(param_init)
    log_init = log.(init)

    # Hierarchical priors for individual parameters
    theta ~ MvNormal(0, 2.25)     
    # Individual parameters:
    individual_params ~ filldist(Normal(0, 1.5), N)

    
    # =====================================================
    # Likelihood
    # =====================================================
    Tp = eltype(beta0)
    log_lik = Vector{Tp}(undef, N)

    log_alpha      = Vector{Tp}(undef, K)
    log_alpha_next = Vector{Tp}(undef, K)
    log_trans      = Matrix{Tp}(undef, K, K)
    log_trans_base = Matrix{Tp}(undef, K, K)
    logits         = Vector{Tp}(undef, K)

    for i in 1:N
        y_seq = data.observations[i]
        X_seq = data.covariates[i]
        
        # Extract individual parameters
        θ_i = individual_params[1, i]  # Baseline defection tendency
        logit_ρ_i = individual_params[2, i]
        ρ_i = logistic(logit_ρ_i)  # Self-transition propensity

        # t = 1: 
        mul!(logits, beta', view(X_seq, 1, :))
        @inbounds for k in 1:K
            logits[k] += beta0[k] + θ_i  # Add individual baseline
            le = y_seq[1] == 1 ?
                -log1pexp(-logits[k]) :
                -log1pexp( logits[k])
            log_alpha[k] = log_init[k] + le
        end

        # t = 2,...,T
        for t in 2:T_len
            x_trans = view(X_seq, t-1, :)
            
            # Compute base transition matrix A_{j,k}(x_{t-1})
            @inbounds for j in 1:K
                idx = 1
                for k in 1:K
                    if k == j
                        log_trans_base[k, j] = 0.0
                    else
                        log_trans_base[k, j] = gamma0[idx, j] + dot(view(gamma, :, idx, j), x_trans)
                        idx += 1
                    end 
                end
                
                # Normalize base transition (get probabilities)
                lse = logsumexp(view(log_trans_base, :, j))
                @simd for k in 1:K
                    log_trans_base[k, j] = exp(log_trans_base[k, j] - lse)
                end
            end
            
            # Apply individual-specific adjustment: A^(i)_{j,k}
            @inbounds for j in 1:K
                for k in 1:K
                    if k == j
                        # Self-transition: ρ^(i) + (1 - ρ^(i)) * A_{j,k}
                        log_trans[k, j] = log(ρ_i + (1 - ρ_i) * log_trans_base[k, j])
                    else
                        # Other transitions: (1 - ρ^(i)) * A_{j,k}
                        log_trans[k, j] = log((1 - ρ_i) * log_trans_base[k, j])
                    end
                end
            end
            
            # Compute emission probabilities
            mul!(logits, beta', view(X_seq, t, :))
            @simd for k in 1:K
                logits[k] += beta0[k] + θ_i  # Add individual baseline
            end
            
            # Forward algorithm step
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
    # =====================================================
    if track        
        x_cc = [0.0, 0.0, 0.0]
        x_cd = [0.0, 1.0, 0.0]
        x_dc = [1.0, 0.0, 0.0]
        x_dd = [1.0, 1.0, 1.0]
        
        # Population-level predictions (using mean θ and ρ)
        θ_mean = μ[1]
        ρ_mean = logistic(μ[2])
        
        logit_cc = (beta0 .+ θ_mean) .+ beta' * x_cc
        logit_cd = (beta0 .+ θ_mean) .+ beta' * x_cd
        logit_dc = (beta0 .+ θ_mean) .+ beta' * x_dc
        logit_dd = (beta0 .+ θ_mean) .+ beta' * x_dd

        pcc = logistic.(logit_cc)
        pcd = logistic.(logit_cd)
        pdc = logistic.(logit_dc)
        pdd = logistic.(logit_dd)

        # Compute base transition matrices
        trans_base_cc = compute_transition_matrix(gamma0, gamma, x_cc, K)
        trans_base_cd = compute_transition_matrix(gamma0, gamma, x_cd, K)
        trans_base_dc = compute_transition_matrix(gamma0, gamma, x_dc, K)
        trans_base_dd = compute_transition_matrix(gamma0, gamma, x_dd, K)
        
        # Apply population-level ρ adjustment
        trans_cc = adjust_transition_with_rho(trans_base_cc, ρ_mean, K)
        trans_cd = adjust_transition_with_rho(trans_base_cd, ρ_mean, K)
        trans_dc = adjust_transition_with_rho(trans_base_dc, ρ_mean, K)
        trans_dd = adjust_transition_with_rho(trans_base_dd, ρ_mean, K)
        
        σ_θ = sqrt(Σ[1, 1])
        σ_logit_ρ = sqrt(Σ[2, 2])
        correlation_θ_logit_ρ = Σ[1, 2] / (σ_θ * σ_logit_ρ)
        
        return (;
            pcc, pcd, pdc, pdd,
            trans_cc, trans_cd, trans_dc, trans_dd,
            correlation_θ_logit_ρ
        )
    else
        return (; log_lik)
    end
end

# Helper function to apply ρ adjustment to transition matrix
function adjust_transition_with_rho(trans_base::Matrix{T}, ρ::T, K::Int) where T
    trans_adjusted = Matrix{T}(undef, K, K)
    
    for j in 1:K
        for k in 1:K
            if k == j
                trans_adjusted[k, j] = ρ + (1 - ρ) * trans_base[k, j]
            else
                trans_adjusted[k, j] = (1 - ρ) * trans_base[k, j]
            end
        end
    end
    
    return trans_adjusted
end

# Extract parameters for Mealy HMM
function extract_params_mealy_hmm4x(chain, K, D)
    n_iter = size(chain, 1)
    n_chains = size(chain, 3)
    
    param_names = names(chain)
    param_dict = Dict(pname => i for (i, pname) in enumerate(param_names))
    chain_array = chain.value.data
    
    beta0_samples  = Vector{Vector{Float64}}()
    beta_samples   = Vector{Matrix{Float64}}()
    gamma0_samples = Vector{Matrix{Float64}}()
    gamma_samples  = Vector{Array{Float64, 3}}()
    init_samples   = Vector{Vector{Float64}}()
    
    # Pre-calculate indices to avoid repeated dictionary lookups
    # beta0 [K]
    beta0_idx = [param_dict[Symbol("beta0[$k]")] for k in 1:K]
    
    # beta [D, K]
    beta_idx = Int[]
    for k in 1:K
        for d in 1:D
            push!(beta_idx, param_dict[Symbol("beta[$d, $k]")])
        end
    end
    
    # gamma0 [K, K]
    gamma0_idx = Int[]
    for col in 1:K
        for row in 1:K
            push!(gamma0_idx, param_dict[Symbol("gamma0[$row, $col]")])
        end
    end
    
    # gamma [D, K-1, K]
    gamma_idx = Int[]
    for k in 1:K           # 3rd dim
        for j in 1:(K-1)   # 2nd dim
            for d in 1:D   # 1st dim
                push!(gamma_idx, param_dict[Symbol("gamma[$d, $j, $k]")])
            end
        end
    end
    
    # init [K]
    init_idx = [param_dict[Symbol("init[$k]")] for k in 1:K]

    for ch in 1:n_chains
        for iter in 1:n_iter
            # beta0
            push!(beta0_samples, [chain_array[iter, i, ch] for i in beta0_idx])
            
            # beta
            b_vec = [chain_array[iter, i, ch] for i in beta_idx]
            push!(beta_samples, reshape(b_vec, D, K))
            
            # gamma0
            g0_vec = [chain_array[iter, i, ch] for i in gamma0_idx]
            push!(gamma0_samples, reshape(g0_vec, K, K))
            
            # gamma
            g_vec = [chain_array[iter, i, ch] for i in gamma_idx]
            push!(gamma_samples, reshape(g_vec, D, K-1, K))
            
            # init
            push!(init_samples, [chain_array[iter, i, ch] for i in init_idx])
        end
    end
    
    return beta0_samples, beta_samples, gamma0_samples, gamma_samples, init_samples
end

# Viterbi algorithm for Mealy HMM
function viterbi_mealy_hmm4x(data::ExperimentData, subject_idx::Int, 
                             beta0::Vector, beta::Matrix, 
                             gamma0::Matrix, gamma::Array{Float64,3}, 
                             init::Vector)
    K     = length(init)
    T_len = data.n_periods
    y_seq = data.observations[subject_idx]
    X_seq = data.covariates[subject_idx] # T x D

    log_init  = log.(init)
    
    # Pre-allocate
    log_delta = Matrix{Float64}(undef, T_len, K)
    psi       = Matrix{Int}(undef, T_len, K)
    logits    = Vector{Float64}(undef, K)
    log_trans = Matrix{Float64}(undef, K, K) # reused buffer

    # --- t = 1 ---
    # Emission only
    mul!(logits, beta', view(X_seq, 1, :))
    @inbounds for k in 1:K
        logits[k] += beta0[k]
        log_emit = y_seq[1] == 1 ? -log1pexp(-logits[k]) : -log1pexp(logits[k])
        log_delta[1, k] = log_init[k] + log_emit
        psi[1, k] = 0
    end

    # --- t = 2...T ---
    for t in 2:T_len
        # Calculate Transition Matrix based on x_{t-1}
        # Note: Model uses view(X_seq, t-1, :) for transition to t
        x_trans = view(X_seq, t-1, :)
        
        # Reconstruct log_trans logic from mealy_hmm4x
        @inbounds for j in 1:K
            idx = 1
            for k in 1:K
                log_trans[k, j] = gamma0[k, j]
                if k != j
                    log_trans[k, j] += dot(view(gamma, :, idx, j), x_trans)
                    idx += 1
                end
            end
            
            # Normalize column j
            lse = logsumexp(view(log_trans, :, j))
            @simd for k in 1:K
                log_trans[k, j] -= lse
            end
        end

        # Calculate Emission at t
        mul!(logits, beta', view(X_seq, t, :))
        @inbounds for k in 1:K
            logits[k] += beta0[k]
        end

        # Viterbi recursion
        for k in 1:K
            log_emit = y_seq[t] == 1 ? -log1pexp(-logits[k]) : -log1pexp(logits[k])
            
            # Maximize: log_delta[t-1, j] + log_trans[j, k]
            # Note: log_trans is column-stochastic (normalized over rows for fixed col j) in the model definition?
            # Based on model code: manual_logsumexp(log_trans, k, ...) accesses log_trans[j, k]
            # where j is prev state, k is curr state.
            
            best_val = -Inf
            best_j = 0
            
            for j in 1:K
                val = log_delta[t-1, j] + log_trans[j, k]
                if val > best_val
                    best_val = val
                    best_j = j
                end
            end

            log_delta[t, k] = best_val + log_emit
            psi[t, k] = best_j
        end
    end

    # Backtracking
    states = Vector{Int}(undef, T_len)
    states[T_len] = argmax(log_delta[T_len, :])
    for t in (T_len-1):-1:1
        states[t] = psi[t+1, states[t+1]]
    end

    return states
end

# Decode wrapper for Mealy HMM
function decode_all_subjects_mealy_hmm4x(data::ExperimentData, chain, K::Int; n_samples::Int = 100)
    N = data.n_subjects
    T_len = data.n_periods
    D = data.d_cov
    
    # Extract params
    beta0_s, beta_s, gamma0_s, gamma_s, init_s = extract_params_mealy_hmm4x(chain, K, D)
    
    total_samples = length(beta0_s)
    sample_indices = round.(Int, range(1, total_samples, length=min(n_samples, total_samples)))
    
    all_states = Array{Int}(undef, N, T_len, length(sample_indices))
    
    println("Decoding Mealy HMM for $N subjects using $(length(sample_indices)) samples...")
    
    for (idx, s_idx) in enumerate(sample_indices)
        if idx % 10 == 0
            print("\rProgress: $idx/$(length(sample_indices))")
        end
        
        for i in 1:N
            all_states[i, :, idx] = viterbi_mealy_hmm4x(
                data, i,
                beta0_s[s_idx],
                beta_s[s_idx],
                gamma0_s[s_idx],
                gamma_s[s_idx],
                init_s[s_idx]
            )
        end
    end
    println("\nDecoding completed!")
    
    return all_states
end