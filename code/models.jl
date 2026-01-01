"""
State decoding using Viterbi algorithm
"""

## Extract samples of parameters from MCMC chain
function extract_params_glmhmm(chain, K)
    
    n_iter = size(chain, 1)
    n_chains = size(chain, 3)
    n_samples = n_iter * n_chains
    
    param_names = names(chain)
    param_dict = Dict(pname => i for (i, pname) in enumerate(param_names))
    chain_array = chain.value.data  # Chain is 3-dimensional Aray: (iter, params, chain)
    
    # Vector of parameter samples (n_iter * n_chains Vector)
    beta0_samples = Vector{Vector{Float64}}()
    beta_samples  = Vector{Matrix{Float64}}()
    trans_samples = Vector{Matrix{Float64}}()
    init_samples  = Vector{Vector{Float64}}()


    # Reconstruct parameters for all samples ---------------------------------------------
        
    beta0_idx = [param_dict[Symbol("beta0[$k]")] for k in 1:K]
    
    beta_idx = []
    for d in 1:3
        for k in 1:K
            push!(beta_idx, param_dict[Symbol("beta[$d, $k]")])
        end
    end

    trans_idx = []
    for to in 1:K
        for from in 1:K
            push!(trans_idx, param_dict[Symbol("trans[$from, $to]")])
        end
    end

    init_idx = [param_dict[Symbol("init[$k]")] for k in 1:K]

    for ch in 1:n_chains
        for iter in 1:n_iter
            # beta0
            beta0_m = [chain_array[iter, idx, ch] for idx in beta0_idx]
            push!(beta0_samples, beta0_m)
            
            # beta (reshape to D x K)
            beta_vec = [chain_array[iter, idx, ch] for idx in beta_idx]
            beta_m = reshape(beta_vec, 3, K)
            push!(beta_samples, beta_m)
            
            # trans (reshape to K x K)
            trans_vec = [chain_array[iter, idx, ch] for idx in trans_idx]
            trans_m = reshape(trans_vec, K, K)
            push!(trans_samples, trans_m)

            # init
            init_m = [chain_array[iter, idx, ch] for idx in init_idx]
            push!(init_samples, init_m)
        end
    end
    
    return beta0_samples, beta_samples, trans_samples, init_samples
end

## Viterbi algorithm for GLM-HMM
function viterbi_glmhmm(data::ExperimentData, subject_idx::Int, beta0::Vector, beta::Matrix, trans::Matrix, init::Vector)
    K     = length(init)
    T_len = data.n_periods

    y_seq = data.observations[subject_idx]
    X_seq = data.covariates[subject_idx]

    log_trans = log.(trans)
    log_init  = log.(init)

    # Viterbi tables
    log_delta = Matrix{Float64}(undef, T_len, K)
    psi       = Matrix{Int}(undef, T_len, K)

    logits = Vector{Float64}(undef, K)

    # t = 1
    mul!(logits, beta', view(X_seq, 1, :))
    for k in 1:K
        logits[k] += beta0[k]
        log_emit = y_seq[1] == 1 ?
            -log1pexp(-logits[k]) :
            -log1pexp( logits[k])

        log_delta[1, k] = log_init[k] + log_emit
        psi[1, k] = 0
    end

    # t = 2,...,T
    for t in 2:T_len
        mul!(logits, beta', view(X_seq, t, :))
        @inbounds for k in 1:K
            logits[k] += beta0[k]
        end

        for k in 1:K
            log_emit = y_seq[t] == 1 ?
                -log1pexp(-logits[k]) :
                -log1pexp( logits[k])

            vals = @views log_delta[t-1, :] .+ log_trans[:, k]
            j = argmax(vals)

            psi[t, k]       = j
            log_delta[t, k] = vals[j] + log_emit
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

## return (subject, time, sample)
function decode_glmhmm(data::ExperimentData, chain, K::Int; n_samples::Int = 100)
    N = data.n_subjects
    T_len = data.n_periods
    
    # Extract vector of parameter samples
    beta0_samples, beta_samples, trans_samples, init_samples = 
        extract_params_glmhmm(chain, K)
    
    # Sample indices (e.g., use thinned posterior samples)
    total_samples = length(beta0_samples)
    sample_indices = round.(Int, range(1, total_samples, length=min(n_samples, total_samples)))
        
    # Store all decoded states
    all_states = Array{Int}(undef, N, T_len, length(sample_indices))
    
    println("Decoding states for $N subjects using $(length(sample_indices)) posterior samples...")
    println("Total MCMC samples available: $total_samples")
    
    for (idx, s_idx) in enumerate(sample_indices)
        if idx % 10 == 0
            print("\rProgress: $idx/$(length(sample_indices))")
        end
        
        for i in 1:N
            all_states[i, :, idx] = 
                viterbi_glmhmm(
                    data, i,
                    beta0_samples[s_idx],
                    beta_samples[s_idx],
                    trans_samples[s_idx],
                    init_samples[s_idx]
                )
        end
    end
    println("\nDecoding completed!")
    
    return all_states
end


## Extract samples of parameters from MCMC chain
function extract_params_iohmm(chain, K, D)
    n_iter = size(chain, 1)
    n_chains = size(chain, 3)
    
    param_names = names(chain)
    param_dict = Dict(pname => i for (i, pname) in enumerate(param_names))
    chain_array = chain.value.data  # (iter, params, chain)
    
    beta0_samples  = Vector{Vector{Float64}}()
    beta_samples   = Vector{Matrix{Float64}}()
    gamma0_samples = Vector{Matrix{Float64}}()
    gamma_samples  = Vector{Array{Float64, 3}}()
    init_samples   = Vector{Vector{Float64}}()
    
    # Pre-calculate indices
    # beta0 [K]
    beta0_idx = [param_dict[Symbol("beta0[$k]")] for k in 1:K]
    
    # beta [D, K]
    beta_idx = Int[]
    for k in 1:K
        for d in 1:D
            push!(beta_idx, param_dict[Symbol("beta[$d, $k]")])
        end
    end
    
    # gamma0 [K-1, K]
    gamma0_idx = Int[]
    for col in 1:K           # 2nd dim
        for row in 1:(K-1)   # 1st dim
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
            
            # gamma0 [K-1, K]
            g0_vec = [chain_array[iter, i, ch] for i in gamma0_idx]
            push!(gamma0_samples, reshape(g0_vec, K-1, K))
            
            # gamma [D, K-1, K]
            g_vec = [chain_array[iter, i, ch] for i in gamma_idx]
            push!(gamma_samples, reshape(g_vec, D, K-1, K))
            
            # init
            push!(init_samples, [chain_array[iter, i, ch] for i in init_idx])
        end
    end
    
    return beta0_samples, beta_samples, gamma0_samples, gamma_samples, init_samples
end

## Viterbi algorithm for IO-HMM
function viterbi_iohmm(data::ExperimentData, subject_idx::Int, 
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
    log_trans = Matrix{Float64}(undef, K, K)

    # --- t = 1 ---
    mul!(logits, beta', view(X_seq, 1, :))
    @inbounds for k in 1:K
        logits[k] += beta0[k]
        log_emit = y_seq[1] == 1 ? -log1pexp(-logits[k]) : -log1pexp(logits[k])
        log_delta[1, k] = log_init[k] + log_emit
        psi[1, k] = 0
    end

    # --- t = 2...T ---
    for t in 2:T_len
        # Transition matrix based on x_{t-1}
        x_trans = view(X_seq, t-1, :)
        
        # Compute log_trans[from, to] following io_hmm model
        @inbounds for j in 1:K  # to state
            idx = 1
            for k in 1:K  # from state
                if k == j
                    log_trans[k, j] = 1.0  # self-transition baseline
                else
                    log_trans[k, j] = gamma0[idx, j] + dot(view(gamma, :, idx, j), x_trans)
                    idx += 1
                end
            end
            
            # Normalize column j (from all states to state j)
            lse = logsumexp(view(log_trans, :, j))
            @simd for k in 1:K
                log_trans[k, j] -= lse
            end
        end

        # Emission at time t
        mul!(logits, beta', view(X_seq, t, :))
        @inbounds for k in 1:K
            logits[k] += beta0[k]
        end

        # Viterbi recursion
        for k in 1:K  # current state
            log_emit = y_seq[t] == 1 ? -log1pexp(-logits[k]) : -log1pexp(logits[k])
            
            best_val = -Inf
            best_j = 0
            
            for j in 1:K  # previous state
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

## Decode all subjects for IO-HMM
function decode_iohmm(data::ExperimentData, chain, K::Int; n_samples::Int = 100)
    N = data.n_subjects
    T_len = data.n_periods
    D = data.d_cov
    
    # Extract parameter samples
    beta0_s, beta_s, gamma0_s, gamma_s, init_s = extract_params_iohmm(chain, K, D)
    
    total_samples = length(beta0_s)
    sample_indices = round.(Int, range(1, total_samples, length=min(n_samples, total_samples)))
    
    all_states = Array{Int}(undef, N, T_len, length(sample_indices))
    
    println("Decoding IO-HMM for $N subjects using $(length(sample_indices)) posterior samples...")
    println("Total MCMC samples available: $total_samples")
    
    for (idx, s_idx) in enumerate(sample_indices)
        if idx % 10 == 0
            print("\rProgress: $idx/$(length(sample_indices))")
        end
        
        for i in 1:N
            all_states[i, :, idx] = viterbi_iohmm(
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


## Extract samples of parameters from MCMC chain
function extract_params_moore(chain, K, D)
    n_iter = size(chain, 1)
    n_chains = size(chain, 3)
    
    param_names = names(chain)
    param_dict = Dict(pname => i for (i, pname) in enumerate(param_names))
    chain_array = chain.value.data  # (iter, params, chain)
    
    beta0_samples  = Vector{Vector{Float64}}()
    gamma0_samples = Vector{Matrix{Float64}}()
    gamma_samples  = Vector{Array{Float64, 3}}()
    init_samples   = Vector{Vector{Float64}}()
    
    # Pre-calculate indices
    # beta0 [K]
    beta0_idx = [param_dict[Symbol("beta0[$k]")] for k in 1:K]
    
    # gamma0 [K-1, K]
    gamma0_idx = Int[]
    for col in 1:K           # 2nd dim
        for row in 1:(K-1)   # 1st dim
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
            
            # gamma0 [K-1, K]
            g0_vec = [chain_array[iter, i, ch] for i in gamma0_idx]
            push!(gamma0_samples, reshape(g0_vec, K-1, K))
            
            # gamma [D, K-1, K]
            g_vec = [chain_array[iter, i, ch] for i in gamma_idx]
            push!(gamma_samples, reshape(g_vec, D, K-1, K))
            
            # init
            push!(init_samples, [chain_array[iter, i, ch] for i in init_idx])
        end
    end
    
    return beta0_samples, gamma0_samples, gamma_samples, init_samples
end

## Viterbi algorithm for moore HMM with 4 covariates
function viterbi_moore(data::ExperimentData, subject_idx::Int, beta0::Vector, gamma0::Matrix, gamma::Array{Float64,3}, init::Vector)
    K     = length(init)
    T_len = data.n_periods
    y_seq = data.observations[subject_idx]
    X_seq = data.covariates[subject_idx] # T x D

    log_init  = log.(init)
    
    # Pre-allocate
    log_delta = Matrix{Float64}(undef, T_len, K)
    psi       = Matrix{Int}(undef, T_len, K)
    logits    = Vector{Float64}(undef, K)
    log_trans = Matrix{Float64}(undef, K, K)

    # --- t = 1 ---
    @inbounds for k in 1:K
        log_emit = y_seq[1] == 1 ? -log1pexp(-beta0[k]) : -log1pexp(beta0[k])
        log_delta[1, k] = log_init[k] + log_emit
        psi[1, k] = 0
    end

    # --- t = 2...T ---
    for t in 2:T_len
        x_trans = view(X_seq, t-1, :)
        
        # Compute log_trans[from, to]
        @inbounds for j in 1:K  # to state
            idx = 1
            for k in 1:K  # from state
                if k == j
                    log_trans[k, j] = 0.0
                else
                    log_trans[k, j] = gamma0[idx, j] + dot(view(gamma, :, idx, j), x_trans)
                    idx += 1
                end
            end
            
            # Normalize column j (from all states to state j)
            lse = logsumexp(view(log_trans, :, j))
            @simd for k in 1:K
                log_trans[k, j] -= lse
            end
        end

        # Viterbi recursion
        for k in 1:K  # current state
            log_emit = y_seq[t] == 1 ? -log1pexp(-beta0[k]) : -log1pexp(beta0[k])
            
            best_val = -Inf
            best_j = 0
            
            for j in 1:K  # previous state
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

## Decode all subjects for moore HMM with 4 covariates
function decode_moore(data::ExperimentData, chain, K::Int; n_samples::Int = 100)
    N = data.n_subjects
    T_len = data.n_periods
    D = data.d_cov
    
    # Extract parameter samples
    beta0_s, gamma0_s, gamma_s, init_s = extract_params_moore(chain, K, D)
    
    total_samples = length(beta0_s)
    sample_indices = round.(Int, range(1, total_samples, length=min(n_samples, total_samples)))
    
    all_states = Array{Int}(undef, N, T_len, length(sample_indices))
    
    println("Decoding IO-HMM for $N subjects using $(length(sample_indices)) posterior samples...")
    println("Total MCMC samples available: $total_samples")
    
    for (idx, s_idx) in enumerate(sample_indices)
        if idx % 10 == 0
            print("\rProgress: $idx/$(length(sample_indices))")
        end
        
        for i in 1:N
            all_states[i, :, idx] = viterbi_moore(
                data, i,
                beta0_s[s_idx],
                gamma0_s[s_idx],
                gamma_s[s_idx],
                init_s[s_idx]
            )
        end
    end
    println("\nDecoding completed!")
    
    return all_states
end


function decoder(model_name, data::ExperimentData, chain, K::Int; n_samples::Int = 100)
    if model_name == "glmhmm"
        return decode_glmhmm(data, chain, K, n_samples=n_samples)
    elseif model_name == "iohmm"
        return decode_iohmm(data, chain, K, n_samples=n_samples)
    elseif model_name == "moorehmm"
        return decode_moore(data, chain, K, n_samples=n_samples)
    else
        error("Unknown model name: $model_name")
    end
end
