## Viterbi algorithm for GLM-HMM
function viterbi_glmhmm(data::ExperimentData, subject_idx::Int, beta0::Vector, beta::Matrix, trans::Matrix, init::Vector)
    K     = length(init)
    T_len = data.n_periods

    y_seq = data.observations[subject_idx]
    X_seq = data.covariates[subject_idx]

    log_trans = log.(trans)
    log_init  = log.(init)
    log_delta = Matrix{Float64}(undef, T_len, K)
    psi       = Matrix{Int}(undef, T_len, K)
    logits = Vector{Float64}(undef, K)

    # t = 1
    mul!(logits, beta, view(X_seq, 1, :))
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
        mul!(logits, beta, view(X_seq, t, :))
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

## Viterbi algorithm for IO-HMM (Mealy)
function viterbi_iohmm_mealy(data::ExperimentData, subject_idx::Int, beta0::Vector, beta::Matrix, gamma0::Matrix, gamma::Array{Float64,3}, init::Vector)
    K     = length(init)
    T_len = data.n_periods
    y_seq = data.observations[subject_idx]
    X_seq = data.covariates[subject_idx]

    log_init  = log.(init)
    log_delta = Matrix{Float64}(undef, T_len, K)
    psi       = Matrix{Int}(undef, T_len, K)
    logits    = Vector{Float64}(undef, K)
    log_trans = Matrix{Float64}(undef, K, K)

    # t = 1 
    mul!(logits, beta, view(X_seq, 1, :))
    @inbounds for k in 1:K
        logits[k] += beta0[k]
        log_emit = y_seq[1] == 1 ? -log1pexp(-logits[k]) : -log1pexp(logits[k])
        log_delta[1, k] = log_init[k] + log_emit
        psi[1, k] = 0
    end

    # t = 2,...,T 
    for t in 2:T_len
        x_trans = view(X_seq, t-1, :)
        
        @inbounds for j in 1:K  # to state
            idx = 1
            for k in 1:K  # from state
                if k == j
                    log_trans[k, j] = 0.0
                else
                    log_trans[k, j] = gamma0[idx, j] + dot(view(gamma, idx, j, :), x_trans)
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
        mul!(logits, beta, view(X_seq, t, :))
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

## Viterbi algorithm for IO-HMM (Moore)
function viterbi_iohmm_moore(data::ExperimentData, subject_idx::Int, beta0::Vector, gamma0::Matrix, gamma::Array{Float64,3}, init::Vector)
    K     = length(init)
    T_len = data.n_periods
    y_seq = data.observations[subject_idx]
    X_seq = data.covariates[subject_idx]

    log_init  = log.(init)
    log_delta = Matrix{Float64}(undef, T_len, K)
    psi       = Matrix{Int}(undef, T_len, K)
    log_trans = Matrix{Float64}(undef, K, K)

    # t = 1 
    @inbounds for k in 1:K
        log_emit = y_seq[1] == 1 ? -log1pexp(-beta0[k]) : -log1pexp(beta0[k])
        log_delta[1, k] = log_init[k] + log_emit
        psi[1, k] = 0
    end

    # t = 2,...,T 
    for t in 2:T_len
        x_trans = view(X_seq, t-1, :)
        
        @inbounds for j in 1:K  # to state
            idx = 1
            for k in 1:K  # from state
                if k == j
                    log_trans[k, j] = 0.0
                else
                    log_trans[k, j] = gamma0[idx, j] + dot(view(gamma, idx, j, :), x_trans)
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

## Extract samples of parameters from MCMC chain
function extract_params(model::String, chain::Chains, K::Int)
    n_iter = size(chain, 1)
    n_chains = size(chain, 3)
    D = 3
    
    param_names = names(chain)
    param_dict = Dict(pname => i for (i, pname) in enumerate(param_names))
    chain_array = chain.value.data  # (iter, params, chain)
    
    beta0_samples  = Vector{Vector{Float64}}()
    beta_samples   = Vector{Matrix{Float64}}()
    trans_samples  = Vector{Matrix{Float64}}()
    gamma0_samples = Vector{Matrix{Float64}}()
    gamma_samples = Vector{Array{Float64,3}}()
    init_samples   = Vector{Vector{Float64}}()
    
    
    # Pre-calculate indices
    # beta0 [K]
    beta0_idx = [param_dict[Symbol("beta0[$k]")] for k in 1:K]
    init_idx = [param_dict[Symbol("init[$k]")] for k in 1:K]

    if model == "iohmm_mealy" || model == "glmhmm"
        beta1_idx = [param_dict[Symbol("beta1[$k]")] for k in 1:K]
        beta2_idx = [param_dict[Symbol("beta2[$k]")] for k in 1:K]
        beta3_idx = [param_dict[Symbol("beta3[$k]")] for k in 1:K]
    end

    # gamma0[K-1, K]
    if model == "iohmm_mealy" || model == "iohmm_moore"
        gamma0_idx = Int[]
        for col in 1:K           # 2nd dim
            for row in 1:(K-1)   # 1st dim
                push!(gamma0_idx, param_dict[Symbol("gamma0[$row, $col]")])
            end
        end

        gamma1_idx = Int[]
        for col in 1:K           # 2nd dim
            for row in 1:(K-1)   # 1st dim
                push!(gamma1_idx, param_dict[Symbol("gamma1[$row, $col]")])
            end
        end

        gamma2_idx = Int[]
        for col in 1:K           # 2nd dim
            for row in 1:(K-1)   # 1st dim
                push!(gamma2_idx, param_dict[Symbol("gamma2[$row, $col]")])
            end
        end

        gamma3_idx = Int[]
        for col in 1:K           # 2nd dim
            for row in 1:(K-1)   # 1st dim
                push!(gamma3_idx, param_dict[Symbol("gamma3[$row, $col]")])
            end
        end 
    end    

    if model == "glmhmm"
        trans_idx = Int[]
        for col in 1:K           # 2nd dim
            for row in 1:K       # 1st dim
                push!(trans_idx, param_dict[Symbol("trans[$row, $col]")])
            end
        end
    end

    for ch in 1:n_chains
        for iter in 1:n_iter
            push!(beta0_samples, [chain_array[iter, i, ch] for i in beta0_idx])
            push!(init_samples, [chain_array[iter, i, ch] for i in init_idx])

            if model == "iohmm_mealy" || model == "glmhmm"
                # beta [K x D]
                b1_vec = [chain_array[iter, i, ch] for i in beta1_idx]
                b2_vec = [chain_array[iter, i, ch] for i in beta2_idx]
                b3_vec = [chain_array[iter, i, ch] for i in beta3_idx]
                beta_m = hcat(reshape(b1_vec, K), reshape(b2_vec, K), reshape(b3_vec, K))
                push!(beta_samples, beta_m)
            end

            if model == "iohmm_mealy" || model == "iohmm_moore"
                # gamma0 [K-1, K]
                g0_vec = [chain_array[iter, i, ch] for i in gamma0_idx]
                push!(gamma0_samples, reshape(g0_vec, K-1, K))
                
                # gamma [K-1, K, D]
                g1_vec = [chain_array[iter, i, ch] for i in gamma1_idx]
                g2_vec = [chain_array[iter, i, ch] for i in gamma2_idx]
                g3_vec = [chain_array[iter, i, ch] for i in gamma3_idx]
                gamma_m = cat(reshape(g1_vec, K-1, K), reshape(g2_vec, K-1, K), reshape(g3_vec, K-1, K); dims = 3)
                push!(gamma_samples, gamma_m)
            end

            if model == "glmhmm"
                # trans [K, K]
                tr_vec = [chain_array[iter, i, ch] for i in trans_idx]
                push!(trans_samples, reshape(tr_vec, K, K))
            end
        end
    end
    
    return beta0_samples, beta_samples, trans_samples, gamma0_samples, gamma_samples, init_samples
end

## Decode all subjects for IO-HMM
function decode(model::String, data::ExperimentData, chain, K::Int)
    N = data.n_subjects
    T_len = data.n_periods
    D = 3
    
    # Extract parameter samples
    beta0_s, beta_s, trans_s, gamma0_s, gamma_s, init_s = extract_params(model, chain, K)
    
    total_samples = length(beta0_s)
    sample_indices = range(1, total_samples)
    all_states = Array{Int}(undef, N, T_len, length(sample_indices))
    
    println("Decoding IO-HMM for $N subjects using $(length(sample_indices)) posterior samples...")
    println("Total MCMC samples available: $total_samples")
    
    for (idx, s_idx) in enumerate(sample_indices)
        if idx % 10 == 0
            print("\rProgress: $idx/$(length(sample_indices))")
        end
        
        if model == "glmhmm"
            for i in 1:N
                all_states[i, :, idx] = viterbi_glmhmm(
                    data, i,
                    beta0_s[s_idx],
                    beta_s[s_idx],
                    trans_s[s_idx],
                    init_s[s_idx]
                )
            end
        elseif model == "iohmm_mealy"
            for i in 1:N
                all_states[i, :, idx] = viterbi_iohmm_mealy(
                    data, i,
                    beta0_s[s_idx],
                    beta_s[s_idx],
                    gamma0_s[s_idx],
                    gamma_s[s_idx],
                    init_s[s_idx]
                )
            end
        elseif model == "iohmm_moore"
            for i in 1:N
                all_states[i, :, idx] = viterbi_iohmm_moore(
                    data, i,
                    beta0_s[s_idx],
                    gamma0_s[s_idx],
                    gamma_s[s_idx],
                    init_s[s_idx]
                )
            end
        end
    end
    println("\nDecoding completed!")
    
    return all_states
end

# Plot transition dynamics and state composition
function plot_transition(all_states; title = "Transition Dynamics", figsize = (2000, 800))
    
    # compute MAP
    N, T_len, _ = size(all_states)
    states_mat = Matrix{Int}(undef, N, T_len)
    for i in 1:N
        for t in 1:T_len
            states_mat[i, t] = mode(view(all_states, i, t, :))
        end
    end

    
    count_1to2 = zeros(Int, T_len - 1)
    count_2to1 = zeros(Int, T_len - 1)
    ratio_state1 = zeros(Float64, T_len)

    # State 1 ratio calculation
    for t in 1:T_len
        current_states = view(states_mat, :, t)
        n_state1 = count(x -> x == 1, current_states)
        ratio_state1[t] = n_state1 / N
    end

    # Count transitions (t=1...T-1)
    for t in 1:(T_len - 1)
        for i in 1:N
            s_curr = states_mat[i, t]
            s_next = states_mat[i, t+1]
            
            if s_curr == 1 && s_next == 2
                count_1to2[t] += 1
            elseif s_curr == 2 && s_next == 1
                count_2to1[t] += 1
            end
        end
    end

    # plotting
    start_period = 2
    periods_trans = start_period:(start_period + T_len - 2)
    periods_full = start_period:(start_period + T_len - 1)

    x_min = minimum(periods_full)
    x_max = maximum(periods_full)
    tick_values = unique(sort([x_min; collect(5:5:x_max)]))


    p = plot(size = figsize, 
             title = title, 
             titlefontsize = 12,
             xlabel = "Round",
             legend = :topleft,
             ylims = (0,18),
             grid = false,
             margin = 10Plots.mm)


    bar_w = 0.3
    offset = 0.35
    bar!(p, periods_trans .- offset, count_1to2, 
         label = "Transition: 1 -> 2", 
         color = :skyblue, 
         alpha = 0.7,
         ylabel = "Number of Transitions (Count)",
         bar_width = bar_w,
         xticks = tick_values)

    bar!(p, periods_trans .+ offset, count_2to1, 
         label = "Transition: 2 -> 1", 
         color = :orange, 
         alpha = 0.6,
         bar_width = bar_w)


    p_twin = twinx(p)    
    plot!(p_twin, periods_full, ratio_state1,
          label = "Ratio of State 1",
          color = :red,
          linewidth = 3,
          linestyle = :solid,
          ylabel = "Ratio of State 1",
          ylims = (0.0, 1.05),
          legend = :topright,
          grid = false)

    return p
end