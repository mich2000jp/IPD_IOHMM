# Run MCMC sampling and save results
function RunMCMC(SEED, model, K_states, sampler, n_iter, n_chains, n_burnin, output_path)
    # Run MCMC sampling
    println("=== MCMC Sampling ===")
    chain = sample(
    Xoshiro(SEED), 
    model, 
    sampler, 
    MCMCThreads(), 
    n_iter, 
    n_chains; 
    discard_initial = n_burnin,
    progress = true
    )
    println("=== MCMC Completed! ===")
    println("\nSaving results to '$output_path'...")


    try
        model_name = nameof(model.f)
        model_source = try
            method_list = methods(model.f)
            io = IOBuffer()
            for m in method_list
                println(io, "# Method: ", m)
                code_lowered(m) |> x -> println(io, x)
                println(io, "\n")
            end
            String(take!(io))
        catch e
            "Could not retrieve model source: $e"
        end

        metadata = Dict(
            "K_states" => K_states,
            "n_iter" => n_iter,
            "n_burnin" => n_burnin,
            "n_chains" => n_chains,
            "model_name" => model_name,
            "model_source" => model_source,
            "sampler" => string(sampler),
        )
            
        jldsave(output_path; chain, metadata)
        println("\nResults saved successfully.")
    catch e
        @warn "Failed to save results: $e"
    end

    return chain
end

# Convert generated quantities to Chains format
function convert_gq(gq)
    # dimensions
    n_iter, n_chains = size(gq)

    # inspect one sample
    sample = gq[1, 1]
    fields = fieldnames(typeof(sample))

    # ---------- パラメータ名生成 ----------
    param_names = String[]

    function names_for_field(fname::Symbol, val)
        if isa(val, Number)
            return [string(fname)]
        elseif isa(val, AbstractVector)
            return [string(fname, "[", i, "]") for i in eachindex(val)]
        elseif isa(val, AbstractMatrix)
            return [
                string(fname, "[", i, ",", j, "]")
                for i in axes(val, 1), j in axes(val, 2)
            ]
        else
            error("Unsupported parameter type for $fname: $(typeof(val))")
        end
    end

    for f in fields
        append!(param_names, names_for_field(f, getfield(sample, f)))
    end

    n_params = length(param_names)
    chains_3d = Array{Float64,3}(undef, n_iter, n_params, n_chains)

    # ---------- 値の収集 ----------
    function collect_values(val)
        if isa(val, Number)
            return Float64[val]
        elseif isa(val, AbstractVector)
            return Float64.(val)
        elseif isa(val, AbstractMatrix)
            return Float64.(vec(val))
        else
            error("Unsupported parameter type: $(typeof(val))")
        end
    end

    for ch in 1:n_chains
        for it in 1:n_iter
            s = gq[it, ch]
            collected = Float64[]

            for f in fields
                append!(collected, collect_values(getfield(s, f)))
            end

            chains_3d[it, :, ch] .= collected
        end
    end

    return Chains(chains_3d, Symbol.(param_names))
end

# Convert generated quantities to log-likelihood array for PSIS-LOO
function gq_to_loglik_array(gq)
    n_iter, n_chains = size(gq)
    N = length(gq[1,1].log_lik)

    loglik = Array{Float64}(undef, N, n_iter, n_chains)

    for ch in 1:n_chains
        for it in 1:n_iter
            loglik[:, it, ch] .= gq[it, ch].log_lik
        end
    end
    return loglik
end

# Run PSIS-LOOCV and report pareto k diagnostics
function RunPSISLOO(model, chain)
    gq = generated_quantities(model, chain)
    log_lik = gq_to_loglik_array(gq)
    psis_result = psis_loo(log_lik)
    
    pareto_k = psis_result.pointwise(statistic = :pareto_k) |> collect
    bad_ids = findall(>(0.7), pareto_k)
    n_bad = length(bad_ids)

    if n_bad == 0
        println("All subjects have pareto k ≤ 0.7.")
    else
        println("There are $n_bad subjects with pareto k > 0.7.")
        println("Subject IDs = ", bad_ids)
        for i in bad_ids
            println("Subject ", i, ": pareto k = ", pareto_k[i])
        end
    end
    
    return psis_result 
end

# Relabel MCMC chain to address label switching, based on 'init' parameter
function relabel_chain(chain, K::Int=2)
    if K <= 1
        return chain
    end

    param_names = names(chain)
    n_iter = size(chain, 1)
    n_chains = size(chain, 3)
    n_params = length(param_names)

    # init_means[ch, k]
    init_means = zeros(n_chains, K)
    for ch in 1:n_chains
        for k in 1:K
            init_means[ch, k] = mean(chain[:,Symbol("init[$k]"),ch])
        end
    end
    
    relabeled_array = zeros(n_iter, n_params, n_chains)

    for ch in 1:n_chains
        inv_perm = sortperm(init_means[ch, :], rev=true)        
        if inv_perm == collect(1:K)
            relabeled_array[:, :, ch] .= chain.value.data[:, :, ch]
            continue
        end

        # 各パラメータを並び替え
        for (param_idx, param_name) in enumerate(param_names)
            param_str = string(param_name)
            
            # set default
            relabeled_array[:, param_idx, ch] .= chain[:, param_idx, ch]
            
            

            # 1-d parameters: param[k] : init[k], beta0[k]
            m = match(r"^(\w+)\[(\d+)\]$", param_str)
            if !isnothing(m)
                base_name = m.captures[1]
                k = parse(Int, m.captures[2])
                
                if k <= K
                    old_k = inv_perm[k]
                    old_param_name = Symbol("$(base_name)[$old_k]")
                    old_idx = findfirst(==(old_param_name), param_names)
                    
                    if !isnothing(old_idx)
                        relabeled_array[:, param_idx, ch] .= chain[:, old_idx, ch]
                    end
                end

            end
            
            # 2-d parameters:
            m = match(r"^(\w+)\[(\d+), \s*(\d+)\]$", param_str)
            if !isnothing(m)
                base_name = m.captures[1]
                i = parse(Int, m.captures[2])
                j = parse(Int, m.captures[3])
                
                # beta[d, k]
                if base_name == "beta"
                    old_i = i
                    old_j = inv_perm[j]
                    old_param_name = Symbol("$(base_name)[$old_i, $old_j]")
                    old_idx = findfirst(==(old_param_name), param_names)
                    
                    if !isnothing(old_idx)
                        relabeled_array[:, param_idx, ch] .= chain[:, old_idx, ch]
                    end
                # trans[from, to], gamma0[from, to]
                elseif base_name == "trans"
                    old_i = inv_perm[i]
                    old_j = inv_perm[j]
                    old_param_name = Symbol("$(base_name)[$old_i, $old_j]")
                    old_idx = findfirst(==(old_param_name), param_names)
                    
                    if !isnothing(old_idx)
                        relabeled_array[:, param_idx, ch] .= chain[:, old_idx, ch]
                    end
                elseif base_name == "gamma0"
                    old_i = i
                    old_j = inv_perm[j]
                    old_param_name = Symbol("$(base_name)[$old_i, $old_j]")
                    old_idx = findfirst(==(old_param_name), param_names)
                    
                    if !isnothing(old_idx)
                        relabeled_array[:, param_idx, ch] .= chain[:, old_idx, ch]
                    end
                end
            end

            # 3-d parameters: gamma[d, from, to]
            m = match(r"^(\w+)\[(\d+), \s*(\d+), \s*(\d+)\]$", param_str)
            if !isnothing(m)
                base_name = m.captures[1]
                d = parse(Int, m.captures[2])
                i = parse(Int, m.captures[3])
                j = parse(Int, m.captures[4])
                
                if base_name == "gamma" 
                    old_d = d
                    old_i = i
                    old_j = inv_perm[j]
                    old_param_name = Symbol("$(base_name)[$old_d, $old_i, $old_j]")
                    old_idx = findfirst(==(old_param_name), param_names)
                    
                    if !isnothing(old_idx)
                        relabeled_array[:, param_idx, ch] .= chain[:, old_idx, ch]
                    end
                end
            end
        end
    end
    new_chain = Chains(
        relabeled_array,
        param_names,
        (internals = chain.name_map[:internals],
         parameters = chain.name_map[:parameters]);
    )
    return new_chain
end


# Post MCMC analysis: summary, PSIS-LOO, plots
function RunPostAnalysis(model_gq, chain::Chains, K_states, POST_PATH)
    SUMMARY_PATH = POST_PATH[1]
    LOO_PATH     = POST_PATH[2]
    PLOT_PATH     = POST_PATH[3]
    PLOT_GQ_PATH  = POST_PATH[4]

    chain_relabeled = chain
    println("genarating quantities...")
    gq = generated_quantities(model_gq, chain_relabeled)
    chain_gq = convert_gq(gq)

    println("Summarizing MCMC Results...")
    df_summary = DataFrame(summarystats(chain_relabeled))
    df_summary_gq = DataFrame(summarystats(chain_gq))
    df_hpd = DataFrame(MCMCChains.hpd(chain_relabeled, alpha=0.05))
    df_hpd_gq = DataFrame(MCMCChains.hpd(chain_gq, alpha=0.05))
    df = leftjoin(df_summary, df_hpd, on = :parameters)
    df_gq = leftjoin(df_summary_gq, df_hpd_gq, on = :parameters)
    df_stacked = vcat(df, df_gq)
    display(df_stacked)
    CSV.write(SUMMARY_PATH, df_stacked)

    println("Plotting MCMC Results...")
    p1 = plot(chain_relabeled)
    p2 = plot(chain_gq)
    savefig(p1, PLOT_PATH)
    savefig(p2, PLOT_GQ_PATH)

    println("PSIS-LOO Calculation...")
    loo = RunPSISLOO(model, chain_relabeled)
    df_loo =DataFrame(loo.estimates)
    df_loo = unstack(df_loo,
    :statistic,
    :column,
    :value
    )
    CSV.write(LOO_PATH, df_loo)
    display(loo)

    println("Post Analysis Completed!")
end

# Check and print parameter names in the chain
function check_parameter_names(chain::Chains)
    param_names = names(chain)
    
    println("Total parameters: ", length(param_names))
    println("\nParameter names:")
    for name in param_names
        println("  ", name)
    end
    
    # パターン別にグループ化
    beta0_params = filter(n -> occursin("beta0", string(n)), param_names)
    beta_params = filter(n -> occursin(r"^beta\[", string(n)), param_names)
    a_raw_params = filter(n -> occursin("a_raw", string(n)), param_names)
    gamma_params = filter(n -> occursin("gamma", string(n)), param_names)
    init_params = filter(n -> occursin("init", string(n)), param_names)
    
    println("\nbeta0 parameters (", length(beta0_params), "): ", beta0_params)
    println("beta parameters (", length(beta_params), "): ", beta_params[1:min(5, length(beta_params))])
    println("a_raw parameters (", length(a_raw_params), "): ", a_raw_params)
    println("gamma parameters (", length(gamma_params), "): ", gamma_params)
    println("init parameters (", length(init_params), "): ", init_params)
end

# Plot transition dynamics and state composition
function plot_transition(all_states; title = "Transition Dynamics", figsize = (2000, 800))
    
    # -------------------------------------------------------
    # 1. データの前処理: 次元の確認と代表値(mode)の計算
    # -------------------------------------------------------
    if ndims(all_states) == 3
        N, T_len, n_samples = size(all_states)
        # 各被験者・各時点での最頻値（MAP推定値）を計算
        states_mat = Matrix{Int}(undef, N, T_len)
        for i in 1:N
            for t in 1:T_len
                # viewを使ってメモリ効率よく最頻値を計算
                states_mat[i, t] = mode(view(all_states, i, t, :))
            end
        end
    else
        # 既に (N, T) の形式の場合
        N, T_len = size(all_states)
        states_mat = all_states
    end

    # -------------------------------------------------------
    # 2. 集計処理
    # -------------------------------------------------------
    # 期間 t (1 ~ T-1) における遷移数をカウント
    # count_1to2[t]: 時点 t で State 1 だった人が、t+1 で State 2 になった数
    count_1to2 = zeros(Int, T_len - 1)
    count_2to1 = zeros(Int, T_len - 1)
    
    # 各時点 t における State 1 の割合
    ratio_state1 = zeros(Float64, T_len)

    # State 1 の割合を計算 (t=1...T)
    for t in 1:T_len
        # State 1 (または0) の定義に依存するが、ここではラベルが 1, 2 と仮定
        # もし 0, 1 なら適宜読み替えるが、通常HMMのラベルは 1, 2... K
        # 最小値を確認して調整
        current_states = view(states_mat, :, t)
        
        # State 1 の人数をカウント (ラベルが1の場合)
        # ※ viterbiの出力が1-based index (1, 2) であることを想定
        n_state1 = count(x -> x == 1, current_states)
        ratio_state1[t] = n_state1 / N
    end

    # 遷移数を計算 (t=1...T-1)
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

    # -------------------------------------------------------
    # 3. プロット作成 (2軸グラフ)
    # -------------------------------------------------------
    # 時間軸の調整
    # データがピリオド2から始まるとのことなので、全体を+1シフトして表示
    start_period = 2
    
    # 遷移データ用のX軸 (長さ: T_len - 1)
    # データ上の t=1 -> t=2 の遷移を、プロット上では start_period -> start_period+1 として扱う
    periods_trans = start_period:(start_period + T_len - 2)
    
    # 状態比率用のX軸 (長さ: T_len)
    periods_full = start_period:(start_period + T_len - 1)

    # 横軸ラベル(xticks)の設定: 2, 5, 10, 15... の形式にする
    # 範囲の最小値から最大値まで
    x_min = minimum(periods_full)
    x_max = maximum(periods_full)
    
    # 開始点(2)と、5刻みの値(5, 10, 15...)を結合してソート・ユニーク化
    # これにより [2, 5, 10, 15, ..., 100] のような並びになります
    tick_values = unique(sort([x_min; collect(5:5:x_max)]))

    # --- 左軸: 遷移数 (Bar Chart) ---
    p = plot(size = figsize, 
             title = title, 
             titlefontsize = 12,
             xlabel = "Round",
             legend = :topleft,
             ylims = (0,18),
             grid = false,
             margin = 10Plots.mm)

    # バーの位置調整用設定
    # X軸上で重ならないように、それぞれのバーを左右に少しずらします。
    bar_w = 0.3        # バーの幅
    offset = 0.35      # 中心からのずらし幅

    # State 1 -> 2 の遷移 (左にずらす: periods_trans .- offset)
    bar!(p, periods_trans .- offset, count_1to2, 
         label = "Transition: 1 -> 2", 
         color = :skyblue, 
         alpha = 0.7,
         ylabel = "Number of Transitions (Count)",
         bar_width = bar_w,
         xticks = tick_values) # 指定したラベルを適用

    # State 2 -> 1 の遷移 (右にずらす: periods_trans .+ offset)
    bar!(p, periods_trans .+ offset, count_2to1, 
         label = "Transition: 2 -> 1", 
         color = :orange, 
         alpha = 0.6,
         bar_width = bar_w)

    # --- 右軸: State 1 の割合 (Line Chart) ---
    # twinx() を使って右軸を作成
    p_twin = twinx(p)
    
    plot!(p_twin, periods_full, ratio_state1,
          label = "Ratio of State 1",
          color = :red,
          linewidth = 3,
          linestyle = :solid,
          ylabel = "Ratio of State 1",
          ylims = (0.0, 1.05), # 割合なので0-1
          legend = :topright,
          grid = false) # グリッドが重なると見にくいのでOFF

    return p
end

# # Plot transition dynamics and state composition (generalized for K states)
# function plot_transition2(all_states; title_prefix = "Transition Dynamics", figsize = (2000, 800))
    
#     # -------------------------------------------------------
#     # 1. データの前処理: 次元の確認と代表値(mode)の計算
#     # -------------------------------------------------------
#     if ndims(all_states) == 3
#         N, T_len, n_samples = size(all_states)
#         states_mat = Matrix{Int}(undef, N, T_len)
#         for i in 1:N
#             for t in 1:T_len
#                 states_mat[i, t] = mode(view(all_states, i, t, :))
#             end
#         end
#     else
#         N, T_len = size(all_states)
#         states_mat = all_states
#     end

#     # -------------------------------------------------------
#     # 2. 状態数の自動検出
#     # -------------------------------------------------------
#     K = maximum(states_mat)  # 状態数
    
#     # -------------------------------------------------------
#     # 3. 集計処理
#     # -------------------------------------------------------
#     # 遷移カウント: transition_counts[t, from_state, to_state]
#     transition_counts = zeros(Int, T_len - 1, K, K)
    
#     for t in 1:(T_len - 1)
#         for i in 1:N
#             s_curr = states_mat[i, t]
#             s_next = states_mat[i, t+1]
#             transition_counts[t, s_curr, s_next] += 1
#         end
#     end
    
#     # 各時点における各状態の人数
#     state_counts = zeros(Int, T_len, K)
#     for t in 1:T_len
#         for i in 1:N
#             state_counts[t, states_mat[i, t]] += 1
#         end
#     end
    
#     # 割合に変換
#     state_ratios = state_counts ./ N
    
#     # -------------------------------------------------------
#     # 4. 時間軸の設定
#     # -------------------------------------------------------
#     start_period = 2
#     periods_trans = start_period:(start_period + T_len - 2)
#     periods_full = start_period:(start_period + T_len - 1)
    
#     x_min = minimum(periods_full)
#     x_max = maximum(periods_full)
#     tick_values = unique(sort([x_min; collect(5:5:x_max)]))
    
#     # -------------------------------------------------------
#     # 5. プロット1: 遷移カウント (すべての組み合わせ)
#     # -------------------------------------------------------
#     # 異なる状態への遷移のみをカウント (i != j)
#     transitions_list = []
#     for i in 1:K
#         for j in 1:K
#             if i != j
#                 push!(transitions_list, (i, j))
#             end
#         end
#     end
    
#     n_transitions = length(transitions_list)
    
#     # バーのオフセット自動計算
#     bar_w = 0.8 / n_transitions  # 全体の幅を0.8として均等分割
#     offsets = range(-0.4, 0.4, length=n_transitions)
    
#     # 色の設定 (状態数に応じて自動生成)
#     colors = [:skyblue, :orange, :green, :purple, :pink, :brown, :gray, :yellow]
    
#     p1 = plot(size = figsize, 
#              title = "$title_prefix - Transition Counts",
#              titlefontsize = 12,
#              xlabel = "Round",
#              ylabel = "Number of Transitions",
#              legend = :topleft,
#              ylims = (0, 18),
#              grid = false,
#              margin = 10Plots.mm,
#              xticks = tick_values)
    
#     for (idx, (from_state, to_state)) in enumerate(transitions_list)
#         counts = transition_counts[:, from_state, to_state]
#         bar!(p1, periods_trans .+ offsets[idx], counts,
#              label = "Transition: $from_state → $to_state",
#              color = colors[mod(idx-1, length(colors)) + 1],
#              alpha = 0.7,
#              bar_width = bar_w)
#     end
    
#     # -------------------------------------------------------
#     # 6. プロット2: 状態割合の積み重ねバープロット
#     # -------------------------------------------------------
#     p2 = plot(size = figsize,
#              title = "$title_prefix - State Composition Over Time",
#              titlefontsize = 12,
#              xlabel = "Round",
#              ylabel = "Proportion of States",
#              legend = :topright,
#              ylims = (0, 1.05),
#              grid = false,
#              margin = 10Plots.mm,
#              xticks = tick_values)
    
#     # 積み重ねバープロット (累積和を使って手動で積み重ね)
#     cumsum_bottom = zeros(T_len)
#     for k in 1:K
#         # 現在の状態の上端 = 底辺 + 高さ
#         cumsum_top = cumsum_bottom .+ state_ratios[:, k]
        
#         bar!(p2, periods_full, cumsum_top,
#              label = "State $k",
#              color = colors[k],
#              alpha = 0.7,
#              bar_width = 0.8,
#              fillrange = cumsum_bottom,
#              linecolor = :white,
#              linewidth = 0.5)
        
#         # 次の状態の底辺を更新
#         cumsum_bottom .= cumsum_top
#     end
    
#     return p1, p2
# end
