## GLM-HMM (covariate-independent transitions)
@model function glmhmm(data::ExperimentData, K::Int; track::Bool = false)
    T_len   = data.n_periods
    N       = data.n_subjects
    D       = data.d_cov

    # =====================================================
    # Priors 
    mean_beta = zeros(K)

    beta0 ~ MvNormal(mean_beta, 2.25 * I)
    beta1 ~ MvNormal(mean_beta, 2.25 * I)
    beta2 ~ MvNormal(mean_beta, 2.25 * I)
    beta3 ~ MvNormal(mean_beta, 2.25 * I)
    trans ~ filldist(Dirichlet(K, 1.0), K)
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
        x_cc = [0.0, 0.0, 0.0]
        x_cd = [0.0, 1.0, 0.0]
        x_dc = [1.0, 0.0, 0.0]
        x_dd = [1.0, 1.0, 1.0]
        
        logit_cc = beta0 .+ beta * x_cc
        logit_cd = beta0 .+ beta * x_cd
        logit_dc = beta0 .+ beta * x_dc
        logit_dd = beta0 .+ beta * x_dd

        pcc = logistic.(logit_cc)
        pcd = logistic.(logit_cd)
        pdc = logistic.(logit_dc)
        pdd = logistic.(logit_dd)
        
        return (;
            pcc, pcd, pdc, pdd
        )
    else
        return (; log_lik)
    end
end

@model function glmhmm_permutation(data::ExperimentData, K::Int; track::Bool = false)
    T_len   = data.n_periods
    N       = data.n_subjects
    D       = data.d_cov

    # =====================================================
    # Priors 
    mean_beta = zeros(K)
    param_init = ones(K)

    beta0_raw ~ MvNormal(mean_beta, 2.25 * I)
    beta1_raw ~ MvNormal(mean_beta, 2.25 * I)
    beta2_raw ~ MvNormal(mean_beta, 2.25 * I)
    beta3_raw ~ MvNormal(mean_beta, 2.25 * I)
    trans_raw ~ filldist(Dirichlet(K, 1.0), K)
    init_raw ~ Dirichlet(param_init)

    perm = rand() < 0.5 ? [1, 2] : [2, 1]
    beta0 = beta0_raw[perm]
    beta1 = beta1_raw[perm]
    beta2 = beta2_raw[perm]
    beta3 = beta3_raw[perm]
    trans = trans_raw[perm, perm]
    init = init_raw[perm]


    beta  = hcat(beta1, beta2, beta3)
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
        
        return (;
            beta0,
            beta1,
            beta2,
            beta3,
            trans,
            init

        )
    else
        return (; log_lik)
    end
end

@model function glmhmm_init(data::ExperimentData, K::Int; track::Bool = false)
    T_len   = data.n_periods
    N       = data.n_subjects
    D       = data.d_cov

    # =====================================================
    # Priors 
    mean_beta = zeros(K)
    param_init = ones(K)

    beta0 ~ MvNormal(mean_beta, 2.25 * I)
    beta1 ~ MvNormal(mean_beta, 2.25 * I)
    beta2 ~ MvNormal(mean_beta, 2.25 * I)
    beta3 ~ MvNormal(mean_beta, 2.25 * I)
    trans ~ filldist(Dirichlet(K, 1.0), K)
    init ~ Dirichlet(param_init)

    beta  = hcat(beta1, beta2, beta3)
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
        
        return (;
            beta0,
            beta1,
            beta2,
            beta3,
            trans,
            init

        )
    else
        return (; log_lik)
    end
end

@model function glmhmm_pers(data::ExperimentData, K::Int; track::Bool = false)
    T_len   = data.n_periods
    N       = data.n_subjects
    D       = data.d_cov

    # =====================================================
    # Priors (Raw Parameters)
    # ここでは識別制約をかけずに自由にサンプリングさせます
    # =====================================================
    mean_beta = zeros(K)
    param_init = ones(K)

    # 変数名を _raw に変更（ソート前の生データ）
    beta0_raw ~ MvNormal(mean_beta, 2.25 * I)
    beta1_raw ~ MvNormal(mean_beta, 2.25 * I)
    beta2_raw ~ MvNormal(mean_beta, 2.25 * I)
    beta3_raw ~ MvNormal(mean_beta, 2.25 * I)
    
    # 遷移確率行列 (各列がDirichlet分布)
    trans_raw ~ filldist(Dirichlet(K, 1.0), K)
    init_raw  ~ Dirichlet(param_init)

    # =====================================================
    # Sorting / Permutation (識別制約の適用)
    # === 変更点: 持続確率に基づいて並べ替え ===
    # =====================================================
    
    # 1. 各状態の持続確率（対角成分: 自分自身への遷移確率）を抽出
    #    trans_raw[k, k] が大きいほど、その状態に留まりやすい
    persistence_raw = [trans_raw[k, k] for k in 1:K]

    # 2. 持続確率が「小さい順」になるようなインデックスを取得
    #    State 1 = 持続確率が低い (Transient)
    #    State 2 = 持続確率が高い (Sticky)
    perm = sortperm(persistence_raw)

    # 3. 全パラメータをこの順序に従って一括並べ替え
    beta0 = beta0_raw[perm]
    beta1 = beta1_raw[perm]
    beta2 = beta2_raw[perm]
    beta3 = beta3_raw[perm]
    
    # 【重要】遷移確率は行と列の両方を並べ替える
    trans = trans_raw[perm, perm]
    
    init  = init_raw[perm]

    # 並べ替え後のパラメータを結合して計算に使用
    beta = hcat(beta1, beta2, beta3)
    log_trans = log.(trans)
    log_init  = log.(init)

    # =====================================================
    # Likelihood (以下は元のコードと同じ)
    # =====================================================
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
    # ソート済みのパラメータを返す
    # =====================================================
    if track
        return (;
            beta0,
            beta1,
            beta2,
            beta3,
            trans,
            init
        )
    else
        return (; log_lik)
    end
end


function rerabeling(chain, K::Int)
    n_iter, n_params, n_chains = size(chain)
    df = DataFrame(chain)
    n_rows = nrow(df)

    beta0_cols = [Symbol("beta0_raw[$k]") for k in 1:K]
    beta1_cols = [Symbol("beta1_raw[$k]") for k in 1:K]
    beta2_cols = [Symbol("beta2_raw[$k]") for k in 1:K]
    beta3_cols = [Symbol("beta3_raw[$k]") for k in 1:K]
    init_cols  = [Symbol("init_raw[$k]") for k in 1:K]
    trans_cols = [Symbol("trans_raw[$i, $j]") for i in 1:K for j in 1:K]
    param_names = vcat(beta0_cols, beta1_cols, beta2_cols, beta3_cols, trans_cols, init_cols)
    n_params = length(param_names)
    df = df[:, param_names]

    box = 

    for row in 1:n_rows
        beta0_vals = [df[row, col] for col in beta0_cols]
        beta1_vals = [df[row, col] for col in beta1_cols]
        beta2_vals = [df[row, col] for col in beta2_cols]
        beta3_vals = [df[row, col] for col in beta3_cols]
        init_vals  = [df[row, col] for col in init_cols]
        trans_mat = zeros(K, K)
        for i in 1:K, j in 1:K
            col = Symbol("trans_raw[$i, $j]")
            trans_mat[i, j] = df[row, col]
        end

        perm = sortperm(beta1_vals)
        
        # beta0を並び替え
        for (k, col) in enumerate(beta0_cols)
            df[row, col] = beta0_vals[perm[k]]
        end

        # beta1を並び替え
        for (k, col) in enumerate(beta1_cols)
            df[row, col] = beta1_vals[perm[k]]
        end

        # beta2を並び替え
        for (k, col) in enumerate(beta2_cols)
            df[row, col] = beta2_vals[perm[k]]
        end

        # beta3を並び替え
        for (k, col) in enumerate(beta3_cols)
            df[row, col] = beta3_vals[perm[k]]
        end

        # initを並び替え
        for (k, col) in enumerate(init_cols)
            df[row, col] = init_vals[perm[k]]
        end

        # transを並び替え
        trans_mat_new = trans_mat[perm, perm]
        for i in 1:K, j in 1:K
            col = Symbol("trans_raw[$i, $j]")
            df[row, col] = trans_mat_new[i, j]
        end


    end

    arr = Array{Float64}(undef, n_iter, n_params, n_chains)

    row = 1
    for c in 1:n_chains
        for i in 1:n_iter
            arr[i, :, c] .= collect(df[row, :])
            row += 1
        end
    end
    
    return Chains(arr, param_names)
end



@model function glmhmm_switchtest(data::ExperimentData, K::Int; track::Bool = false, use_permutation_sampler::Bool = false)
    T_len   = data.n_periods
    N       = data.n_subjects
    D       = data.d_cov

    # =====================================================
    # Priors 
    mean_beta = zeros(K)
    param_init = ones(K)
    beta0 ~ MvNormal(mean_beta, 4.0 * I)
    beta  ~ filldist(Normal(0, 2.0), D, K)

    # trans[from, to] 
    trans ~ filldist(Dirichlet(K, 1.0), K)
    log_trans = log.(trans)

    init ~ Dirichlet(param_init)
    log_init = log.(init)

    # =====================================================
    # Permutation sampler: randomly permute state labels
    if use_permutation_sampler
        # set permutation randomly
        perm = rand() < 0.5 ? [1, 2] : [2, 1]
        
        # permute parameters
        beta0_perm = beta0[perm]
        beta_perm = beta[:, perm]
        trans_perm = trans[perm, perm]
        init_perm = init[perm]
        log_trans_perm = log.(trans_perm)
        log_init_perm = log.(init_perm)
    else
        beta0_perm = beta0
        beta_perm = beta
        trans_perm = trans
        init_perm = init
        log_trans_perm = log_trans
        log_init_perm = log_init
    end

    # =====================================================
    # Likelihood
    Tp = eltype(beta0_perm)
    log_lik = Vector{Tp}(undef, N)

    log_alpha      = Vector{Tp}(undef, K)
    log_alpha_next = Vector{Tp}(undef, K)
    logits         = Vector{Tp}(undef, K)

    for i in 1:N
        y_seq = data.observations[i]
        X_seq = data.covariates[i]

        # t = 1: 
        mul!(logits, beta_perm', view(X_seq, 1, :))
        @inbounds for k in 1:K
            logits[k] += beta0_perm[k]
            le = y_seq[1] == 1 ?
                -log1pexp(-logits[k]) :
                -log1pexp( logits[k])
            log_alpha[k] = log_init_perm[k] + le
        end

        # t = 2,...,T
        for t in 2:T_len
            mul!(logits, beta_perm', view(X_seq, t, :))
            @simd for k in 1:K
                logits[k] += beta0_perm[k]
            end
            
            @inbounds for k in 1:K
                lp = manual_logsumexp(log_trans_perm, k, log_alpha, K)
                
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
        
        logit_cc = beta0_perm .+ beta_perm' * x_cc
        logit_cd = beta0_perm .+ beta_perm' * x_cd
        logit_dc = beta0_perm .+ beta_perm' * x_dc
        logit_dd = beta0_perm .+ beta_perm' * x_dd

        pcc = logistic.(logit_cc)
        pcd = logistic.(logit_cd)
        pdc = logistic.(logit_dc)
        pdd = logistic.(logit_dd)
        
        return (;
            pcc = pcc,
            pcd = pcd,
            pdc = pdc,
            pdd = pdd
        )
    else
        return (; log_lik)
    end
end

function relabel_chain_posthoc(chain, K::Int)
    df = DataFrame(chain)
    
    beta0_cols = [Symbol("beta0[$k]") for k in 1:K]
    
    # betaの列を動的に検出
    all_cols = names(df)
    beta_pattern = r"beta\[(\d+),(\d+)\]"
    beta_cols = filter(col -> occursin(beta_pattern, String(col)), all_cols)
    
    # transとinitの列を検出
    trans_cols = [Symbol("trans[$i,$j]") for i in 1:K for j in 1:K 
                  if Symbol("trans[$i,$j]") in all_cols]
    init_cols = [Symbol("init[$k]") for k in 1:K if Symbol("init[$k]") in all_cols]
    
    for row in 1:nrow(df)
        beta0_vals = [df[row, col] for col in beta0_cols]
        perm = sortperm(beta0_vals)
        
        # beta0を並び替え
        for (k, col) in enumerate(beta0_cols)
            df[row, col] = beta0_vals[perm[k]]
        end
        
        # betaを並び替え
        if !isempty(beta_cols)
            # 各共変量ごとに処理
            D_max = maximum([parse(Int, match(beta_pattern, String(col)).captures[1]) 
                           for col in beta_cols])
            
            for d in 1:D_max
                beta_d_cols = [Symbol("beta[$d,$k]") for k in 1:K 
                              if Symbol("beta[$d,$k]") in all_cols]
                if !isempty(beta_d_cols)
                    beta_d_vals = [df[row, col] for col in beta_d_cols]
                    for (k, col) in enumerate(beta_d_cols)
                        df[row, col] = beta_d_vals[perm[k]]
                    end
                end
            end
        end
        
        # transを並び替え
        if !isempty(trans_cols)
            trans_mat = zeros(K, K)
            for i in 1:K, j in 1:K
                col = Symbol("trans[$i,$j]")
                if col in all_cols
                    trans_mat[i, j] = df[row, col]
                end
            end
            trans_mat_new = trans_mat[perm, perm]
            for i in 1:K, j in 1:K
                col = Symbol("trans[$i,$j]")
                if col in all_cols
                    df[row, col] = trans_mat_new[i, j]
                end
            end
        end
        
        # initを並び替え
        if !isempty(init_cols)
            init_vals = [df[row, col] for col in init_cols]
            for (k, col) in enumerate(init_cols)
                df[row, col] = init_vals[perm[k]]
            end
        end
    end
    
    return df
end
# ============================================================================
# 第1段階：探索用モデル（順列サンプラー使用）
# ============================================================================

@model function glmhmm_exploratory(data::ExperimentData, K::Int)
    """
    識別制約なしで順列サンプラーを使った探索的推定。
    どのパラメータがスイッチするか、またパラメータ間の相関を見るために使用。
    """
    T_len   = data.n_periods
    N       = data.n_subjects
    D       = data.d_cov

    # Priors（制約なし）
    beta0 ~ MvNormal(zeros(K), 4.0 * I)
    beta  ~ filldist(Normal(0, 2.0), D, K)
    trans ~ filldist(Dirichlet(K, 1.0), K)
    init  ~ Dirichlet(ones(K))

    # 順列サンプラー：ランダムにラベルスイッチ
    perm = rand() < 0.5 ? [1, 2] : [2, 1]  # K=2の場合
    
    beta0_perm = beta0[perm]
    beta_perm = beta[:, perm]
    trans_perm = trans[perm, perm]
    init_perm = init[perm]
    
    log_trans = log.(trans_perm)
    log_init = log.(init_perm)

    # Likelihood
    Tp = eltype(beta0_perm)
    log_lik = Vector{Tp}(undef, N)
    log_alpha = Vector{Tp}(undef, K)
    log_alpha_next = Vector{Tp}(undef, K)
    logits = Vector{Tp}(undef, K)

    for i in 1:N
        y_seq = data.observations[i]
        X_seq = data.covariates[i]
        
        mul!(logits, beta_perm', view(X_seq, 1, :))
        @inbounds for k in 1:K
            logits[k] += beta0_perm[k]
            le = y_seq[1] == 1 ? -log1pexp(-logits[k]) : -log1pexp(logits[k])
            log_alpha[k] = log_init[k] + le
        end

        for t in 2:T_len
            mul!(logits, beta_perm', view(X_seq, t, :))
            @simd for k in 1:K
                logits[k] += beta0_perm[k]
            end
            
            @inbounds for k in 1:K
                lp = manual_logsumexp(log_trans, k, log_alpha, K)
                le = y_seq[t] == 1 ? -log1pexp(-logits[k]) : -log1pexp(logits[k])
                log_alpha_next[k] = lp + le
            end
            log_alpha, log_alpha_next = log_alpha_next, log_alpha
        end

        ll = logsumexp(log_alpha)
        log_lik[i] = ll
        Turing.@addlogprob! ll
    end
    
    return (; log_lik)
end

# ============================================================================
# 探索的分析のための可視化関数
# ============================================================================

"""
    analyze_label_switching(chain, K; save_path=nothing)

順列サンプラーの結果から、どのパラメータがスイッチし、
どのパラメータ間に相関があるかを可視化する。

論文の図1-4に相当する分析を行う。

# 判定基準
1. **スイッチの有無**: (param[1], param[2])の散布図が45度線に関して2つの塊に分かれる
2. **同一状態の判定**: (paramA[1] vs paramB[1])の散布図が2つの塊に分かれ、
   かつparamA大きい↔paramB大きい の相関があれば同じ状態

# Returns
- summary: パラメータごとのスイッチ判定と推奨される制約
"""
function analyze_label_switching(chain, K::Int; save_path=nothing)
    df = DataFrame(chain)
    
    # パラメータ名を抽出
    beta0_cols = [Symbol("beta0[$k]") for k in 1:K]
    
    # betaの列を動的に検出
    all_cols = names(df)
    beta_pattern = r"beta\[(\d+),(\d+)\]"
    beta_cols = filter(col -> occursin(beta_pattern, String(col)), all_cols)
    
    # D（共変量数）を推定
    D = 0
    if !isempty(beta_cols)
        D = maximum([parse(Int, match(beta_pattern, String(col)).captures[1]) 
                    for col in beta_cols])
    end
    
    trans_diag_cols = [Symbol("trans[$k,$k]") for k in 1:K if Symbol("trans[$k,$k]") in all_cols]
    init_cols = [Symbol("init[$k]") for k in 1:K if Symbol("init[$k]") in all_cols]
    
    println("="^70)
    println("ラベルスイッチング分析")
    println("="^70)
    
    # ============================================================
    # 1. 各パラメータのスイッチ判定
    # ============================================================
    results = Dict{String, Any}()
    
    # beta0の分析
    println("\n【beta0の分析】")
    beta0_switching = analyze_parameter_switching(df, beta0_cols, "beta0")
    results["beta0"] = beta0_switching
    
    # 各共変量のbetaの分析
    if D > 0
        println("\n【betaの分析（共変量ごと）】")
        for d in 1:D
            beta_d_cols = [Symbol("beta[$d,$k]") for k in 1:K]
            beta_d_switching = analyze_parameter_switching(df, beta_d_cols, "beta[$d,:]")
            results["beta_$d"] = beta_d_switching
        end
    end
    
    # transの分析（対角要素のみ）
    if !isempty(trans_diag_cols)
        println("\n【trans対角要素の分析】")
        trans_switching = analyze_parameter_switching(df, trans_diag_cols, "trans_diag")
        results["trans"] = trans_switching
    end
    
    # initの分析
    if !isempty(init_cols)
        println("\n【initの分析】")
        init_switching = analyze_parameter_switching(df, init_cols, "init")
        results["init"] = init_switching
    end
    
    # ============================================================
    # 2. パラメータ間の相関分析（同じ状態かどうか）
    # ============================================================
    println("\n" * "="^70)
    println("パラメータ間の相関分析")
    println("="^70)
    
    # スイッチするパラメータのリスト
    switching_params = []
    if results["beta0"][:has_switching]
        push!(switching_params, ("beta0", beta0_cols))
    end
    for d in 1:D
        if haskey(results, "beta_$d") && results["beta_$d"][:has_switching]
            beta_d_cols = [Symbol("beta[$d,$k]") for k in 1:K]
            push!(switching_params, ("beta[$d,:]", beta_d_cols))
        end
    end
    
    # ペアワイズ相関を分析
    correlations = analyze_parameter_correlations(df, switching_params, K)
    results["correlations"] = correlations
    
    # ============================================================
    # 3. 推奨される識別制約の出力
    # ============================================================
    println("\n" * "="^70)
    println("推奨される識別制約")
    println("="^70)
    
    constraints = generate_constraint_recommendations(results, correlations, K, D)
    results["recommended_constraints"] = constraints
    
    for (i, constraint_group) in enumerate(constraints)
        println("\n制約グループ $i:")
        println("  パラメータ: ", join(constraint_group[:params], ", "))
        println("  制約: ", constraint_group[:constraint])
        println("  理由: ", constraint_group[:reason])
    end
    
    # ============================================================
    # 4. 可視化
    # ============================================================
    println("\n" * "="^70)
    println("散布図を生成中...")
    println("="^70)
    
    plots = create_diagnostic_plots(df, results, switching_params, K, D)
    
    if !isnothing(save_path)
        savefig(plots, save_path)
        println("図を保存: $save_path")
    end
    
    display(plots)
    
    return results
end

"""
個別パラメータのスイッチング判定
"""
function analyze_parameter_switching(df, param_cols, param_name)
    if length(param_cols) < 2
        return Dict(:has_switching => false, :confidence => 0.0)
    end
    
    # (param[1], param[2])の散布図データ
    x = df[!, param_cols[1]]
    y = df[!, param_cols[2]]
    
    # 45度線からの距離を計算
    distance_from_diagonal = abs.(x .- y) ./ sqrt(2)
    mean_distance = mean(distance_from_diagonal)
    
    # 2つのクラスタへの分離度を評価
    # 簡易的な指標：45度線からの距離の標準偏差が大きいほど分離
    separation_score = std(distance_from_diagonal) / (mean(abs.(x)) + mean(abs.(y)) + 1e-10)
    
    # 判定（閾値は調整可能）
    has_switching = separation_score > 0.15  # 経験的閾値
    
    println("  $param_name:")
    println("    平均距離: $(round(mean_distance, digits=4))")
    println("    分離スコア: $(round(separation_score, digits=4))")
    println("    スイッチあり: $(has_switching ? "はい ✓" : "いいえ ✗")")
    
    return Dict(
        :has_switching => has_switching,
        :separation_score => separation_score,
        :mean_distance => mean_distance
    )
end

"""
パラメータ間の相関分析（同じ状態を共有するか判定）
"""
function analyze_parameter_correlations(df, switching_params, K)
    correlations = Dict()
    
    if length(switching_params) < 2
        println("  相関分析: スイッチするパラメータが2つ未満のためスキップ")
        return correlations
    end
    
    # 全ペアを分析
    for i in 1:(length(switching_params)-1)
        for j in (i+1):length(switching_params)
            name_i, cols_i = switching_params[i]
            name_j, cols_j = switching_params[j]
            
            # 状態1同士の相関
            corr_state1 = cor(df[!, cols_i[1]], df[!, cols_j[1]])
            # 状態2同士の相関
            corr_state2 = cor(df[!, cols_i[2]], df[!, cols_j[2]])
            
            # 平均相関（絶対値）
            avg_corr = (abs(corr_state1) + abs(corr_state2)) / 2
            
            # 同じ符号の相関か（正の相関なら同じ方向にスイッチ）
            same_direction = sign(corr_state1) == sign(corr_state2)
            
            # 強い正の相関 → 同じ状態を共有
            share_state = avg_corr > 0.7 && same_direction && corr_state1 > 0
            
            correlations["$(name_i)_vs_$(name_j)"] = Dict(
                :corr_state1 => corr_state1,
                :corr_state2 => corr_state2,
                :avg_corr => avg_corr,
                :share_state => share_state
            )
            
            println("\n  $name_i vs $name_j:")
            println("    状態1の相関: $(round(corr_state1, digits=3))")
            println("    状態2の相関: $(round(corr_state2, digits=3))")
            println("    同じ状態を共有: $(share_state ? "はい ✓" : "いいえ ✗")")
        end
    end
    
    return correlations
end

"""
識別制約の推奨事項を生成
"""
function generate_constraint_recommendations(results, correlations, K, D)
    constraints = []
    
    # まずbeta0
    if results["beta0"][:has_switching]
        # beta0と同じ状態を共有するパラメータを探す
        group_params = ["beta0"]
        
        for d in 1:D
            key = "beta0_vs_beta[$d,:]"
            if haskey(correlations, key) && correlations[key][:share_state]
                push!(group_params, "beta[$d,:]")
            end
        end
        
        push!(constraints, Dict(
            :params => group_params,
            :constraint => "昇順ソート（state1 < state2）",
            :reason => "これらのパラメータは同じ状態を共有（強い正の相関）"
        ))
    end
    
    # beta0と相関しないがスイッチするパラメータ
    for d in 1:D
        if haskey(results, "beta_$d") && results["beta_$d"][:has_switching]
            key = "beta0_vs_beta[$d,:]"
            if !haskey(correlations, key) || !correlations[key][:share_state]
                # 独立した制約グループ
                push!(constraints, Dict(
                    :params => ["beta[$d,:]"],
                    :constraint => "独自の順序制約",
                    :reason => "beta0と独立にスイッチ"
                ))
            end
        end
    end
    
    return constraints
end

"""
診断用の散布図を生成（論文の図1-4のような図）
"""
function create_diagnostic_plots(df, results, switching_params, K, D)
    # プロット数を決定
    n_plots = 0
    
    # beta0 vs beta0
    if results["beta0"][:has_switching]
        n_plots += 1
    end
    
    # beta0 vs 各beta
    for d in 1:D
        if haskey(results, "beta_$d") && results["beta_$d"][:has_switching]
            n_plots += 1
        end
    end
    
    # beta間の相関（上位3つ）
    n_plots += min(3, length(switching_params) * (length(switching_params) - 1) ÷ 2)
    
    # グリッドレイアウト
    n_cols = 3
    n_rows = ceil(Int, n_plots / n_cols)
    
    plots_array = []
    
    # 1. beta0 vs beta0 (論文の(μ,μ)に相当)
    if results["beta0"][:has_switching]
        beta0_cols = [Symbol("beta0[$k]") for k in 1:K]
        p = scatter(
            df[!, beta0_cols[1]], 
            df[!, beta0_cols[2]],
            xlabel="beta0[1]",
            ylabel="beta0[2]",
            title="beta0: スイッチあり",
            legend=false,
            alpha=0.3,
            markersize=2
        )
        # 45度線を追加
        xlims = Plots.xlims(p)
        plot!(p, xlims, xlims, line=:dash, color=:red, label="45度線")
        push!(plots_array, p)
    end
    
    # 2. beta0 vs 各beta (論文の(μ,σ)に相当)
    beta0_cols = [Symbol("beta0[$k]") for k in 1:K]
    for d in 1:D
        if haskey(results, "beta_$d") && results["beta_$d"][:has_switching]
            beta_d_cols = [Symbol("beta[$d,$k]") for k in 1:K]
            p = scatter(
                df[!, beta0_cols[1]], 
                df[!, beta_d_cols[1]],
                xlabel="beta0[1]",
                ylabel="beta[$d,1]",
                title="beta0 vs beta[$d,:]\n相関=$(round(cor(df[!, beta0_cols[1]], df[!, beta_d_cols[1]]), digits=2))",
                legend=false,
                alpha=0.3,
                markersize=2
            )
            push!(plots_array, p)
        end
    end
    
    # レイアウト
    if isempty(plots_array)
        return plot(title="スイッチするパラメータなし")
    end
    
    return plot(plots_array..., layout=(n_rows, n_cols), size=(1200, 400*n_rows))
end

# ============================================================================
# 使用例
# ============================================================================