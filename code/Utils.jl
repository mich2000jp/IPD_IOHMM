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

                # trans[from, to], gammaX[from, to]
                elseif base_name == "trans"
                    old_i = inv_perm[i]
                    old_j = inv_perm[j]
                    old_param_name = Symbol("$(base_name)[$old_i, $old_j]")
                    old_idx = findfirst(==(old_param_name), param_names)
                    
                    if !isnothing(old_idx)
                        relabeled_array[:, param_idx, ch] .= chain[:, old_idx, ch]
                    end
                elseif base_name in ["gamma0", "gamma1", "gamma2", "gamma3"]
                    old_i = i
                    old_j = inv_perm[j]
                    old_param_name = Symbol("$(base_name)[$old_i, $old_j]")
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
function RunPSISLOO(model, chain, PARATO_PATH)
    gq = generated_quantities(model, chain)
    log_lik = gq_to_loglik_array(gq)
    psis_result = psis_loo(log_lik)
    
    pareto_k = psis_result.pointwise(statistic = :pareto_k) |> collect
    bad_ids_07 = findall(>(0.7), pareto_k)
    bad_ids = findall(>(0.5), pareto_k)
    n_bad_07 = length(bad_ids_07)
    n_bad = length(bad_ids)
    n_bad_05 = n_bad - n_bad_07

    if n_bad == 0
        println("All subjects have pareto k ≤ 0.5.")
    else
        println("There are $n_bad_07 subjects with pareto k > 0.7, and $n_bad_05 subjects with 0.5 < pareto k ≤ 0.7.")
        for i in bad_ids
            println("Subject ", i, ": pareto k = ", pareto_k[i])
        end
    end
    
    p = histogram(
    pareto_k;
    bins = :auto,
    xlabel = "Pareto k",
    ylabel = "Frequency",
    title = "PSIS-LOO Pareto k diagnostics"
    )
    vline!(
        [0.5, 0.7, 1.0],
        linestyle = :dash,
    )
    savefig(p, PARATO_PATH)

    return psis_result 
end

function posterior_plot(chain, K)
    namev = names(chain, :parameters)
    plots = []
    for i in 1:length(namev)
        p1 = plot(chain[:, namev[i],:],title=namev[i],legend=false, yticks=false)
        p2 = plot(chain[:, namev[i],:], seriestype=:density, title=namev[i],legend=false, yticks=false)
        p = plot(p1, p2, layout=(1,2), margin=1Plots.mm)
        push!(plots, p)
    end
    p = plot(plots..., layout=(length(plots) ÷ K, K), size=(1000*K, 150*length(plots)))
    return p
end

# Post MCMC analysis: summary, PSIS-LOO, plots
function RunPostAnalysis(model_gq, chain::Chains, K_states, OUTPUT_PATH)
    SUMMARY_PATH  = OUTPUT_PATH[1]
    LOO_PATH      = OUTPUT_PATH[2]
    PLOT_PATH     = OUTPUT_PATH[3]
    PLOT_GQ_PATH  = OUTPUT_PATH[4]
    PARATO_PATH   = OUTPUT_PATH[5]

    println("relabeling states...")
    chain_relabeled = relabel_chain(chain, K_states)

    println("generating quantities...")
    gq = generated_quantities(model_gq, chain_relabeled)

    chain_gq = convert_gq(gq)
    println("summarizing results...")
    df_summary = DataFrame(summarystats(chain_relabeled))
    df_summary_gq = DataFrame(summarystats(chain_gq))
    df_hpd = DataFrame(MCMCChains.hpd(chain_relabeled, alpha=0.05))
    df_hpd_gq = DataFrame(MCMCChains.hpd(chain_gq, alpha=0.05))
    df = leftjoin(df_summary, df_hpd, on = :parameters)
    df_gq = leftjoin(df_summary_gq, df_hpd_gq, on = :parameters)
    df_stacked = vcat(df, df_gq)
    display(df_stacked)
    CSV.write(SUMMARY_PATH, df_stacked, delim = ';')

    println("Plotting MCMC Results...")
    p1 = posterior_plot(chain_relabeled, K_states)
    p2 = posterior_plot(chain_gq, K_states)
    savefig(p1, PLOT_PATH)
    savefig(p2, PLOT_GQ_PATH)

    println("PSIS-LOO Calculation...")
    loo = RunPSISLOO(model, chain_relabeled, PARATO_PATH)
    df_loo =DataFrame(loo.estimates)
    df_loo = unstack(df_loo, :statistic, :column, :value)
    CSV.write(LOO_PATH, df_loo, delim = ';')

    println("All done!")
end

# Plot player context sequence
function plot_player_context(data::ExperimentData, player_id::Int)

    cont_seq = data.contexts[player_id]
    levels = ["CCC", "CCD", "CDC", "CDD", "DCC", "DCD", "DDC", "DDD"]
    map_context = Dict(l => i for (i, l) in enumerate(levels))
    y = [map_context[c] for c in cont_seq]
    
    T = length(y)

    p = plot(
        2:(T+1),
        y;
        seriestype = :step,
        linewidth = 2,
        legend = false,
        xlabel = "Round",
        ylabel = "Context",
        xlims = (2, T+1),
        ylims = (0.5, length(levels) + 0.5),
        yticks = (1:length(levels), string.(levels)),
        title = "Context experience of Player $player_id",
        margin = 5Plots.mm,
        size = (900, 500)
    )

    return p
end

# Plot ratio of context 
function plot_context_ratio(data::ExperimentData; title = "Context Ratio over Rounds")
    N = data.n_subjects
    T = data.n_periods

    levels = ["CCC", "CDC", "DCC", "DDC", "CCD", "CDD", "DCD", "DDD"]
    n_levels = length(levels)
    map_context = Dict(l => i for (i, l) in enumerate(levels))

    # -------------------------------------------------------
    # 1. 集計
    # -------------------------------------------------------
    context_counts = zeros(Int, n_levels, T)
    for i in 1:N
        cont_seq = data.contexts[i]
        for t in 1:T
            context_counts[map_context[cont_seq[t]], t] += 1
        end
    end

    context_ratios = context_counts ./ N   # (n_levels × T)

    # -------------------------------------------------------
    # 2. プロット設定
    # -------------------------------------------------------

    p = plot(
        size = (900, 500),
        title = title,
        xlabel = "Round",
        ylabel = "Context Ratio",
        ylims = (0, 1.05),
        xlims = (2, T+1),
        legend = :outerbottom,
        legend_columns = 8,
        palette = :Spectral_8,
        grid = false,
        margin = 5Plots.mm
    )

    # -------------------------------------------------------
    # 3. 積み重ねバー（累積和方式）
    # -------------------------------------------------------
    cumsum_bottom = zeros(T)

    for k in 1:n_levels
        cumsum_top = cumsum_bottom .+ context_ratios[k, :]

        bar!(
            p,
            2:(T+1),
            cumsum_top;
            fillrange = cumsum_bottom,
            label = levels[k],
            alpha = 0.7,
            bar_width = 0.8,
            linecolor = :white,
            linewidth = 0.5
        )

        cumsum_bottom .= cumsum_top
    end

    return p
end

# Plot cooporation rate over rounds
function plot_cooperation_rate(data::ExperimentData; title = "Cooperation Rate over Rounds")
    N = data.n_subjects
    T = data.n_periods

    coop_counts = zeros(T)
    total_counts = zeros(T)

    for i in 1:N
        y_seq = data.observations[i]
        for t in 1:T
            if y_seq[t] == 1
                coop_counts[t] += 1
            end
            total_counts[t] += 1
        end
    end

    coop_rates = coop_counts ./ total_counts

    p = plot(
        1:T,
        coop_rates;
        seriestype = :line,
        linewidth = 2,
        legend = false,
        xlabel = "Round",
        ylabel = "Cooperation Rate",
        ylims = (0, 1),
        xlims = (1, T),
        fill = (0, 0.15, :auto),
        title = title,
        size = (900, 500),
        margin = 5Plots.mm
    )

    return p
end

function plot_conditional_coop(data::ExperimentData;
                               title = "P(C | XX) over Rounds")
    N = data.n_subjects
    T = data.n_periods

    histories = ["CC", "CD", "DC", "DD"]
    n_hist = length(histories)
    map_hist = Dict(h => i for (i, h) in enumerate(histories))

    counts_C = zeros(Int, n_hist, T)
    counts_D = zeros(Int, n_hist, T)

    for i in 1:N
        cont_seq = data.contexts[i]   # e.g. "CCC", "CDD", ...
        for t in 1:T
            xx = cont_seq[t][1:2]
            idx = map_hist[xx]
            if cont_seq[t][3] == 'C'
                counts_C[idx, t] += 1
            else
                counts_D[idx, t] += 1
            end
        end
    end

    ratio = counts_C ./ (counts_C .+ counts_D)   # (4 × T)

    p = plot(
        xlabel = "Round",
        ylabel = "P(C | XX)",
        title = title,
        ylims = (0, 1),
        xlims = (2, T+1),
        palette = :Spectral_4,
        legend = :outerbottom,
        legend_columns = 4,
        size = (900, 500),
        margin = 5Plots.mm
    )

    for i in 1:n_hist
        plot!(
            p,
            2:(T+1),
            ratio[i, :];
            label = histories[i],
            linewidth = 2,
            fill = (0, 0.12, :auto)
        )
    end

    return p
end
