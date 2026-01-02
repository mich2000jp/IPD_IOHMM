# Define a structure to hold the prepared data
struct ExperimentData
    observations::Vector{Vector{Int}}
    covariates::Vector{Matrix{Float64}}
    n_subjects::Int
    n_periods::Int
    d_cov::Int
end

# Constructor of ExperimentData
function ExperimentData(obs, covs)
    n_subj = length(obs)
    n_periods = maximum(length.(obs))
    d_cov = size(covs[1], 2)
    return ExperimentData(
        obs,
        covs,
        n_subj,
        n_periods,
        d_cov
    )
end

# Overload getindex to access covariates by string key
Base.getindex(data::ExperimentData, key::String) = data.covariates[key]

# Function to map action (C,D) strings to numerical values (1,0)
# Missing values are mapped to 0.0
function map_action(val::Union{String1, Missing})
        if ismissing(val) return 0.0 end
        return val == "D" ? 0.0 : 1.0
end

# Main data preparation function
function prepare_data(fix_path::String, rand_path::String, output_path::String)
    println("Reading CSV files...")

    df_fix  = CSV.read(fix_path, DataFrame; delim=';')
    df_rand = CSV.read(rand_path, DataFrame; delim=';')

    observations = Vector{Vector{Int}}()
    previous_opp = Vector{Vector{Int}}()
    covariates   = Vector{Matrix{Float64}}()

    treatments = Int[]
    player_ids = String[]

    for (df, gid) in [(df_fix, 1), (df_rand, 2)]
        grouped = groupby(df, :player)

        for sub_df in grouped
            sort!(sub_df, :round)
            sub_df = sub_df[sub_df.round .> 1, :] # Exclude round 1
            T = nrow(sub_df)

            # --------------------------------------------------
            # observations y_t
            y = map(map_action, sub_df.action_player)

            # --------------------------------------------------
            # covariates x_t = [prev_action_player, prev_action_opp, interaction]
            prev      = map(map_action, sub_df.prev_player)
            prev_opp  = map(map_action, sub_df.prev_opp)
            interaction = prev .* prev_opp
            X = hcat(prev, prev_opp, interaction)

            
            push!(observations, y)
            push!(previous_opp, prev_opp)
            push!(covariates, X)
            push!(treatments, gid)
            push!(player_ids, String(sub_df.player[1]))
        end
    end

    println("Saving processed data to $output_path ...")
    jldsave(
        output_path;
        observations,
        previous_opp,
        covariates,
        treatments,
        player_ids
    )

    println("Data saved successfully.")
end

# Loading function from JLD2 file
function load_data(filepath::String; condition::Symbol)
    data = load(filepath)

    y    = data["observations"]
    X    = data["covariates"]
    trts = data["treatments"]

    target_trt = (condition == :FP) ? 1 : 2
    idx = findall(==(target_trt), trts)

    return ExperimentData(
        y[idx],
        X[idx]
    )
end
