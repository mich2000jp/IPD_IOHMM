# Define a structure to hold the prepared data
struct ExperimentData
    observations::Vector{Vector{Int}}
    covariates::Vector{Matrix{Float64}}
    experiences::Vector{Vector{String3}}
    n_subjects::Int
    n_periods::Int
    d_cov::Int
end

# Constructor of ExperimentData
function ExperimentData(obs, covs, experiences)
    n_subjects = length(obs)
    n_periods = maximum(length.(obs))
    d_cov = size(covs[1], 2)
    return ExperimentData(
        obs,
        covs,
        experiences,
        n_subjects,
        n_periods,
        d_cov
    )
end

# Overload getindex to access covariates by string key
Base.getindex(data::ExperimentData, key::String) = data.covariates[key]

# Function to map action (C,D) strings to numerical values (1,0)
function map_action(val::Union{String1, Missing})
        if ismissing(val) return missing end
        return val == "D" ? 0.0 : 1.0
end

# Main data preparation function
function prepare_data(input_path::String, output_path::String)
    println("Reading CSV files...")
    df  = CSV.read(input_path, DataFrame; delim=';')

    # Player ID -------------------------------------------------
    all_players = String.(df.player)
    unique_players = sort(unique(all_players))
    player_id_map = Dict{String, Int}(p => i for (i, p) in enumerate(unique_players))



    observations = Vector{Vector{Int}}()
    previous_opp = Vector{Vector{Int}}()
    covariates   = Vector{Matrix{Float64}}()
    experiences  = Vector{Vector{String3}}()
    player_ids = Int[]

    grouped = groupby(df, :player)

    for sub_df in grouped
        sort!(sub_df, :round)
        sub_df = sub_df[sub_df.round .> 1, :] # Exclude round 1
        T = nrow(sub_df)

        # observations y_t --------------------------------------------------
        y = map(map_action, sub_df.action_player)

        # covariates x_t = [prev_action_player, prev_action_opp, interaction]
        prev      = map(map_action, sub_df.prev_player)
        prev_opp  = map(map_action, sub_df.prev_opp)
        interaction = prev .* prev_opp
        X = hcat(prev, prev_opp, interaction)

        # Player ID ---------------------------------------------------------
        pid = player_id_map[String(sub_df.player[1])]
        
        push!(observations, y)
        push!(previous_opp, prev_opp)
        push!(covariates, X)
        push!(experiences, sub_df.context)
        push!(player_ids, pid)
    end

    println("Saving processed data to $output_path ...")
    jldsave(
        output_path;
        observations,
        previous_opp,
        covariates,
        experiences,
        player_ids
    )

    println("Data saved successfully.")
end

# Loading function from JLD2 file
function load_data(filepath::String; ID_excluded=Int[])
    data = load(filepath)

    y    = data["observations"]
    X    = data["covariates"]
    exp  = data["experiences"]

    idx = findall(i ->
        !(i in ID_excluded),
        eachindex(y)
    )

    println("Data loaded successfully. Excluded IDs: $(ID_excluded)")
    return ExperimentData(
        y[idx],
        X[idx],
        exp[idx]
    )
end