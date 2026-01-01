include("models/glm.jl")
include("models/glmhmm.jl")
include("models/iohmm.jl")

function model_selector(name, data::ExperimentData, K::Int)
    if name == "glm"
       model = glm(data)
       model_gq = glm(data; track=true)
    elseif name == "glmhmm"
        model = glmhmm(data, K)
        model_gq = glmhmm(data, K; track=true)
    elseif name == "iohmm_mealy"
        model = iohmm_mealy(data, K)
        model_gq = iohmm_mealy(data, K; track=true)
    elseif name == "iohmm_moore"
        model = iohmm_moore(data, K)
        model_gq = iohmm_moore(data, K; track=true)
    else
        error("Unknown model name: $name")
    end
    return model, model_gq
end