using Flux

using ReinforcementLearning
using Statistics
using Flux.Losses
ENV["LANG"] = "C.UTF-8"
using StableRNGs
include("./hs2048env.jl")


env = HS2048Env()
#reset!(env)
seed = 123
rng = StableRNG(seed)
N_ENV = 128
UPDATE_FREQ = 10

ns, na = length(state(env)), length(action_space(env))

n_atoms = 51
agent = Agent(
    policy = QBasedPolicy(
        learner = RainbowLearner(
            approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 1024, relu; init = glorot_uniform(rng)),
                    Dense(1024, 256, relu; init = glorot_uniform(rng)),
                    Dense(256, na * n_atoms; init = glorot_uniform(rng)),
                ) |> gpu,
                optimizer = ADAM(0.0005),
            ),
            target_approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 1024, relu; init = glorot_uniform(rng)),
                    Dense(1024, 256, relu; init = glorot_uniform(rng)),
                    Dense(256, na * n_atoms; init = glorot_uniform(rng)),
                ) |> gpu,
                optimizer = ADAM(0.0005),
            ),
            n_actions = na,
            n_atoms = n_atoms,
            Vₘₐₓ = 250.0f0,
            Vₘᵢₙ = 0.0f0,
            update_freq = 2,
            γ = 0.985f0,
            update_horizon = 1,
            batch_size = 32,
            stack_size = nothing,
            min_replay_history = 64,
            loss_func = (ŷ, y) -> logitcrossentropy(ŷ, y; agg = identity),
            target_update_freq = 100,
            rng = rng,
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            ϵ_stable = 0.01,
            decay_steps = 15000,
            rng = rng,
        ),
    ),
    trajectory = CircularArrayPSARTTrajectory(
        capacity = 7500,
        state = Vector{Float32} => (ns,),
    ),
)


stopCondition = StopAfterEpisode(500, is_show_progress=!haskey(ENV, "CI"))
hook = RewardsPerEpisode()
run(RandomPolicy(), env, stopCondition, hook)

using Plots
using RollingFunctions

rewards = hook.rewards .|> last .|> log2

means = rollmean(rewards, 50)
stds = rollstd(rewards, 50)

plot(rewards, ribbon = stds, label = "Rolling score over time")

savefig("gamepipe.png")