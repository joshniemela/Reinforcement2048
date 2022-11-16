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
#env = CartPoleEnv(; T = Float32, rng = rng)
ns, na = length(state(env)), length(action_space(env))

agent = Agent(
    policy = QBasedPolicy(
        learner = BasicDQNLearner(
            approximator = NeuralNetworkApproximator(
                model = Chain(
                    Dense(ns, 128, relu),
                    Dense(128, 128, relu),
                    Dense(128, 64, relu),
                    Dense(64, 32, relu),
                    Dense(32, na),
                ) |> gpu,
                optimizer = ADAM(),
            ),
            batch_size = 32,
            min_replay_history = 100,
            loss_func = huber_loss,
            rng = rng,
        ),
        explorer = EpsilonGreedyExplorer(
            kind = :exp,
            ϵ_stable = 0.01,
            decay_steps = 500,
            rng = rng,
        ),
    ),
    trajectory = CircularArraySARTTrajectory(
        capacity = 1000,
        state = Vector{Float32} => (ns,),
    ),
)
#agent = Agent(
#    policy = QBasedPolicy(
#        learner = DQNLearner(
#            approximator = NeuralNetworkApproximator(
#                model = DuelingNetwork(
#                    base = Chain(
#                        Dense(ns, 128, relu; init = glorot_uniform()),
#                        Dense(128, 128, relu; init = glorot_uniform()),
#                    ),
#                    val = Dense(128, na; init = glorot_uniform()),
#                    adv = Dense(128, na; init = glorot_uniform()),
#                ),
#                optimizer = ADAM(),
#            ) |> gpu,
#            target_approximator = NeuralNetworkApproximator(
#                model = DuelingNetwork(
#                    base = Chain(
#                        Dense(ns, 128, relu; init = glorot_uniform()),
#                        Dense(128, 128, relu; init = glorot_uniform()),
#                    ),
#                    val = Dense(128, na; init = glorot_uniform()),
#                    adv = Dense(128, na; init = glorot_uniform()),
#                ),
#            ) |> gpu,
#            loss_func = huber_loss,
#            stack_size = nothing,
#            batch_size = 32,
#            update_horizon = 1,
#            min_replay_history = 100,
#            update_freq = 1,
#            target_update_freq = 100,
#        ),
#        explorer = EpsilonGreedyExplorer(
#            kind = :exp,
#            ϵ_stable = 0.01,
#            decay_steps = 500,
#        ),
#    ),
#    trajectory = CircularArraySARTTrajectory(
#        capacity = 1000,
#        state = Vector{Float32} => (ns,),
#    ),
#)

stopCondition = StopAfterEpisode(500, is_show_progress=!haskey(ENV, "CI"))
hook = RewardsPerEpisode()

run(agent, env, stopCondition, hook)

using Plots
plot(hook.rewards)