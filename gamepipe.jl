using Sockets
using Flux
using Flux.Losses
include("./hs2048env.jl")
using ReinforcementLearning
# accept connection
ENV["LANG"] = "C.UTF-8"

env = HS2048Env()
reset!(env)

ns, na = length(state(env)), length(action_space(env))
agent = Agent(
        policy = QBasedPolicy(
            learner = BasicDQNLearner(
                approximator = NeuralNetworkApproximator(
                    model = Chain(
                        Dense(ns, 64, relu; init = glorot_uniform()),
                        Dense(64, 64, relu; init = glorot_uniform()),
                        Dense(64, na; init = glorot_uniform()),
                    ) |> cpu,
                    optimizer = ADAM(),
                ),
                batch_size = 32,
                min_replay_history = 100,
                loss_func = huber_loss,
            ),
            explorer = EpsilonGreedyExplorer(
                kind = :exp,
                ϵ_stable = 0.01,
                decay_steps = 500,
            ),
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 50_000,
            state = Vector{Float32} => (ns,),
        ),
    )

hook = TotalRewardPerEpisode()
run(agent, env, StopAfterEpisode(50), hook)

