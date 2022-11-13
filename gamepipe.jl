using Flux

using ReinforcementLearning
include("./hs2048env.jl")

ENV["LANG"] = "C.UTF-8"

env = HS2048Env()
#reset!(env)

ns, na = length(state(env)), length(action_space(env))
N_ENV = 16
UPDATE_FREQ = 10
agent = Agent(
        policy = QBasedPolicy(
            learner = A2CLearner(
                approximator = ActorCritic(
                    actor = Chain(
                        Dense(ns, 256, relu; init = glorot_uniform()),
                        Dense(256, na; init = glorot_uniform()),
                    ),
                    critic = Chain(
                        Dense(ns, 256, relu; init = glorot_uniform()),
                        Dense(256, 1; init = glorot_uniform()),
                    ),
                    optimizer = ADAM(1e-3),
                ) |> gpu,
                γ = 0.99f0,
                actor_loss_weight = 1.0f0,
                critic_loss_weight = 0.5f0,
                entropy_loss_weight = 0.001f0,
                update_freq = UPDATE_FREQ,
            ),
            explorer = BatchExplorer(GumbelSoftmaxExplorer()),
        ),
        trajectory = CircularArraySARTTrajectory(;
            capacity = UPDATE_FREQ,
            state = Matrix{Float32} => (ns, N_ENV),
            action = Vector{Int} => (N_ENV,),
            reward = Vector{Float32} => (N_ENV,),
            terminal = Vector{Bool} => (N_ENV,),
        ),
    )

hook = TotalRewardPerEpisode()
run(agent, env, StopAfterEpisode(50), hook)

