using Sockets

export HS2048Env

Base.@kwdef mutable struct HS2048Env <: AbstractEnv
    reward::UInt = 0
    state::Vector{UInt8} = zeros(UInt, 16)
    actions::Vector{String} = ["w", "a", "s", "d"]
    isDone::Bool = false
    sock::TCPSocket = TCPSocket()
    legalActions::Vector{UInt8} = [1, 2, 3, 4]
end


function (env::HS2048Env)(action)
    write(env.sock, ["w", "a", "s", "d"][action])
    readBoardState(env)
end

RLBase.action_space(::HS2048Env) = Base.OneTo(4)
RLBase.state(env::HS2048Env) = env.state
RLBase.is_terminated(env::HS2048Env) = env.isDone

# State space is a 16 element vector of UInt8
RLBase.state_space(::HS2048Env) = Space(fill(0:16, 16))
RLBase.reward(env::HS2048Env) = env.reward
RLBase.legal_action_space_mask(env::HS2048Env)::Vector{Bool} = map(x -> x ∈ env.legalActions, 1:4)
RLBase.legal_action_space(env::HS2048Env) = env.legalActions

RLBase.ActionStyle(::HS2048Env) = FULL_ACTION_SET
#RLBase.RewardStyle(::HS2048Env) = TERMINAL_REWARD

readBoardState(env) = begin
    data = readavailable(env.sock) |> String 
    data = split(data, "\n") |> first
    if length(data) == 0
        return
    else
        if occursin("gameover", data)
            env.isDone = true
            println(env.reward)
            println("Max score reached:", env.state |> maximum |> exp2 |> Int)
        else
            splitData = split(data, " ")
            parsed = map(x -> parse(UInt, x), splitData)
            env.reward, env.state = parsed[1], parsed[2:17]
            env.legalActions = parsed[18:end]
        end
    end
    env
end


function ReinforcementLearningBase.reset!(env::HS2048Env)
    close(env.sock)
    port, connection = listenany(ip"127.0.0.1", 2048)
    #run(`alacritty -e ./hs2048 -p $port '&'`, wait=false)
    run(`xterm -e ./hs2048 -p $port`, wait=false)
    state = readBoardState(env)
    env.sock = accept(connection)
    readBoardState(env)
    env.isDone = false
end

