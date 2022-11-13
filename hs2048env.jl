using Sockets

export HS2048Env

Base.@kwdef mutable struct HS2048Env <: AbstractEnv
    reward::UInt = 0
    state::Vector{UInt8} = zeros(UInt, 16)
    actions::Vector{String} = ["w", "a", "s", "d"]
    isDone::Bool = false
    sock::TCPSocket = TCPSocket()
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


readBoardState(env) = begin
    #sleep(0.001) # wait for data to be written (MIGHT NOT BE NEEDED?)
    data = readavailable(env.sock) |> String
    if length(data) == 0
        return
    else 
        if occursin("gameover", data)
            env.isDone = true
        else
            splitData = split(data, " ") 
            parsed = map(x -> parse(UInt, x), splitData)
            env.reward, env.state = parsed[1], parsed[2:end] 
        end
    end
    env
end


function ReinforcementLearningBase.reset!(env::HS2048Env)
    close(env.sock)
    port, connection = listenany(ip"127.0.0.1", 2048)
    #run(`./hs2048 -p $port '&'`, wait=false)
    run(`alacritty -e ./hs2048 -p $port`, wait=false)
    state = readBoardState(env)
    env.sock = accept(connection)
    readBoardState(env)
    env.isDone = false
end

