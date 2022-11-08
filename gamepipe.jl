using Sockets
using ReinforcementLearning
using .Threads
# accept connection
ENV["LANG"] = "C.UTF-8"


function game(sock)
    gameover = false
    @async while !gameover
        data = readavailable(sock)
        if length(data) == 0
            break
        end
        global oldData = (String(data))
        global splitData = split(oldData, " ")
        score, board = splitData[1], splitData[2:end]
        if score == "gameover"
            close(sock)
            gameover = true
        end
        board = map(x -> parse(Int, x), board)
        println("Score:", score)
        for i ∈ 1:4
            println(board[4i-3:4i])
        end
    end
end


port, connection = listenany(ip"127.0.0.1", 2048)
run(`alacritty -e ./hs2048 -p $port`, wait=false)
sock = accept(connection)
game(sock)

wasd = ["w", "a", "s", "d"]
while isopen(sock)
    action = rand(wasd)
    println("Action:", action)
    write(sock, action)
    sleep(0.001)
end