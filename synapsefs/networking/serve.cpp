#include <iostream>
#include <string>
#include <cstring>
#include <cstdint>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "push.cpp"
#include "pull.cpp"
#include "network_common.hpp"

int serve(uint16_t port, bool ro) {
    int server = socket(AF_INET, SOCK_STREAM, 0);
    if(server < 0) {
        perror("socket");
        return 1;
    }

    int opt = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if(bind(server, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server);
        return 1;
    }
    if(listen(server, SOMAXCONN) < 0) {
        perror("listen");
        close(server);
        return 1;
    }
    std::cout << "serving on port " << port << std::endl;

    while(true) {
        int client = accept(server, nullptr, nullptr);
        if(client < 0) {
            if(errno == EINTR)
                continue;

            perror("accept");
            close(server);
            return 1;
        }

        uint8_t op;
        if(!recv_all(client, &op, sizeof(op))) {
            close(client);
            continue;
        }

        std::string client_branch;
        if(!recv_string(client, client_branch)) {
            close(client);
            continue;
        }

        std::cerr << "client requested branch "
            << client_branch << std::endl;

        switch(static_cast<Operation>(op)) {
            case Operation::Op_PUSH:
                // handle_push(client, branch);
                if(!ro) {
                    pull(client, client_branch);
                } else {
                    close(client);
                }
                break;
            case Operation::Op_PULL:
                // handle_pull(client, branch);
                push(client, client_branch);
                break;
            default:
                std::cerr << "unknown operation: "
                          << static_cast<int>(op) << std::endl;
                break;
        }

        close(client);
    }
}






