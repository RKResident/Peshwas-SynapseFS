#include <iostream>
#include <string>
#include <cstring>
#include <cstdint>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "serve.cpp"

int connect_server(const std::string ip, const uint16_t port, const std::string branch,
        Operation op, bool &success) {
    int client = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr(ip.c_str());

    if(connect(client, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect");
        close(client);
        success = false;
        return 1;
    }

    std::cout << "sending operation: " << (op == Op_PULL ? "pull" : "push") << std::endl;
    char op_char = op;
    if(!send_all(client, &op_char, sizeof(op_char))) {
        std::cerr << "could not send operation" << std::endl;
        close(client);
        success = false;
        return 1;
    } else {
        std::cout << "sent operation" << std::endl;
    }
    std::cout << "sending branch: " << branch << std::endl;
    if(!send_string(client, branch)) {
        std::cerr << "could not send branch" << std::endl;
        close(client);
        success = false;
        return -1;
    } else {
        std::cout << "sent branch" << std::endl;
    }

    success = true;
    return client;
}


int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "Improper usage" << std::endl;
        return 1;
    }
    if(std::string(argv[1]) == "serve") {
        if(argc != 3) {
            std::cout << "Improper usage" << std::endl;
            return 1;
        }
        uint16_t port;
        try {
            port = std::stoi(argv[2]);
            if(port < 0 || port > UINT16_MAX) {
                throw std::exception();
            }
        } catch (const std::exception &e) {
            std::cout << "Invalid port" << std::endl;
            return 1;
        }
        return serve(port);
    } else if(std::string(argv[1]) == "push"
            || std::string(argv[1]) == "pull") {
        if(argc != 5) {
            std::cerr << "Improper usage" << std::endl;
            return 1;
        }
        const std::string ip = argv[2];
        struct sockaddr_in sa;
        if(inet_pton(AF_INET, argv[2], &sa.sin_addr) != 1) {
            std::cout << "Invalid IPv4 address" << std::endl;
            return 1;
        }
        uint16_t port;
        try {
            port = std::stoi(argv[3]);
            if(port < 0 || port > UINT16_MAX) {
                throw std::exception();
            }
        } catch (const std::exception &e) {
            std::cout << "Invalid port" << std::endl;
            return 1;
        }
        const std::string branch = argv[4];
        bool success;
        int socket = connect_server(ip, port, branch,
                std::string(argv[1]) == "push" ? Op_PUSH : Op_PULL, success);
        if(!success) {
            std::cerr << "could not connect to server" << std::endl;
            return 1;
        }
        if(std::string(argv[1]) == "pull") {
            if(!pull(socket, branch)) {
                close(socket);
                return 1;
            } else {
                return 0;
            }
        } else {
            if(!push(socket, branch)) {
                close(socket);
                return 1;
            } else {
                return 0;
            }
        }
    } else {
        std::cerr << "Improper usage" << std::endl;
        return 1;
    }
}










