#include <iostream>
#include <string>
#include <cstring>
#include <cstdint>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "network_common.hpp"
#include "serve.cpp"

int connect_server(const std::string ip, const uint16_t port, const std::string branch,
        Operation op, bool &success) {
    if(!is_valid_branch_name(branch)) {
        return Err_INVALID_INVOKATION;
    }
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

    Status st;
    if(!recv_all(client, &st, sizeof(st))) {
        std::cerr << "could not recieve status" << std::endl;
        close(client);
        success = false;
        return 1;
    }
    if(st != St_ACCEPTED) {
        std::cerr << "server declined connection; status: " << st << std::endl;
        close(client);
        success = false;
        return 1;
    }

    std::cout << "connected to server" << std::endl;

    success = true;
    return client;
}

const std::string serve_usage =     "Usage: `synapsefs serve <port> [ro=<0|1>]`\n";
const std::string push_pull_usage = "Usage: `synapsefs <push|pull> <ip> <port> <branch>\n";


int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "Improper usage\n" << serve_usage << push_pull_usage << std::endl;
        return Err_INVALID_INVOKATION;
    }
    if(std::string(argv[1]) == "serve") {
        if(argc != 3 && argc != 4) {
            std::cout << "Improper usage\n" << serve_usage << std::endl;
            return Err_INVALID_INVOKATION;
        }
        uint16_t port;
        try {
            port = std::stoi(argv[2]);
            if(port < 0 || port > UINT16_MAX) {
                throw std::exception();
            }
        } catch(const std::exception &e) {
            std::cout << "Invalid port: " << argv[2] << std::endl;
            return Err_INVALID_INVOKATION;
        }
        bool ro = false;
        if(argc == 4) {
            std::string ro_arg = argv[3];
            if(ro_arg.substr(0, 3) == "ro=" && ro_arg.size() == 4) {
                if(ro_arg[3] == '0') {
                    ro = false;
                } else if(ro_arg[3] == '1') {
                    ro = true;
                } else {
                    std::cerr << "Invalid read-only flag (ro=<0|1>)" << std::endl;
                    return Err_INVALID_INVOKATION;
                }
            } else {
                std::cerr << "Invalid read-only flag (ro=<0|1>)" << std::endl;
                return Err_INVALID_INVOKATION;
            }
        }
        return serve(port, ro);
    } else if(std::string(argv[1]) == "push"
            || std::string(argv[1]) == "pull") {
        if(argc != 5) {
            std::cerr << "Improper usage\n" << push_pull_usage << std::endl;
            return Err_INVALID_INVOKATION;
        }
        const std::string ip = argv[2];
        struct sockaddr_in sa;
        if(inet_pton(AF_INET, argv[2], &sa.sin_addr) != 1) {
            std::cout << "Invalid IPv4 address" << std::endl;
            return Err_INVALID_INVOKATION;
        }
        uint16_t port;
        try {
            port = std::stoi(argv[3]);
            if(port < 0 || port > UINT16_MAX) {
                throw std::exception();
            }
        } catch (const std::exception &e) {
            std::cout << "Invalid port" << argv[3] << std::endl;
            return Err_INVALID_INVOKATION;
        }
        const std::string branch = argv[4];
        if(!is_valid_branch_name(branch)) {
            return Err_INVALID_INVOKATION;
        }
        bool success;
        int socket = connect_server(ip, port, branch,
                std::string(argv[1]) == "push" ? Op_PUSH : Op_PULL, success);
        if(!success) {
            return socket;
        }
        const int rv = (std::string(argv[1]) == "pull")
            ? pull(socket, branch)
            : push(socket, branch);
        close(socket);
        if(rv) {
            std::cerr << "Error: " << get_err(rv) << std::endl;
            return rv;
        }
        return 0;
    } else {
        std::cerr << "Improper usage\n" << serve_usage << push_pull_usage << std::endl;
        return Err_INVALID_INVOKATION;
    }
}










