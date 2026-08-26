#include <iostream>
#include <string>
#include <cstring>
#include <cstdint>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <nlohmann/json.hpp>


void listener(const uint16_t port, const std::string branch) {
    int server = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    bind(server, (sockaddr*)&addr, sizeof(addr));
    listen(server, 1);

    std::cout << "Waiting...\n";

    int client = accept(server, nullptr, nullptr);

    uint32_t length;
    recv(client, &length, 4, 0);

    length = ntohl(length);

    std::string message(length, '\0');
    recv(client, message.data(), length, 0);

    std::cout << "Received: " << message << '\n';

    close(client);
    close(server);
}

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "Improper usage" << std::endl;
    }
    if(std::string(argv[1]) == "listener") {
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
        const std::string branch = argv[3];
        listener(port, branch);
    } else {
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
        sender(ip, port, branch);
    }
}










