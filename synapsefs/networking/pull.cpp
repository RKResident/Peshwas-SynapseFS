#include <filesystem>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdint>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <system_error>
#include <unistd.h>

#include "network_common.hpp"

int pull(const int client, const std::string branch) {
    std::vector<Hash> req_hash_objects;

    uint32_t rho_len;
    if(!recv_all(client, &rho_len, sizeof(rho_len))) { return 1; }
    rho_len = ntohl(rho_len);
    std::cout << "received required hash objects size (" << rho_len << " records)" << std::endl;
    req_hash_objects.resize(rho_len);
    if(!recv_all(client, req_hash_objects.data(), rho_len * hash_len)) { return 1; }
    std::cout << "received required hash objects (" << rho_len << " records)" << std::endl;

    // char instead of bool cuz std::vector<bool> is weird and stupid and horrible
    std::vector<char> hash_exists(rho_len, 0);

    // nearly forgotten security issue lol
    for(const Hash &h : req_hash_objects) {
        if(!is_hex_hash(h)) {
            std::cerr << "peer sent a non-hex object hash; refusing" << std::endl;
            return 1;
        }
    }

    for(uint32_t i = 0; i < rho_len; i++) {
        hash_exists[i] = has_hash(req_hash_objects[i]);
    }
    std::cout << "sending hash status..." << std::endl;
    if(!send_all(client, hash_exists.data(), hash_exists.size())) { return 1; }
    std::cout << "sent hash status" << std::endl;
    
    for(uint32_t i = 0; i < rho_len; i++) {
        if(hash_exists[i]) {
            std::cout << "skipping hash " << req_hash_objects[i] << ": already exists" << std::endl;
            continue;
        }
        std::cout << "receiving hash " << req_hash_objects[i] << "..." << std::endl;
        if(!recv_file(client, hash_path(req_hash_objects[i]))) { return 1; }
        std::cout << "received hash " << req_hash_objects[i] << std::endl;
    }

    if(req_hash_objects.empty()) {
        std::cerr << "peer sent no objects for branch '" << branch
                  << "'; leaving refs untouched" << std::endl;
        return 1;
    }

    std::filesystem::path branch_head_path = branch_path(branch);
    std::error_code ec;
    std::filesystem::create_directories(branch_head_path.parent_path(), ec);
    if(ec) {
        std::cerr << "failed to create head directory: " << ec.message() << std::endl;
        return 1;
    }

    std::ofstream file(branch_head_path);
    if(!file) {
        std::cerr << "could not create branch file" << std::endl;
        return 1;
    }
    file << req_hash_objects.back();
    file.close();
    if(!file) {
        std::cerr << "could not write to branch file" << std::endl;
        return 1;
    }

    if(!recv_file(client, head_path())) { return 1; }

    close(client);
    return 0;
}










