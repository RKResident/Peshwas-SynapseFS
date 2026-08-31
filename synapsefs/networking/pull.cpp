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
    std::vector<Hash> req_chunks;

    uint32_t rho_len;
    if(!recv_all(client, &rho_len, sizeof(rho_len))) { return 1; }
    rho_len = ntohl(rho_len);
    std::cout << "received required hash objects size (" << rho_len << " records)" << std::endl;
    req_hash_objects.resize(rho_len);
    if(!recv_all(client, req_hash_objects.data(), rho_len * hash_len)) { return 1; }
    std::cout << "received required hash objects (" << rho_len << " records)" << std::endl;

    uint32_t rc_len;
    if(!recv_all(client, &rc_len, sizeof(rc_len))) { return 1; }
    rc_len = ntohl(rc_len);
    std::cout << "received required chunks size (" << rc_len << " records)" << std::endl;
    req_chunks.resize(rc_len);
    if(!recv_all(client, req_chunks.data(), rc_len * hash_len)) { return 1; }
    std::cout << "received required chunks (" << rc_len << " records)" << std::endl;

    // char instead of bool cuz std::vector<bool> is weird and stupid and horrible
    std::vector<char> hash_exists(rho_len, 0);
    std::vector<char> chunk_exists(rc_len, 0);

    // Every hash here came off the wire and is about to become a path.
    // Refuse the whole transfer rather than creating files from it.
    for(const Hash &h : req_hash_objects) {
        if(!is_hex_hash(h)) {
            std::cerr << "peer sent a non-hex object hash; refusing" << std::endl;
            return 1;
        }
    }
    for(const Hash &h : req_chunks) {
        if(!is_hex_hash(h)) {
            std::cerr << "peer sent a non-hex chunk hash; refusing" << std::endl;
            return 1;
        }
    }

    for(uint32_t i = 0; i < rho_len; i++) {
        hash_exists[i] = has_hash(req_hash_objects[i]);
    }
    // Bound is rc_len, not rho_len. With rc_len > rho_len the tail stayed 0
    // and those chunks were re-sent every time; with rc_len < rho_len this
    // wrote past the end of chunk_exists.
    for(uint32_t i = 0; i < rc_len; i++) {
        chunk_exists[i] = has_chunk(req_chunks[i]);
    }
    std::cout << "sending hash status..." << std::endl;
    if(!send_all(client, hash_exists.data(), hash_exists.size())) { return 1; }
    std::cout << "sent hash status" << std::endl;
    
    std::cout << "sending chunk status..." << std::endl;
    if(!send_all(client, chunk_exists.data(), chunk_exists.size())) { return 1; }
    std::cout << "sent chunk status" << std::endl;

    for(uint32_t i = 0; i < rho_len; i++) {
        if(hash_exists[i]) {
            std::cout << "skipping hash " << req_hash_objects[i] << ": already exists" << std::endl;
            continue;
        }
        std::cout << "receiving hash " << req_hash_objects[i] << "..." << std::endl;
        if(!recv_file(client, hash_path(req_hash_objects[i]))) { return 1; }
        std::cout << "received hash " << req_hash_objects[i] << std::endl;
    }
    for(uint32_t i = 0; i < rc_len; i++) {
        if(chunk_exists[i]) {
            std::cout << "skipping chunk " << req_chunks[i] << ": already exists" << std::endl;
            continue;
        }
        std::cout << "receiving chunk " << req_chunks[i] << "..." << std::endl;
        if(!recv_file(client, chunk_path(req_chunks[i]))) { return 1; }
        std::cout << "received chunk " << req_chunks[i] << std::endl;
    }

    // The branch tip is the last entry because the sender inserts a commit
    // only after recursing into its manifest and parents. An empty list means
    // the peer resolved nothing -- writing `back()` on it read off the end of
    // the vector and crashed.
    if(req_hash_objects.empty()) {
        std::cerr << "peer sent no objects for branch '" << branch
                  << "'; leaving refs untouched" << std::endl;
        return 1;
    }

    std::filesystem::path head_path = branch_path(branch);
    std::error_code ec;
    std::filesystem::create_directories(head_path.parent_path(), ec);
    if(ec) {
        std::cerr << "failed to create head directory: " << ec.message() << std::endl;
        return 1;
    }

    std::ofstream file(head_path);
    if(!file) {
        std::cerr << "could not create head file" << std::endl;
        return 1;
    }
    file << req_hash_objects.back();
    file.close();
    if(!file) {
        std::cerr << "could not write to head file" << std::endl;
        return 1;
    }

    close(client);
    return 0;
}










