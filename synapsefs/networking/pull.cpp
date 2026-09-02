#include <cerrno>
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

// pulls a repositry from a client running the push function. returns 0 on success,
// non-zero on failure. requires the network client (which it does not own) and the
// name of a valid branch to pull
int pull(const int client, const std::string branch) {
    if(!is_valid_branch_name(branch)) {
        return Err_INVALID_INVOKATION;
    }
    std::vector<Hash> req_hash_objects;

    uint32_t rho_len;
    if(!recv_all(client, &rho_len, sizeof(rho_len))) { return Err_NETWORK_ERROR; }
    rho_len = ntohl(rho_len);
    if((rho_len & err_high) == err_high) {
        // detect errors
        std::cerr << "peer error: " << get_err(rho_len ^ err_high);
        return Err_PEER_ERROR;
    }
    std::cout << "received required hash objects size (" << rho_len << " records)" << std::endl;
    req_hash_objects.resize(rho_len);
    if(!recv_all(client, req_hash_objects.data(), rho_len * hash_len)) { return Err_NETWORK_ERROR; }
    std::cout << "received required hash objects (" << rho_len << " records)" << std::endl;

    // char instead of bool cuz std::vector<bool> is weird and stupid and horrible
    std::vector<char> hash_exists(rho_len, 0);

    // nearly forgotten security issue lol
    for(const Hash &h : req_hash_objects) {
        if(!is_hex_hash(h)) {
            std::cerr << "peer sent a non-hex object hash; refusing" << std::endl;
            return Err_INVALID_HASH;
        }
    }

    for(uint32_t i = 0; i < rho_len; i++) {
        hash_exists[i] = has_hash(req_hash_objects[i]);
    }
    std::cout << "sending hash status" << std::endl;
    if(!send_all(client, hash_exists.data(), hash_exists.size())) { return Err_NETWORK_ERROR; }
    std::cout << "sent hash status" << std::endl;

    int skipped = 0;
    int transferred = 0;
    
    // recieve nonexistent hash objects, skip existing ones
    for(uint32_t i = 0; i < rho_len; i++) {
        if(hash_exists[i]) {
            std::cout << "skipping hash " << req_hash_objects[i] << ": already exists" << std::endl;
            skipped++;
            continue;
        }
        std::cout << "receiving hash " << req_hash_objects[i] << std::endl;
        if(int err = recv_file(client, hash_path(req_hash_objects[i]))) {
            return err;
        }
        std::cout << "received hash " << req_hash_objects[i] << std::endl;
        transferred++;
    }

    // Since hashes are stored in reverse hierarchical order (near-DFS), the last one is the root,
    // which is the commit object. The hash of this is stored in refs/heads/<branch>
    if(req_hash_objects.empty()) {
        std::cerr << "peer sent no objects for branch '" << branch
                  << "'; leaving refs untouched" << std::endl;
        return Err_INVALID_DATA;
    }

    std::filesystem::path branch_head_path = branch_path(branch);
    std::error_code ec;
    std::filesystem::create_directories(branch_head_path.parent_path(), ec);
    if(ec) {
        std::cerr << "failed to create branch directory: " << ec.message() << std::endl;
        return Err_FILESYSTEM_ERROR;
    }

    std::filesystem::path branch_head_tmp_path = tmp_path(branch);
    std::filesystem::create_directories(branch_head_tmp_path.parent_path(), ec);
    if(ec) {
        std::cerr << "failed to create temporary directory: " << ec.message() << std::endl;
        return Err_FILESYSTEM_ERROR;
    }
    std::ofstream file(branch_head_tmp_path);
    if(!file) {
        std::cerr << "could not create temporary file" << std::endl;
        return Err_FILESYSTEM_ERROR;
    }
    file << req_hash_objects.back();
    file.close();
    if(!file) {
        std::cerr << "could not write to temporary file" << std::endl;
        return Err_FILESYSTEM_ERROR;
    }

    // make sure it's atomic!
    std::filesystem::rename(branch_head_tmp_path, branch_head_path, ec);
    if(ec) {
        std::filesystem::remove(branch_head_tmp_path);
        return Err_FILESYSTEM_ERROR;
    }

    std::cout << "Transfer Complete: "
        << (skipped+transferred) << " objects, " << transferred << " transferred"
        << skipped << " skipped, " << std::endl;

    return 0;
}










