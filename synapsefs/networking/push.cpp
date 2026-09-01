#include <chrono>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdint>
#include <fstream>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "nlohmann/json.hpp"
#include "network_common.hpp"


int tensor_get_objects(const Hash &hash, HashList &req_hash_objects) {
    if(req_hash_objects.contains(hash)) {return 0;}
    std::string fp = hash_path(hash);
    std::ifstream f(fp, std::ios::binary);
    if(!f.is_open()) {
        std::cerr << "Cannot open file " << fp << std::endl;
        return Err_FILESYSTEM_ERROR;
    }

    nlohmann::json json;
    try {
        json = nlohmann::json::parse(f);
    } catch (const nlohmann::json::parse_error &e) {
        std::cerr << "Failed to parse JSON: " << e.what() << std::endl;
        return Err_FAILED_TO_PARSE_JSON;
    }

    if(!json.contains("base_tensor_manifest") || !json["base_tensor_manifest"].is_string()
            && !json["base_tensor_manifest"].is_null()) {
        std::cerr << "Missing or invalid base_tensor_manifest" << std::endl;
        return Err_INVALID_DATA;
    }
    if(!json.contains("base_row_permutation") || !json["base_row_permutation"].is_string()
            && !json["base_row_permutation"].is_null()) {
        std::cerr << "Missing or invalid base_row_permutation" << std::endl;
        return Err_INVALID_DATA;
    }
    if(!json.contains("base_col_permutation") || !json["base_col_permutation"].is_string()
            && !json["base_col_permutation"].is_null()) {
        std::cerr << "Missing or invalid base_col_permutation" << std::endl;
        return Err_INVALID_DATA;
    }
    if(!json.contains("chunks") || !json["chunks"].is_array()) {
        std::cerr << "Missing or invalid chunks" << std::endl;
        return Err_INVALID_DATA;
    }

    if(!json["base_tensor_manifest"].is_null()) {
        const std::string parent = json["base_tensor_manifest"];
        Hash parent_hash;
        if(!is_hex_hash(parent)) {
            std::cerr << "invalid header object hash" << std::endl;
            return Err_INVALID_HASH;
        } else {
            make_hash(parent_hash, parent.c_str());
        }
        int rv = tensor_get_objects(parent_hash, req_hash_objects);
        if(rv) {
            return rv;
        }
    }

    if(!json["base_row_permutation"].is_null()) {
        const std::string row_perm = json["base_row_permutation"];
        Hash row_perm_hash;
        if(!is_hex_hash(row_perm)) {
            std::cerr << "invalid row permutation hash" << std::endl;
            return Err_INVALID_HASH;
        } else {
            make_hash(row_perm_hash, row_perm.c_str());
        }
        req_hash_objects.insert(row_perm_hash);
    }

    if(!json["base_col_permutation"].is_null()) {
        const std::string col_perm = json["base_col_permutation"];
        Hash col_perm_hash;
        if(!is_hex_hash(col_perm)) {
            std::cerr << "invalid column permutation hash" << std::endl;
            return Err_INVALID_HASH;
        } else {
            make_hash(col_perm_hash, col_perm.c_str());
        }
        req_hash_objects.insert(col_perm_hash);
    }

    for(const auto &chunk : json["chunks"]) {
        if(!chunk.is_object() || !chunk.contains("object")) {
            std::cerr << "invalid chunk" << std::endl;
            return Err_INVALID_DATA;
        }
        std::string chunk_obj_str = chunk["object"];
        if(!is_hex_hash(chunk_obj_str)) {
            std::cerr << "invalid chunk object" << std::endl;
            return Err_INVALID_DATA;
        }
        Hash chunk_obj_hash;
        make_hash(chunk_obj_hash, chunk_obj_str.c_str());
        req_hash_objects.insert(chunk_obj_hash);
    }

    req_hash_objects.insert(hash);

    return 0;
}

int checkpoint_get_objects(const Hash &hash, HashList &req_hash_objects) {
    if(req_hash_objects.contains(hash)) {return 0;}
    std::string fp = hash_path(hash);
    std::ifstream f(fp, std::ios::binary);
    if(!f.is_open()) {
        std::cerr << "Cannot open file " << fp << std::endl;
        return Err_FILESYSTEM_ERROR;
    }

    nlohmann::json json;
    try {
        json = nlohmann::json::parse(f);
    } catch (const nlohmann::json::parse_error &e) {
        std::cerr << "Failed to parse JSON: " << e.what() << std::endl;
        return Err_FAILED_TO_PARSE_JSON;
    }

    if(!json.contains("header_object") || !json["header_object"].is_string()) {
        std::cerr << "Missing or invalid header_object" << std::endl;
        return Err_INVALID_DATA;
    }
    if(!json.contains("tensors") || !json["tensors"].is_object()) {
        std::cerr << "Missing or invalid tensors" << std::endl;
        return Err_INVALID_DATA;
    }

    const std::string header_object = json["header_object"];
    Hash header_object_hash;
    if(!is_hex_hash(header_object)) {
        std::cerr << "invalid header object hash" << std::endl;
        return Err_INVALID_HASH;
    } else {
        make_hash(header_object_hash, header_object.c_str());
    }
    req_hash_objects.insert(header_object_hash);

    /* The config the checkpoint was aligned against is a stored object too,
     * and it was being skipped -- a pull then landed a repo that `verify`
     * rejected with "referenced but not present". It is optional: absent on
     * checkpoints committed without one, and null-able in the manifest.
     */
    if(json.contains("topology_config_hash")
            && json["topology_config_hash"].is_string()) {
        const std::string config = json["topology_config_hash"];
        if(config.size() != hash_len) {
            std::cerr << config << "invalid topology config hash" << std::endl;
            return Err_INVALID_HASH;
        }
        Hash config_hash;
        make_hash(config_hash, config.c_str());
        req_hash_objects.insert(config_hash);
    }

    for(const auto &[k, tensor] : json["tensors"].items()) {
        if(!tensor.is_string()) {
            std::cerr << "invalid pack hash" << std::endl;
            return Err_INVALID_HASH;
        }
        std::string tensor_str = tensor;
        Hash tensor_hash;
        if(!is_hex_hash(tensor_str)) {
            std::cerr << "invalid pack hash" << std::endl;
            return Err_INVALID_HASH;
        }
        make_hash(tensor_hash, tensor_str.c_str());
        int rv = tensor_get_objects(tensor_hash, req_hash_objects);
        if(rv) {
            return rv;
        }
    }
    req_hash_objects.insert(hash);
    
    return 0;
}

int commit_get_objects(const Hash &hash, HashList &req_hash_objects) {
    if(req_hash_objects.contains(hash)) {return 0;}
    std::string fp = hash_path(hash);
    std::ifstream f(fp, std::ios::binary);
    if(!f.is_open()) {
        std::cerr << "Cannot open file " << fp << std::endl;
        return Err_FILESYSTEM_ERROR;
    }

    nlohmann::json json;
    try {
        json = nlohmann::json::parse(f);
    } catch (const nlohmann::json::parse_error &e) {
        std::cerr << "Failed to parse JSON: " << e.what() << std::endl;
        return Err_FAILED_TO_PARSE_JSON;
    }

    if(!json.contains("checkpoint_manifest") || !json["checkpoint_manifest"].is_string()) {
        std::cerr << "Missing or invalid checkpoint_manifest" << std::endl;
        return Err_INVALID_DATA;
    }
    if(!json.contains("parents") || !json["parents"].is_array()) {
        std::cerr << "Missing or invalid parents" << std::endl;
        return Err_INVALID_DATA;
    }

    const std::string checkpoint = json["checkpoint_manifest"];
    Hash checkpoint_hash;
    if(!is_hex_hash(checkpoint)) {
        std::cerr << "invalid checkpoint-manifest hash" << std::endl;
        return Err_INVALID_HASH;
    } else {
        make_hash(checkpoint_hash, checkpoint.c_str());
    }
    int cgo_rv = checkpoint_get_objects(checkpoint_hash, req_hash_objects);
    if(cgo_rv) {
        return cgo_rv;
    }
    for(const auto &parent_json : json["parents"]) {
        if(!parent_json.is_string()) {
            std::cerr << "invalid parent hash" << std::endl;
            return Err_INVALID_HASH;
        }
        std::string parent = parent_json;
        if(!is_hex_hash(parent)) {
            std::cerr << "invalid parent hash" << std::endl;
            return Err_INVALID_HASH;
        }
        Hash ph;
        make_hash(ph, parent.c_str());
        int cgo_rv = commit_get_objects(ph, req_hash_objects);
        if(cgo_rv) {
            return cgo_rv;
        }
    }
    req_hash_objects.insert(hash);
    
    return 0;
}

int branch_get_objects(const std::string &branch, HashList &req_hash_objects) {
    std::string fp = branch_path(branch);
    std::ifstream f(fp, std::ios::binary);
    if(!f.is_open()) {
        std::cerr << "Cannot open file " << fp << std::endl;
        return Err_FILESYSTEM_ERROR;
    }
    Hash commit_hash;
    f.read(commit_hash.data(), hash_len);

    if(f.gcount() != hash_len) {
        std::cerr << "Hash too short " << fp << std::endl;
        return Err_INVALID_HASH;
    }

    int rv = commit_get_objects(commit_hash, req_hash_objects);
    if(rv) { return rv; }
    return 0;
}

int push(const int client, const std::string branch) {
    if(!is_valid_branch_name(branch)) {
        return Err_INVALID_INVOKATION;
    }
    HashList req_hash_objects;
    if(int err = branch_get_objects(branch, req_hash_objects)) {
        std::cerr << "could not resolve branch: " << branch << std::endl;
        // The peer is waiting on a length-prefixed lists; send an empty one so
        // it fails cleanly instead of blocking on a socket that never speaks.
        uint32_t err_net = htonl(err_high | err);
        send_all(client, &err_net, sizeof(err_net));
        return err;
    }

    std::cout << "sending required hash objects (" << req_hash_objects.ordered.size()
        << " records)" << std::endl;
    uint32_t rho_len = htonl(req_hash_objects.ordered.size());
    if(!send_all(client, &rho_len, sizeof(rho_len))) { return Err_NETWORK_ERROR; }
    if(!send_all(client, req_hash_objects.ordered.data(),
                req_hash_objects.ordered.size() * hash_len)) { return Err_NETWORK_ERROR; }
    std::cout << "sent required hash objects (" << req_hash_objects.ordered.size()
        << " records)" << std::endl;

    rho_len = ntohl(rho_len);

    std::vector<char> hash_exists(rho_len);

    if(!recv_all(client, hash_exists.data(), rho_len)) { return Err_NETWORK_ERROR; }
    std::cout << "received hash status" << std::endl;

    for(uint32_t i = 0; i < rho_len; i++) {
        if(hash_exists[i]) {
            std::cout << "skipping hash " << req_hash_objects.ordered[i]
                << ": already exists" << std::endl;
            continue;
        }
        std::cout << "sending hash " << req_hash_objects.ordered[i] << "..." << std::endl;
        if(int err = send_file(client, hash_path(req_hash_objects.ordered[i]))) {
            return err;
        }
        std::cout << "sent hash " << req_hash_objects.ordered[i] << std::endl;
    }


    return 0;
}










