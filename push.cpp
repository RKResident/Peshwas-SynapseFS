#include <array>
#include <iostream>
#include <string>
#include <cstring>
#include <cstdint>
#include <fstream>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <nlohmann/json.hpp>
#include <unordered_set>

const int hash_len = 64;

typedef std::array<char, hash_len> Hash;

struct HashHasher {
    std::size_t operator()(const Hash &hash) const noexcept {
        std::size_t h = 0;

        for(int i = 0; i < sizeof(std::size_t); i++) {
            ((char*)&h)[i] = ((hash[2*i] <= '9' ? hash[2*i] - '0' : hash[2*i] - 'a' + 10) << 4) |
                (hash[2*i+1] <= '9' ? hash[2*i+1] - '0' : hash[2*i+1] - 'a' + 10);
        }

        return h;
    }
};
struct HashList {
    std::vector<Hash> ordered;
    std::unordered_set<Hash, HashHasher> seen;

    bool insert(const Hash &hash) {
        if (!seen.insert(hash).second)
            return false;

        ordered.push_back(hash);
        return true;
    }

    bool contains(const Hash &hash) const {
        return seen.find(hash) != seen.end();
    }
};
inline void make_hash(Hash &out, const char *str) {
    if(std::strlen(str) != 64) {
        throw std::invalid_argument("Hash must be exactly 64 characters");
    }

    std::memcpy(out.data(), str, 64);
}

std::string branch_path(const std::string branch) {
    return "refs/heads/" + branch;
}
std::string hash_path(const Hash hash) {
    std::string path = "objects/";
    path.append(hash.data(), 2);
    path.append(hash.data() + 2, hash_len - 2);
    return path;
}
std::string pack_path(const Hash pack_hash) {
    return std::string("objects/pack/pack-").append(pack_hash.data(), hash_len).append(".pack");
}
std::string idx_path(const Hash pack_hash) {
    return std::string("objects/pack/pack-").append(pack_hash.data(), hash_len).append(".idx");
}

/* Dependency Tree:
 * branch   > commit
 * commit   > checkpoint_manifest
 *          > parents
 * checkpoint_manifest  > header_object
 *                      > tensor_manifests
 * tensor_manifests > parents
 *                  > base_row_permutation
 *                  > base_col_permutation
 *                  > chunks
 */

int tensor_get_objects(const Hash &hash, HashList &req_hash_objects, HashList &req_packs) {
    if(req_hash_objects.contains(hash)) {return 0;}
    std::string fp = hash_path(hash);
    std::ifstream f(fp, std::ios::binary);
    if(!f.is_open()) {
        std::cerr << "Cannot open file " << fp << std::endl;
        return 1;
    }

    nlohmann::json json;
    try {
        json = nlohmann::json::parse(f);
    } catch (const nlohmann::json::parse_error &e) {
        std::cerr << "Failed to parse JSON: " << e.what() << std::endl;
        return 1;
    }

    if(!json.contains("base_tensor_manifest") || !json["base_tensor_manifest"].is_string()) {
        std::cerr << "Missing or invalid base_tensor_manifest" << std::endl;
        return 1;
    }
    if(!json.contains("base_row_permutation") || !json["base_row_permutation"].is_string()) {
        std::cerr << "Missing or invalid base_row_permutation" << std::endl;
        return 1;
    }
    if(!json.contains("base_col_permutation") || !json["base_col_permutation"].is_string()) {
        std::cerr << "Missing or invalid base_col_permutation" << std::endl;
        return 1;
    }
    if(!json.contains("chunks") || !json["chunks"].is_array()) {
        std::cerr << "Missing or invalid chunks" << std::endl;
        return 1;
    }

    if(!json["base_tensor_manifest"].is_null()) {
        const std::string parent = json["base_tensor_manifest"];
        Hash parent_hash;
        if(parent.size() != 64) {
            std::cerr << "invalid header object hash" << std::endl;
            return 1;
        } else {
            make_hash(parent_hash, parent.c_str());
        }
        tensor_get_objects(parent_hash, req_hash_objects, req_packs);
    }

    if(!json["base_row_permutation"].is_null()) {
        const std::string row_perm = json["base_row_permutation"];
        Hash row_perm_hash;
        if(row_perm.size() != 64) {
            std::cerr << "invalid row permutation hash" << std::endl;
            return 1;
        } else {
            make_hash(row_perm_hash, row_perm.c_str());
        }
        req_packs.insert(row_perm_hash);
    }

    for(const auto &chunk : json["chunks"]) {
        if(!chunk.is_object() || !chunk.contains("object")) {
            std::cerr << "invalid chunk" << std::endl;
            return 1;
        }
        std::string chunk_obj_str = chunk["object"];
        if(chunk_obj_str.size() != 64) {
            std::cerr << "invalid chunk object" << std::endl;
            return 1;
        }
        Hash chunk_obj_hash;
        make_hash(chunk_obj_hash, chunk_obj_str.c_str());
        req_packs.insert(chunk_obj_hash);
    }


    return 0;
}

int checkpoint_get_objects(const Hash &hash, HashList &req_hash_objects, HashList &req_packs) {
    if(req_hash_objects.contains(hash)) {return 0;}
    std::string fp = hash_path(hash);
    std::ifstream f(fp, std::ios::binary);
    if(!f.is_open()) {
        std::cerr << "Cannot open file " << fp << std::endl;
        return 1;
    }

    nlohmann::json json;
    try {
        json = nlohmann::json::parse(f);
    } catch (const nlohmann::json::parse_error &e) {
        std::cerr << "Failed to parse JSON: " << e.what() << std::endl;
        return 1;
    }

    if(!json.contains("header_object") || !json["header_object"].is_string()) {
        std::cerr << "Missing or invalid header_object" << std::endl;
        return 1;
    }
    if(!json.contains("tensors") || !json["tensors"].is_object()) {
        std::cerr << "Missing or invalid tensors" << std::endl;
        return 1;
    }

    const std::string header_object = json["header_object"];
    Hash header_object_hash;
    if(header_object.size() != 64) {
        std::cerr << "invalid header object hash" << std::endl;
        return 1;
    } else {
        make_hash(header_object_hash, header_object.c_str());
    }
    req_hash_objects.insert(header_object_hash);

    for(const auto &[k, tensor] : json["tensors"].items()) {
        if(!tensor.is_string()) {
            std::cerr << "invalid pack hash" << std::endl;
            return 1;
        }
        std::string tensor_str = tensor;
        if(tensor_str.size() != 64) {
            std::cerr << "invalid pack hash" << std::endl;
            return 1;
        }
        Hash tensor_hash;
        make_hash(tensor_hash, tensor_str.c_str());
        int rv = tensor_get_objects(tensor_hash, req_hash_objects, req_packs);
        if(rv) {
            return rv;
        }
    }
    req_hash_objects.insert(hash);
    
    return 0;
}

int commit_get_objects(const Hash &hash, HashList &req_hash_objects, HashList &req_packs) {
    if(req_hash_objects.contains(hash)) {return 0;}
    std::string fp = hash_path(hash);
    std::ifstream f(fp, std::ios::binary);
    if(!f.is_open()) {
        std::cerr << "Cannot open file " << fp << std::endl;
        return 1;
    }

    nlohmann::json json;
    try {
        json = nlohmann::json::parse(f);
    } catch (const nlohmann::json::parse_error &e) {
        std::cerr << "Failed to parse JSON: " << e.what() << std::endl;
        return 1;
    }

    if(!json.contains("checkpoint_manifest") || !json["checkpoint_manifest"].is_string()) {
        std::cerr << "Missing or invalid checkpoint_manifest" << std::endl;
        return 1;
    }
    if(!json.contains("parents") || !json["parents"].is_array()) {
        std::cerr << "Missing or invalid parents" << std::endl;
        return 1;
    }

    const std::string checkpoint = json["checkpoint_manifest"];
    Hash checkpoint_hash;
    if(checkpoint.size() != 64) {
        std::cerr << "invalid checkpoint-manifest hash" << std::endl;
        return 1;
    } else {
        make_hash(checkpoint_hash, checkpoint.c_str());
    }
    int cgo_rv = checkpoint_get_objects(checkpoint_hash, req_hash_objects, req_packs);
    if(cgo_rv) {
        return cgo_rv;
    }
    for(const auto &parent_json : json["parents"]) {
        if(!parent_json.is_string()) {
            std::cerr << "invalid parent hash" << std::endl;
            return 1;
        }
        std::string parent = parent_json;
        if(parent.size() != 64) {
            std::cerr << "invalid parent hash" << std::endl;
            return 1;
        }
        Hash ph;
        make_hash(ph, parent.c_str());
        int cgo_rv = commit_get_objects(ph, req_hash_objects, req_packs);
        if(cgo_rv) {
            return cgo_rv;
        }
    }
    req_hash_objects.insert(hash);
    
    return 0;
}

int branch_get_objects(const std::string &branch, HashList &req_hash_objects, HashList &req_packs) {
    std::string fp = branch_path(branch);
    std::ifstream f(fp, std::ios::binary);
    if(!f.is_open()) {
        std::cerr << "Cannot open file " << fp << std::endl;
        return 1;
    }
    Hash commit_hash;
    f.read(commit_hash.data(), hash_len);

    if(f.gcount() != hash_len) {
        std::cerr << "Hash too short " << fp << std::endl;
        return 1;
    }

    int rv = commit_get_objects(commit_hash, req_hash_objects, req_packs);
    if(rv) { return rv; }
    return 0;
}

void sender(const std::string ip, const uint16_t port, const std::string branch) {
    int client = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(client, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect");
        return;
    }
    HashList req_hash_objects;
    HashList req_packs;





    close(client);
}

