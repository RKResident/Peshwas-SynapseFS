#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <stdexcept>
#include <sys/socket.h>
#include <system_error>
#include <vector>
#include <unordered_set>

const int hash_len = 64;

enum Operation : uint16_t {
    Op_PUSH = 0,
    Op_PULL = 1
};
enum Status : uint16_t {
    St_DECLINED = 0,
    St_ACCEPTED = 1
};
const uint32_t err_high = 0xFFFF0000;
enum Error : uint16_t {
    Err_OK      = 0,
    Err_INVALID_INVOKATION  = 1,
    Err_FILESYSTEM_ERROR    = 2,
    Err_FAILED_TO_PARSE_JSON            = 3,
    Err_INVALID_DATA   = 4,
    Err_INVALID_HASH        = 5,
    Err_NETWORK_ERROR       = 6,
    Err_PEER_ERROR          = 7,
};

inline const std::string get_err(const uint32_t err) {
    switch((Error)err) {
        case Err_OK:
            return "No Error";
        case Err_INVALID_INVOKATION:
            return "Invalid Invokation";
        case Err_FILESYSTEM_ERROR:
            return "Filesystem Error";
        case Err_FAILED_TO_PARSE_JSON:
            return "Failed to Parse JSON";
        case Err_INVALID_DATA:
            return "Invalid Data";
        case Err_INVALID_HASH:
            return "Invalid Hash";
        case Err_NETWORK_ERROR:
            return "Network Error";
        case Err_PEER_ERROR:
            return "Peer Error";
        default:
            return "Unrecognized Error";
    }
}

typedef std::array<char, hash_len> Hash;

inline std::ostream& operator<<(std::ostream& os, const Hash& hash) {
    for (char c : hash) {
        os << c;
    }
    return os;
}

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
/* A hash arriving over the network becomes a FILE PATH, so validating it is a
 * security boundary rather than tidiness: 64 bytes containing '/' and '.' walk
 * out of the object store and write anywhere the process can reach. Length
 * alone was checked, which does not stop that.
 */
inline bool is_hex_hash(const char *str, std::size_t len) {
    if(len != hash_len) {
        return false;
    }
    for(std::size_t i = 0; i < len; i++) {
        const char c = str[i];
        if(!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            return false;
        }
    }
    return true;
}
inline bool is_hex_hash(const Hash &hash) {
    return is_hex_hash(hash.data(), hash_len);
}
inline bool is_hex_hash(const std::string &hash) {
    return is_hex_hash(hash.c_str(), hash.size());
}

inline void make_hash(Hash &out, const char *str) {
    if(!is_hex_hash(str, std::strlen(str))) {
        std::cerr << str << std::endl;
        throw std::invalid_argument("Hash must be exactly 64 lowercase hex characters");
    }

    std::memcpy(out.data(), str, hash_len);
}

inline std::string tmp_path(const std::string &file_name) {
    return "objects/tmp/" + file_name;
}

inline std::string branch_path(const std::string &branch) {
    return "refs/heads/" + branch;
}
inline bool is_valid_branch_name(const std::string &branch) {
    if(branch.size() == 0) {
        return false;
    }
    char prev = '/';
    for(char c : branch) {
        if((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
                || c == '_' || c == '-' || (c == '/' && prev != '/')) {
            prev = c;
        } else {
            return false;
        }
    }
    if(prev == '/') {
        return false;
    } else {
        return true;
    }
}

// objects/<ab>/<cd>/<60-hex>, for EVERY object kind.
inline std::string hash_path(const Hash hash) {
    std::string path = "objects/";
    path.append(hash.data(), 2);
    path.push_back('/');
    path.append(hash.data() + 2, 2);
    path.push_back('/');
    path.append(hash.data() + 4, hash_len - 4);
    return path;
}

inline bool has_hash(const Hash hash) {
    std::filesystem::path file_path = hash_path(hash);
    if(std::filesystem::exists(file_path)) {
        return true;
    } else {
        return false;
    }
}

inline bool send_all(int sock, const void* data, size_t len) {
    const char* ptr = static_cast<const char*>(data);
    while(len > 0) {
        ssize_t n = send(sock, ptr, len, 0);
        if(n < 0) {
            if(errno == EINTR) {
                continue;
            }
            return false;
        }
        if(n == 0) {
            return false;
        }
        ptr += n;
        len -= n;
    }
    return true;
}
inline bool recv_all(int sock, void* data, size_t len) {
    char* ptr = static_cast<char*>(data);
    while(len > 0) {
        ssize_t n = recv(sock, ptr, len, 0);
        if(n < 0) {
            if(errno == EINTR) {
                continue;
            }
            return false;
        }
        if(n == 0) {
            return false;
        }
        ptr += n;
        len -= n;
    }
    return true;
}

inline bool send_string(int sock, const std::string &str) {
    uint32_t len = htonl(static_cast<uint32_t>(str.size()));

    return send_all(sock, &len, sizeof(len)) &&
           send_all(sock, str.data(), str.size());
}
inline bool recv_string(int sock, std::string &str) {
    uint32_t net_len;
    if(!recv_all(sock, &net_len, sizeof(net_len))) { return false; }
    uint32_t len = ntohl(net_len);
    str.resize(len);
    return recv_all(sock, str.data(), len);
}

inline int send_file(int sock, const std::filesystem::path& path) {
    std::error_code ec;
    auto file_size = std::filesystem::file_size(path, ec);
    if(ec || file_size > UINT32_MAX) {
        uint32_t err = htonl(err_high | Err_FILESYSTEM_ERROR);
        if(!send_all(sock, &err, sizeof(err))) { return Err_NETWORK_ERROR; }
        return Err_FILESYSTEM_ERROR;
    }

    uint32_t len = static_cast<uint32_t>(file_size);
    uint32_t net_len = htonl(len);

    std::ifstream file(path, std::ios::binary);
    if(!file) {
        uint32_t err = htonl(err_high | Err_FILESYSTEM_ERROR);
        if(!send_all(sock, &err, sizeof(err))) { return Err_NETWORK_ERROR; }
        return Err_FILESYSTEM_ERROR;
    }

    std::vector<char> data(len);
    if(len > 0) {
        file.read(data.data(), len);
        if(!file) {
            uint32_t err = htonl(err_high | Err_FILESYSTEM_ERROR);
            if(!send_all(sock, &err, sizeof(err))) { return Err_NETWORK_ERROR; }
            return Err_FILESYSTEM_ERROR;
        }
    }

    // [4 bytes][file data]
    if(!send_all(sock, &net_len, sizeof(net_len))) {
        return Err_NETWORK_ERROR;
    }
    if(len > 0 && !send_all(sock, data.data(), len)) {
        return Err_NETWORK_ERROR;
    }

    return 0;
}
inline int recv_file(int sock, const std::filesystem::path& final_path) {
    uint32_t net_len;
    if(!recv_all(sock, &net_len, sizeof(net_len))) {
        return Err_NETWORK_ERROR;
    }
    uint32_t len = ntohl(net_len);
    if((len & err_high) == err_high) {
        return len & ~err_high;
    }

    // Temporary file in the same directory as the final file.
    std::filesystem::path tmp_path = final_path;
    tmp_path += ".tmp";

    std::error_code ec;
    std::filesystem::create_directories(tmp_path.parent_path(), ec);
    if(ec) {
        std::cerr << "failed to create directory: " << ec.message() << '\n';
        return Err_FILESYSTEM_ERROR;
    }

    std::ofstream file(tmp_path, std::ios::binary | std::ios::trunc);
    if(!file) {
        return Err_FILESYSTEM_ERROR;
    }

    std::vector<char> data(len);
    if(len > 0) {
        if(!recv_all(sock, data.data(), len)) {
            file.close();
            std::filesystem::remove(tmp_path);
            return Err_NETWORK_ERROR;
        }
        file.write(data.data(), len);
        if(!file) {
            file.close();
            std::filesystem::remove(tmp_path);
            return Err_NETWORK_ERROR;
        }
    }

    file.close();
    if(!file) {
        std::filesystem::remove(tmp_path);
        return Err_FILESYSTEM_ERROR;
    }

    // Atomic replacement/install.
    std::filesystem::rename(tmp_path, final_path, ec);

    if(ec) {
        std::filesystem::remove(tmp_path);
        return Err_FILESYSTEM_ERROR;
    }
    return 0;
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



