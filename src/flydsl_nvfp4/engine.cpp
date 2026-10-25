// src/flydsl_nvfp4/engine.cpp - Native FlyDSL NVFP4 engine for Qwen3.8-Flash-Next (qwen4_exp) on AMD RDNA4 (gfx1201).
//
// All 85 compiled gfx1201 GPU kernels (~899 KiB total) are embedded directly into the binary via
// `embedded_kernels_gfx1201.cpp` and loaded via `hipModuleLoadData` (with an optional `--cache-dir` override
// for kernel developers). Weights are loaded directly from the NVFP4 `.safetensors` shards.

#include "strata/flydsl_nvfp4/engine.hpp"
#include "strata/flydsl_nvfp4/embedded_kernels_gfx1201.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/vmm.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <hip/hip_runtime_api.h>
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::flydsl_nvfp4 {
namespace {

#define HIP_CHECK(call)                                                                                     \
    do {                                                                                                    \
        hipError_t err__ = (call);                                                                          \
        if (err__ != hipSuccess) {                                                                          \
            std::fprintf(stderr, "HIP error at %s:%d: %s (%d)\n", __FILE__, __LINE__,                       \
                         hipGetErrorString(err__), (int) err__);                                            \
            std::exit(1);                                                                                   \
        }                                                                                                   \
    } while (0)

inline uint16_t f32_to_bf16_bits(float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

inline float bf16_bits_to_f32(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float v = 0.0f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

static std::string get_executable_dir() {
    char buf[4096]{};
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return ".";
    buf[n] = '\0';
    std::string p(buf);
    auto slash = p.find_last_of('/');
    return (slash == std::string::npos) ? "." : p.substr(0, slash);
}

// ====================================================================================================================
// 1. Embedded + Optional Disk-Override FlyDSL Kernel Loader (`FlyDslKernelCache`)
// ====================================================================================================================

struct LoadedKernel {
    hipModule_t mod = nullptr;
    hipFunction_t func = nullptr;
    std::string kernel_name;
    int bx = 256;
    int by = 1;
    int bz = 1;
    int gx = 1;
    int gy = 1;
    int gz = 1;
};

class FlyDslKernelCache {
public:
    explicit FlyDslKernelCache(std::string cache_dir)
        : cache_dir_(std::move(cache_dir)) {}

    ~FlyDslKernelCache() {
        for (auto& kv : kernels_) {
            if (kv.second.mod) {
                (void) hipModuleUnload(kv.second.mod);
            }
        }
    }

    const LoadedKernel& get(const std::string& key) {
        auto it = kernels_.find(key);
        if (it != kernels_.end()) {
            return it->second;
        }
        // 1. Optional developer override: if --cache-dir was explicitly passed and contains <key>.{hsaco,json}, use it
        if (!cache_dir_.empty()) {
            std::string hsaco_path = cache_dir_ + "/" + key + ".hsaco";
            std::string json_path = cache_dir_ + "/" + key + ".json";
            if (file_exists(hsaco_path) && file_exists(json_path)) {
                LoadedKernel lk = load_from_disk(key, hsaco_path, json_path);
                auto [ins_it, _] = kernels_.emplace(key, lk);
                return ins_it->second;
            }
        }
        // 2. Production path: load directly from the embedded gfx1201 HSACO table compiled into the binary
        const EmbeddedKernelDesc* emb = find_embedded_kernel(key);
        if (!emb) {
            std::fprintf(stderr, "[flydsl-nvfp4] Fatal: kernel '%s' not found in embedded gfx1201 kernel table!\n",
                         key.c_str());
            std::exit(1);
        }
        LoadedKernel lk;
        lk.kernel_name = emb->kernel_name;
        lk.bx = emb->bx;
        lk.by = emb->by;
        lk.bz = emb->bz;
        lk.gx = emb->gx;
        lk.gy = emb->gy;
        lk.gz = emb->gz;
        HIP_CHECK(hipModuleLoadData(&lk.mod, emb->hsaco_data));
        HIP_CHECK(hipModuleGetFunction(&lk.func, lk.mod, lk.kernel_name.c_str()));
        auto [ins_it, _] = kernels_.emplace(key, lk);
        return ins_it->second;
    }

    void launch(const std::string& key, int gx, int gy, int gz, void** args, hipStream_t stream = nullptr) {
        const LoadedKernel& lk = get(key);
        if (profile_enabled_) {
            (void) hipDeviceSynchronize();
            auto t0 = std::chrono::steady_clock::now();
            HIP_CHECK(hipModuleLaunchKernel(lk.func, gx, gy, gz, lk.bx, lk.by, lk.bz, 0, stream, args, nullptr));
            (void) hipDeviceSynchronize();
            double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            prof_us_[tag_ + key] += us;
            prof_cnt_[tag_ + key] += 1;
        } else {
            HIP_CHECK(hipModuleLaunchKernel(lk.func, gx, gy, gz, lk.bx, lk.by, lk.bz, 0, stream, args, nullptr));
        }
    }

    void launch_default(const std::string& key, void** args, hipStream_t stream = nullptr) {
        const LoadedKernel& lk = get(key);
        launch(key, lk.gx, lk.gy, lk.gz, args, stream);
    }

    void set_profile(bool en, std::string tag = "") {
        profile_enabled_ = en;
        tag_ = std::move(tag);
    }

    void dump_profile() const {
        if (prof_us_.empty()) return;
        std::vector<std::pair<std::string, double>> v(prof_us_.begin(), prof_us_.end());
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        double tot = 0.0;
        for (const auto& kv : v) tot += kv.second;
        std::printf("\n=== FlyDSL Per-Kernel Profile (Total: %.2f ms) ===\n", tot / 1000.0);
        for (const auto& kv : v) {
            int c = prof_cnt_.at(kv.first);
            std::printf("  %-36s : %8.2f ms (%4d calls, %6.1f us/call, %5.1f%%)\n",
                        kv.first.c_str(), kv.second / 1000.0, c, kv.second / c, 100.0 * kv.second / tot);
        }
    }

private:
    static bool file_exists(const std::string& p) {
        struct stat st {};
        return ::stat(p.c_str(), &st) == 0 && st.st_size > 0;
    }

    static LoadedKernel load_from_disk(const std::string& key, const std::string& hsaco_path,
                                       const std::string& json_path) {
        std::ifstream jf(json_path);
        if (!jf) {
            std::fprintf(stderr, "Failed to open metadata JSON: %s\n", json_path.c_str());
            std::exit(1);
        }
        std::string jtxt((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());
        LoadedKernel lk;

        // Parse "kernel_name": "..."
        auto kn_pos = jtxt.find("\"kernel_name\"");
        if (kn_pos == std::string::npos) {
            std::fprintf(stderr, "Invalid kernel JSON for %s\n", key.c_str());
            std::exit(1);
        }
        auto q1 = jtxt.find('"', jtxt.find(':', kn_pos) + 1);
        auto q2 = jtxt.find('"', q1 + 1);
        lk.kernel_name = jtxt.substr(q1 + 1, q2 - q1 - 1);

        auto parse_vec3 = [&](const char* field, int& x, int& y, int& z) {
            auto p = jtxt.find(field);
            if (p == std::string::npos) return;
            auto b1 = jtxt.find('[', p);
            auto b2 = jtxt.find(']', b1);
            std::string sub = jtxt.substr(b1 + 1, b2 - b1 - 1);
            std::sscanf(sub.c_str(), "%d, %d, %d", &x, &y, &z);
        };
        parse_vec3("\"block\"", lk.bx, lk.by, lk.bz);
        parse_vec3("\"default_grid\"", lk.gx, lk.gy, lk.gz);

        std::ifstream hf(hsaco_path, std::ios::binary);
        std::vector<char> hsaco((std::istreambuf_iterator<char>(hf)), std::istreambuf_iterator<char>());
        if (hsaco.empty()) {
            std::fprintf(stderr, "Empty HSACO file: %s\n", hsaco_path.c_str());
            std::exit(1);
        }
        HIP_CHECK(hipModuleLoadData(&lk.mod, hsaco.data()));
        HIP_CHECK(hipModuleGetFunction(&lk.func, lk.mod, lk.kernel_name.c_str()));
        return lk;
    }

    std::string cache_dir_;
    std::unordered_map<std::string, LoadedKernel> kernels_;
    bool profile_enabled_ = false;
    std::string tag_;
    std::unordered_map<std::string, double> prof_us_;
    std::unordered_map<std::string, int> prof_cnt_;
};

// ====================================================================================================================
// 2. Safetensors Header Parser & Pread Helper
// ====================================================================================================================

struct TensorSlice {
    int shard_idx = -1;
    uint64_t file_offset = 0;
    uint64_t byte_len = 0;
};

static void pread_exact(int fd, void* buf, size_t count, uint64_t offset) {
    uint8_t* dst = static_cast<uint8_t*>(buf);
    size_t done = 0;
    while (done < count) {
        ssize_t n = ::pread(fd, dst + done, count - done, static_cast<off_t>(offset + done));
        if (n <= 0) {
            std::fprintf(stderr, "pread failed at offset %llu (wanted %zu, got %zd)\n",
                         (unsigned long long) (offset + done), count - done, n);
            std::exit(1);
        }
        done += static_cast<size_t>(n);
    }
}

static void parse_safetensors_header(int fd, int shard_idx,
                                     std::unordered_map<std::string, TensorSlice>& index_out) {
    uint64_t hlen = 0;
    pread_exact(fd, &hlen, 8, 0);
    std::string hdr(hlen, '\0');
    pread_exact(fd, hdr.data(), hlen, 8);
    const uint64_t data_base = 8 + hlen;

    size_t pos = 0;
    while (pos < hdr.size()) {
        size_t k1 = hdr.find('"', pos);
        if (k1 == std::string::npos) break;
        size_t k2 = hdr.find('"', k1 + 1);
        if (k2 == std::string::npos) break;
        std::string key = hdr.substr(k1 + 1, k2 - k1 - 1);
        size_t colon = hdr.find(':', k2 + 1);
        if (colon == std::string::npos) break;
        size_t brace_open = hdr.find('{', colon);
        if (brace_open == std::string::npos) break;

        int depth = 1;
        size_t cur = brace_open + 1;
        while (cur < hdr.size() && depth > 0) {
            if (hdr[cur] == '{') depth++;
            else if (hdr[cur] == '}') depth--;
            cur++;
        }
        if (key != "__metadata__") {
            std::string body = hdr.substr(brace_open, cur - brace_open);
            size_t doff = body.find("\"data_offsets\"");
            if (doff != std::string::npos) {
                size_t b1 = body.find('[', doff);
                size_t b2 = body.find(']', b1);
                unsigned long long start = 0, end = 0;
                if (std::sscanf(body.substr(b1 + 1, b2 - b1 - 1).c_str(), "%llu,%llu", &start, &end) == 2) {
                    index_out[key] = TensorSlice{shard_idx, data_base + start, end - start};
                }
            }
        }
        pos = cur;
    }
}

// ====================================================================================================================
// 3. Qwen Byte-Level BPE Tokenizer (`vocab.json` + `merges.txt`)
// ====================================================================================================================

class QwenTokenizer {
public:
    bool load(const std::string& model_dir) {
        init_byte_tables();
        if (!load_vocab(model_dir + "/vocab.json")) return false;
        if (!load_merges(model_dir + "/merges.txt")) return false;
        // Register special tokens
        const std::pair<std::string, int32_t> specials[] = {
            {"<|endoftext|>", 248044},
            {"<|im_start|>", 248045},
            {"<|im_end|>", 248046},
            {"<tool_call>", 248058},
            {"</tool_call>", 248059},
            {"<tool_response>", 248066},
            {"</tool_response>", 248067},
            {"<think>", 248068},
            {"</think>", 248069},
            {"<|think|>", 248068},
            {"<|/think|>", 248069},
        };
        for (const auto& sp : specials) {
            vocab_[sp.first] = sp.second;
            if (id_to_raw_.find(sp.second) == id_to_raw_.end()) {
                id_to_raw_[sp.second] = sp.first;
            }
            specials_.push_back(sp);
        }
        return true;
    }

    std::vector<int32_t> encode(const std::string& text) const {
        std::vector<int32_t> out;
        size_t pos = 0;
        while (pos < text.size()) {
            // Check if any special token starts at or after pos
            size_t next_spec_pos = std::string::npos;
            size_t spec_idx = 0;
            for (size_t i = 0; i < specials_.size(); ++i) {
                size_t f = text.find(specials_[i].first, pos);
                if (f != std::string::npos && f < next_spec_pos) {
                    next_spec_pos = f;
                    spec_idx = i;
                }
            }
            if (next_spec_pos == pos) {
                out.push_back(specials_[spec_idx].second);
                pos += specials_[spec_idx].first.size();
                continue;
            }
            size_t chunk_end = (next_spec_pos == std::string::npos) ? text.size() : next_spec_pos;
            encode_chunk(text.substr(pos, chunk_end - pos), out);
            pos = chunk_end;
        }
        return out;
    }

    std::string decode_token(int32_t id) const {
        auto it = id_to_raw_.find(id);
        if (it != id_to_raw_.end()) return it->second;
        auto bit = id_to_bpe_.find(id);
        if (bit == id_to_bpe_.end()) return "";
        return bpe_str_to_bytes(bit->second);
    }

private:
    static std::string codepoint_to_utf8(int cp) {
        std::string s;
        if (cp <= 0x7F) {
            s.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            s.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            s.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        return s;
    }

    void init_byte_tables() {
        std::vector<int> bs;
        for (int b = 33; b <= 126; ++b) bs.push_back(b);
        for (int b = 161; b <= 172; ++b) bs.push_back(b);
        for (int b = 174; b <= 255; ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }
        for (size_t i = 0; i < 256; ++i) {
            std::string u = codepoint_to_utf8(cs[i]);
            byte_to_utf8_[bs[i]] = u;
            utf8_to_byte_[u] = static_cast<uint8_t>(bs[i]);
        }
    }

    std::string bpe_str_to_bytes(const std::string& bpe) const {
        std::string res;
        size_t i = 0;
        while (i < bpe.size()) {
            uint8_t c = static_cast<uint8_t>(bpe[i]);
            size_t clen = 1;
            if ((c & 0xE0) == 0xC0) clen = 2;
            else if ((c & 0xF0) == 0xE0) clen = 3;
            else if ((c & 0xF8) == 0xF0) clen = 4;
            std::string ch = bpe.substr(i, clen);
            auto it = utf8_to_byte_.find(ch);
            if (it != utf8_to_byte_.end()) {
                res.push_back(static_cast<char>(it->second));
            } else {
                res += ch;
            }
            i += clen;
        }
        return res;
    }

    static bool parse_json_string(const std::string& src, size_t& pos, std::string& out) {
        out.clear();
        if (pos >= src.size() || src[pos] != '"') return false;
        pos++;
        while (pos < src.size()) {
            char c = src[pos++];
            if (c == '"') return true;
            if (c == '\\' && pos < src.size()) {
                char esc = src[pos++];
                if (esc == '"' || esc == '\\' || esc == '/') out.push_back(esc);
                else if (esc == 'b') out.push_back('\b');
                else if (esc == 'f') out.push_back('\f');
                else if (esc == 'n') out.push_back('\n');
                else if (esc == 'r') out.push_back('\r');
                else if (esc == 't') out.push_back('\t');
                else if (esc == 'u' && pos + 4 <= src.size()) {
                    int cp = std::stoi(src.substr(pos, 4), nullptr, 16);
                    pos += 4;
                    out += codepoint_to_utf8(cp);
                }
            } else {
                out.push_back(c);
            }
        }
        return false;
    }

    bool load_vocab(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::string txt((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        size_t pos = txt.find('{');
        if (pos == std::string::npos) return false;
        pos++;
        std::string key;
        while (pos < txt.size()) {
            while (pos < txt.size() && (txt[pos] == ' ' || txt[pos] == '\n' || txt[pos] == '\r' || txt[pos] == '\t' || txt[pos] == ',')) pos++;
            if (pos >= txt.size() || txt[pos] == '}') break;
            if (!parse_json_string(txt, pos, key)) break;
            while (pos < txt.size() && (txt[pos] == ' ' || txt[pos] == ':')) pos++;
            size_t num_end = pos;
            while (num_end < txt.size() && (txt[num_end] >= '0' && txt[num_end] <= '9')) num_end++;
            int32_t id = std::stoi(txt.substr(pos, num_end - pos));
            vocab_[key] = id;
            id_to_bpe_[id] = key;
            pos = num_end;
        }
        return !vocab_.empty();
    }

    bool load_merges(const std::string& path) {
        std::ifstream f(path);
        if (!f) return false;
        std::string line;
        int rank = 0;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            std::string a = line.substr(0, sp);
            std::string b = line.substr(sp + 1);
            if (!b.empty() && b.back() == '\r') b.pop_back();
            merges_[a + "\x01" + b] = rank++;
        }
        return true;
    }

    // Split text into GPT-2/Qwen pre-tokenization words (contractions, letters, numbers, punctuation, whitespace)
    static std::vector<std::string> pretokenize(const std::string& s) {
        std::vector<std::string> words;
        size_t i = 0;
        auto is_letter = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (static_cast<uint8_t>(c) >= 128); };
        auto is_digit = [](char c) { return c >= '0' && c <= '9'; };
        while (i < s.size()) {
            if (s[i] == ' ' && i + 1 < s.size() && s[i + 1] != ' ' && s[i + 1] != '\n' && s[i + 1] != '\r') {
                size_t start = i++;
                if (is_letter(s[i])) {
                    while (i < s.size() && is_letter(s[i])) i++;
                } else if (is_digit(s[i])) {
                    // Qwen splits digits individually!
                    i++;
                } else {
                    while (i < s.size() && !is_letter(s[i]) && !is_digit(s[i]) && s[i] != ' ' && s[i] != '\n' && s[i] != '\r') i++;
                }
                words.push_back(s.substr(start, i - start));
            } else if (is_letter(s[i])) {
                size_t start = i;
                while (i < s.size() && is_letter(s[i])) i++;
                words.push_back(s.substr(start, i - start));
            } else if (is_digit(s[i])) {
                words.push_back(s.substr(i, 1));
                i++;
            } else if (s[i] == '\n' || s[i] == '\r') {
                size_t start = i;
                while (i < s.size() && (s[i] == '\n' || s[i] == '\r')) i++;
                words.push_back(s.substr(start, i - start));
            } else if (s[i] == ' ') {
                size_t start = i;
                while (i < s.size() && s[i] == ' ' && (i + 1 == s.size() || s[i + 1] == ' ')) i++;
                if (i == start) i++;
                words.push_back(s.substr(start, i - start));
            } else {
                size_t start = i;
                while (i < s.size() && !is_letter(s[i]) && !is_digit(s[i]) && s[i] != ' ' && s[i] != '\n' && s[i] != '\r') i++;
                words.push_back(s.substr(start, i - start));
            }
        }
        return words;
    }

    void encode_chunk(const std::string& chunk, std::vector<int32_t>& out) const {
        for (const std::string& w : pretokenize(chunk)) {
            std::vector<std::string> pieces;
            pieces.reserve(w.size());
            for (unsigned char b : w) {
                pieces.push_back(byte_to_utf8_[b]);
            }
            while (pieces.size() > 1) {
                int best_rank = std::numeric_limits<int>::max();
                size_t best_idx = 0;
                for (size_t i = 0; i + 1 < pieces.size(); ++i) {
                    auto it = merges_.find(pieces[i] + "\x01" + pieces[i + 1]);
                    if (it != merges_.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_idx = i;
                    }
                }
                if (best_rank == std::numeric_limits<int>::max()) break;
                pieces[best_idx] += pieces[best_idx + 1];
                pieces.erase(pieces.begin() + static_cast<ptrdiff_t>(best_idx + 1));
            }
            for (const std::string& p : pieces) {
                auto it = vocab_.find(p);
                if (it != vocab_.end()) {
                    out.push_back(it->second);
                }
            }
        }
    }

    std::string byte_to_utf8_[256];
    std::unordered_map<std::string, uint8_t> utf8_to_byte_;
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<int32_t, std::string> id_to_bpe_;
    std::unordered_map<int32_t, std::string> id_to_raw_;
    std::unordered_map<std::string, int> merges_;
    std::vector<std::pair<std::string, int32_t>> specials_;
};

// ====================================================================================================================
// 4. Qwen4Exp Model Weights & State Structures
// ====================================================================================================================

// Per-expert byte layout inside a layer's 1,415,577,600-byte NVFP4 expert arena
// (keeps each projection's weights and scales adjacent in the same 2 MB hugepage for CPU/GPU TLB locality):
inline constexpr size_t EXP_W_BYTES = 819200;    // [640, 1280] or [2560, 320] U8
inline constexpr size_t EXP_S_BYTES = 102400;    // [640, 160] or [2560, 40] F8_E4M3
inline constexpr size_t EXP_OFF_GATE_W = 0;
inline constexpr size_t EXP_OFF_GATE_S = 819200;
inline constexpr size_t EXP_OFF_UP_W = 921600;
inline constexpr size_t EXP_OFF_UP_S = 1740800;
inline constexpr size_t EXP_OFF_DOWN_W = 1843200;
inline constexpr size_t EXP_OFF_DOWN_S = 2662400;
inline constexpr size_t EXP_STRIDE_BYTES = 2764800;
inline constexpr size_t LAYER_EXPERT_BYTES = 512ULL * EXP_STRIDE_BYTES;   // 1,415,577,600 B

struct HcWeights {
    void* norm = nullptr;   // BF16 [10240]
    void* down = nullptr;   // BF16 [320, 10240]
    void* up = nullptr;     // BF16 [10240, 320]
    void* inj = nullptr;    // BF16 [4, 10240] (nullptr for final mixer)
    int64_t wi_delta = 0;
};

struct GdnWeights {
    void* in_proj_fused = nullptr; // BF16 [16480, 2560] = [qkv(10240) | z(6144) | a(48) | b(48)]
    void* out_proj = nullptr;      // BF16 [2560, 6144]
    void* conv1d = nullptr;        // BF16 [10240, 1, 4]
    void* A_log = nullptr;         // BF16 [48]
    void* dt_bias = nullptr;       // BF16 [48]
    void* norm = nullptr;          // BF16 [128]
};

struct QsaWeights {
    void* in_proj_fused = nullptr;   // BF16 [13952, 2560] = [q(12288) | k(512) | v(512) | idx_qk(640)]
    void* o_proj = nullptr;          // BF16 [2560, 6144]
    void* q_norm = nullptr;          // BF16 [256]
    void* k_norm = nullptr;          // BF16 [256]
    void* idx_q_ln = nullptr;        // BF16 [128]
    void* idx_k_ln = nullptr;        // BF16 [128]
};

struct MoeWeights {
    void* in_proj_fused = nullptr; // BF16 [1856, 2560] = [sh_gate(640) | sh_up(640) | router(512) | sh_sc(1) | pad(63)]
    void* sh_down = nullptr;       // BF16 [2560, 640]
    void* d_ptrs = nullptr;        // I64 [512, 6] on GPU (decode: non-zero for in_vram || in_gpu_cold)
    void* d_ptrs_prefill = nullptr;// I64 [512, 6] on GPU (prefill: non-zero ONLY for in_vram)
    void* d_s2 = nullptr;          // F32 [512, 3] on GPU
    int vram_experts = 0;          // Number of experts in this layer resident in VRAM

    // Zero-D2H Adaptive VRAM Expert Cache state (routing hot experts into VRAM and cold ones out to host RAM)
    int n_pinned = 0;
    int n_dyn = 0;
    uint8_t* vram_base = nullptr;
    uint8_t* dyn_vram_base = nullptr;
    uint8_t* host_cpu_ptr[512]{};
    uint8_t* host_dev_ptr[512]{};
    bool in_vram[512]{};
    bool in_gpu_cold[512]{};       // True if cold expert is assigned to GPU PCIe in decode (pcie_frac)
    bool in_vcache[512]{};         // True if top-ranked CPU expert is pinned in CCD 0's 96 MB 3D V-Cache
    int32_t prof_rank[512]{};      // Profile rank (lower = hotter) for CCD 0 3D V-Cache scheduling
    int16_t dyn_slot[512]{};       // -1 if pinned or in host RAM; 0..n_dyn-1 if in a dynamic VRAM slot
    int32_t slot_exp[64]{};        // Expert ID currently occupying dynamic slot s
    uint32_t slot_last_used[64]{}; // Step counter when dynamic slot s was last used
    uint32_t hit_count[512]{};     // Cumulative routing hits for expert e
    std::vector<int64_t> h_ptrs;   // Host mirror of d_ptrs [512 * 6]
    std::vector<int64_t> h_ptrs_prefill; // Host mirror of d_ptrs_prefill [512 * 6]
    std::vector<float> h_s2;       // Host copy of s2 [512 * 3] (gate_s2, up_s2, down_s2)
};

struct PleWeights {
    void* key_proj = nullptr;      // BF16 [10240, 2560]
    void* value_proj = nullptr;    // BF16 [2560, 2560]
    void* norm_key = nullptr;      // BF16 [10240]
    void* norm_query = nullptr;    // BF16 [10240]
    void* norm_conv = nullptr;     // BF16 [10240]
    void* conv_weight = nullptr;   // BF16 [10240, 1, 4]
    void* weight_scale = nullptr;  // BF16 [1]
    int ple_fd = -1;
    uint64_t shard_offsets[128]{};
};

struct LayerWeights {
    bool is_qsa = false;
    HcWeights attn_hc;
    HcWeights mlp_hc;
    GdnWeights gdn;
    QsaWeights qsa;
    MoeWeights moe;
};

// ====================================================================================================================
// 4b. AVX-512 VNNI + AVX2 Fallback NVFP4 CPU Expert Kernels & Lock-Free Thread Pool (Hybrid CPU+GPU MoE Execution)
// ====================================================================================================================

alignas(64) static float g_f8_e4m3_lut[256];
static bool g_f8_lut_inited = false;
static bool g_cpu_has_avx512vnni = false;

static void init_f8_e4m3_lut() {
    if (g_f8_lut_inited) return;
    for (int i = 0; i < 256; ++i) {
        int s = (i >> 7) & 1;
        int e = (i >> 3) & 0xF;
        int m = i & 0x7;
        float val = 0.0f;
        if (e == 0) {
            val = std::ldexp(static_cast<float>(m), -9);
        } else if (e == 15 && m == 7) {
            val = 0.0f;
        } else {
            val = std::ldexp(1.0f + static_cast<float>(m) * 0.125f, e - 7);
        }
        g_f8_e4m3_lut[i] = s ? -val : val;
    }
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    g_cpu_has_avx512vnni = __builtin_cpu_supports("avx512f") &&
                           __builtin_cpu_supports("avx512bw") &&
                           __builtin_cpu_supports("avx512vl") &&
                           __builtin_cpu_supports("avx512dq") &&
                           __builtin_cpu_supports("avx512vnni");
    if (const char* force_avx2 = std::getenv("STRATA_FORCE_AVX2")) {
        if (std::atoi(force_avx2) != 0) g_cpu_has_avx512vnni = false;
    }
#endif
    g_f8_lut_inited = true;
}

static constexpr int kMaxJobToks = 4;

struct CpuExpertJob {
    const uint8_t* exp_base = nullptr;
    float gate_s2 = 0.0f;
    float up_s2 = 0.0f;
    float down_s2 = 0.0f;
    float weight = 0.0f;
    int nt = 1;
    bool in_vcache = true;
    const int8_t* q_x[kMaxJobToks]{};
    const float* d_x[kMaxJobToks]{};
    const int32_t* neg_bias_x[kMaxJobToks]{};
    uint16_t* y_out[kMaxJobToks]{};
    float* y_f32 = nullptr;
    alignas(64) int8_t q_h[kMaxJobToks][640];
    alignas(64) float d_h[kMaxJobToks][40];
    alignas(64) int32_t neg_bias_h[kMaxJobToks][160];
};

// --------------------------------------------------------------------------------------------------------------------
// Target-isolated AVX-512 VNNI kernels (no STL template instantiations inside this region to prevent COMDAT leakage!)
// --------------------------------------------------------------------------------------------------------------------
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,fma,f16c")
#endif

// Unpack 32 ModelOpt NVFP4 bytes (low nibble = 2*i, high nibble = 2*i+1) into 64 unsigned u8 codes (2*val + 12)
static inline __m512i codes64_modelopt(const uint8_t* wp, __m512i lut, __m512i mask) {
    __m512i w16 = _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(wp)));
    __m512i nibbles = _mm512_and_si512(_mm512_or_si512(w16, _mm512_slli_epi16(w16, 4)), mask);
    return _mm512_shuffle_epi8(lut, nibbles);
}

// Quantize BF16 [2560] activation into int8 [2560], float [160] (0.5 * amax / 127), and int32 [640] (-12 * sum4(q))
static void quantize_act_bf16_2560_avx512(const uint16_t* x_bf16, int8_t* q_x, float* d_x, int32_t* neg_bias_x) {
    const __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    const __m512i zero = _mm512_setzero_si512();
    const __m512i c12 = _mm512_set1_epi8(12);
    for (int b = 0; b < 160; ++b) {
        __m256i bf = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_bf16 + 16 * b));
        __m512 xv = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(bf), 16));
        float amax = _mm512_reduce_max_ps(_mm512_and_ps(xv, sign_mask));
        if (amax == 0.0f) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(q_x + 16 * b), _mm_setzero_si128());
            d_x[b] = 0.0f;
        } else {
            __m512 inv = _mm512_set1_ps(127.0f / amax);
            __m512i q32 = _mm512_cvtps_epi32(_mm512_mul_ps(xv, inv));
            __m128i q8 = _mm512_cvtepi32_epi8(q32);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(q_x + 16 * b), q8);
            d_x[b] = amax * (0.5f / 127.0f);
        }
    }
    for (int b4 = 0; b4 < 160; b4 += 4) {
        __m512i qv = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q_x + 16 * b4));
        __m512i nb = _mm512_sub_epi32(zero, _mm512_dpbusd_epi32(zero, c12, qv));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(neg_bias_x + 4 * b4), nb);
    }
}

// Quantize 32 SwiGLU intermediate floats into int8 [32], float [2] (0.5 * amax / 127), and int32 [8] (-12 * sum4(q))
static inline void quantize_act_f32_32_avx512(const float* h32, int8_t* q_h, float* d_h, int32_t* neg_bias_h) {
    const __m512 sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    for (int sub = 0; sub < 2; ++sub) {
        __m512 xv = _mm512_loadu_ps(h32 + 16 * sub);
        float amax = _mm512_reduce_max_ps(_mm512_and_ps(xv, sign_mask));
        if (amax == 0.0f) {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(q_h + 16 * sub), _mm_setzero_si128());
            d_h[sub] = 0.0f;
        } else {
            __m512 inv = _mm512_set1_ps(127.0f / amax);
            __m512i q32 = _mm512_cvtps_epi32(_mm512_mul_ps(xv, inv));
            __m128i q8 = _mm512_cvtepi32_epi8(q32);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(q_h + 16 * sub), q8);
            d_h[sub] = amax * (0.5f / 127.0f);
        }
    }
    const __m256i zero = _mm256_setzero_si256();
    const __m256i c12 = _mm256_set1_epi8(12);
    __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_h));
    __m256i nb = _mm256_sub_epi32(zero, _mm256_dpbusd_epi32(zero, c12, qv));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(neg_bias_h), nb);
}

// Vectorized 8-byte float8_e4m3fn -> 8x float32 (__m256) decoder in pure SIMD registers (zero LUT memory loads)
static inline __m256 decode_f8_e4m3_8(const uint8_t* sp) {
    __m128i raw8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(sp));
    __m256i b32 = _mm256_cvtepu8_epi32(raw8);
    __m256i sign = _mm256_slli_epi32(_mm256_and_si256(b32, _mm256_set1_epi32(0x80)), 24);
    __m256i exp = _mm256_and_si256(_mm256_srli_epi32(b32, 3), _mm256_set1_epi32(15));
    __m256i man = _mm256_and_si256(b32, _mm256_set1_epi32(7));
    __m256i norm_i = _mm256_or_si256(_mm256_slli_epi32(_mm256_add_epi32(exp, _mm256_set1_epi32(120)), 23),
                                     _mm256_slli_epi32(man, 20));
    __m256 sub = _mm256_mul_ps(_mm256_cvtepi32_ps(man), _mm256_set1_ps(0.001953125f));
    __mmask8 is_sub = _mm256_cmpeq_epi32_mask(exp, _mm256_setzero_si256());
    __m256 mag = _mm256_mask_blend_ps(is_sub, _mm256_castsi256_ps(norm_i), sub);
    return _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(mag), sign));
}

// Stream a single NVFP4 row (NB * 16 elements) in 64-byte cache-line steps (b += 8) with 2 independent FMA
// accumulators (acc0, acc1) and register-resident decode_f8_e4m3_8, hiding Zen 4's 4-cycle FMA latency while
// keeping 1 single linear DRAM stream per core for maximum L1/L2 hardware prefetcher efficiency.
template <int NT, int NB>
static inline void nvfp4_row_dot_avx512(const uint8_t* wp, const uint8_t* sp,
                                        const int8_t* const* q_x, const float* const* d_x,
                                        const int32_t* const* neg_bias_x, float* out_f32,
                                        bool /*temporal*/ = true) {
    static_assert(NB % 8 == 0, "NB must be a multiple of 8");
    const __m512i lut = _mm512_broadcast_i32x4(_mm_setr_epi8(
        12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
    const __m512i mask = _mm512_set1_epi8(0x0F);
    const __m512i expand0 = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    const __m512i expand1 = _mm512_set_epi32(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4);

    __m512 acc0[NT];
    __m512 acc1[NT];
    for (int t = 0; t < NT; ++t) {
        acc0[t] = _mm512_setzero_ps();
        acc1[t] = _mm512_setzero_ps();
    }

    for (int b = 0; b < NB; b += 8) {
        _mm_prefetch(reinterpret_cast<const char*>(wp + 8 * b + 1024), _MM_HINT_T0);
        __m512i c0 = codes64_modelopt(wp + 8 * b, lut, mask);
        __m512i c1 = codes64_modelopt(wp + 8 * b + 32, lut, mask);
        __m256 ws8 = decode_f8_e4m3_8(sp + b);
        for (int t = 0; t < NT; ++t) {
            __m512 sc8_512 = _mm512_castps256_ps512(_mm256_mul_ps(ws8, _mm256_loadu_ps(d_x[t] + b)));
            __m512i qv0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q_x[t] + 16 * b));
            __m512i nb0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(neg_bias_x[t] + 4 * b));
            __m512i raw0 = _mm512_dpbusd_epi32(nb0, c0, qv0);
            __m512 sc16_0 = _mm512_permutexvar_ps(expand0, sc8_512);
            acc0[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(raw0), sc16_0, acc0[t]);

            __m512i qv1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(q_x[t] + 16 * b + 64));
            __m512i nb1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(neg_bias_x[t] + 4 * b + 16));
            __m512i raw1 = _mm512_dpbusd_epi32(nb1, c1, qv1);
            __m512 sc16_1 = _mm512_permutexvar_ps(expand1, sc8_512);
            acc1[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(raw1), sc16_1, acc1[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        out_f32[t] = _mm512_reduce_add_ps(_mm512_add_ps(acc0[t], acc1[t]));
    }
}

template <int NT>
static void gu_tile32_nt_avx512(CpuExpertJob& jb, int r0) {
    const uint8_t* gw_base = jb.exp_base + EXP_OFF_GATE_W + static_cast<size_t>(r0) * 1280;
    const uint8_t* gs_base = jb.exp_base + EXP_OFF_GATE_S + static_cast<size_t>(r0) * 160;
    const uint8_t* uw_base = jb.exp_base + EXP_OFF_UP_W + static_cast<size_t>(r0) * 1280;
    const uint8_t* us_base = jb.exp_base + EXP_OFF_UP_S + static_cast<size_t>(r0) * 160;

    alignas(64) float g_all[32][NT];
    for (int r = 0; r < 32; ++r) {
        nvfp4_row_dot_avx512<NT, 160>(gw_base + static_cast<size_t>(r) * 1280, gs_base + static_cast<size_t>(r) * 160,
                                      jb.q_x, jb.d_x, jb.neg_bias_x, g_all[r]);
    }

    alignas(64) float h32[NT][32];
    for (int r = 0; r < 32; ++r) {
        float u_f32[NT];
        nvfp4_row_dot_avx512<NT, 160>(uw_base + static_cast<size_t>(r) * 1280, us_base + static_cast<size_t>(r) * 160,
                                      jb.q_x, jb.d_x, jb.neg_bias_x, u_f32);
        for (int t = 0; t < NT; ++t) {
            float gf = bf16_bits_to_f32(f32_to_bf16_bits(g_all[r][t] * jb.gate_s2));
            float uf = bf16_bits_to_f32(f32_to_bf16_bits(u_f32[t] * jb.up_s2));
            float silu_g = bf16_bits_to_f32(f32_to_bf16_bits(gf / (1.0f + std::exp(-gf))));
            h32[t][r] = bf16_bits_to_f32(f32_to_bf16_bits(silu_g * uf));
        }
    }
    int b0 = r0 / 16;
    for (int t = 0; t < NT; ++t) {
        quantize_act_f32_32_avx512(h32[t], jb.q_h[t] + r0, jb.d_h[t] + b0, jb.neg_bias_h[t] + 4 * b0);
    }
}

template <int NT>
static void down_tile80_nt_avx512(CpuExpertJob& jb, int r0) {
    const uint8_t* dw_base = jb.exp_base + EXP_OFF_DOWN_W + static_cast<size_t>(r0) * 320;
    const uint8_t* ds_base = jb.exp_base + EXP_OFF_DOWN_S + static_cast<size_t>(r0) * 40;

    const int8_t* q_h_ptrs[NT];
    const float* d_h_ptrs[NT];
    const int32_t* nb_h_ptrs[NT];
    for (int t = 0; t < NT; ++t) {
        q_h_ptrs[t] = jb.q_h[t];
        d_h_ptrs[t] = jb.d_h[t];
        nb_h_ptrs[t] = jb.neg_bias_h[t];
    }

    for (int r = 0; r < 80; ++r) {
        float d_f32[NT];
        nvfp4_row_dot_avx512<NT, 40>(dw_base + static_cast<size_t>(r) * 320, ds_base + static_cast<size_t>(r) * 40,
                                     q_h_ptrs, d_h_ptrs, nb_h_ptrs, d_f32);
        for (int t = 0; t < NT; ++t) {
            jb.y_out[t][r0 + r] = f32_to_bf16_bits(d_f32[t] * jb.down_s2);
        }
    }
}

static void down_tile80_decode_avx512(const CpuExpertJob& jb, int r0) {
    const uint8_t* dw_base = jb.exp_base + EXP_OFF_DOWN_W + static_cast<size_t>(r0) * 320;
    const uint8_t* ds_base = jb.exp_base + EXP_OFF_DOWN_S + static_cast<size_t>(r0) * 40;
    const int8_t* q_h_ptr = jb.q_h[0];
    const float* d_h_ptr = jb.d_h[0];
    const int32_t* nb_h_ptr = jb.neg_bias_h[0];
    const float sd = jb.down_s2;
    const float w = jb.weight;
    float* dst = jb.y_f32 + r0;
    for (int r = 0; r < 80; ++r) {
        float d_f32 = 0.0f;
        nvfp4_row_dot_avx512<1, 40>(dw_base + static_cast<size_t>(r) * 320,
                                    ds_base + static_cast<size_t>(r) * 40,
                                    &q_h_ptr, &d_h_ptr, &nb_h_ptr, &d_f32);
        float y_bf = bf16_bits_to_f32(f32_to_bf16_bits(d_f32 * sd));
        dst[r] = bf16_bits_to_f32(f32_to_bf16_bits(y_bf * w));
    }
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC pop_options
#endif

// --------------------------------------------------------------------------------------------------------------------
// Target-isolated AVX2 + FMA fallback kernels (for non-AVX-512 CPUs)
// --------------------------------------------------------------------------------------------------------------------
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC push_options
#pragma GCC target("avx2,fma,f16c")
#endif

static void quantize_act_bf16_2560_avx2(const uint16_t* x_bf16, int8_t* q_x, float* d_x, int32_t* neg_bias_x) {
    for (int b = 0; b < 160; ++b) {
        float xv[16];
        float amax = 0.0f;
        for (int i = 0; i < 16; ++i) {
            xv[i] = bf16_bits_to_f32(x_bf16[16 * b + i]);
            float a = std::fabs(xv[i]);
            if (a > amax) amax = a;
        }
        if (amax == 0.0f) {
            std::memset(q_x + 16 * b, 0, 16);
            d_x[b] = 0.0f;
            for (int g = 0; g < 4; ++g) neg_bias_x[4 * b + g] = 0;
        } else {
            float inv = 127.0f / amax;
            d_x[b] = amax * (0.5f / 127.0f);
            for (int g = 0; g < 4; ++g) {
                int32_t sum4 = 0;
                for (int j = 0; j < 4; ++j) {
                    int32_t q = static_cast<int32_t>(std::nearbyint(xv[4 * g + j] * inv));
                    q = std::clamp(q, -127, 127);
                    q_x[16 * b + 4 * g + j] = static_cast<int8_t>(q);
                    sum4 += q;
                }
                neg_bias_x[4 * b + g] = -12 * sum4;
            }
        }
    }
}

static inline void quantize_act_f32_32_avx2(const float* h32, int8_t* q_h, float* d_h, int32_t* neg_bias_h) {
    for (int sub = 0; sub < 2; ++sub) {
        const float* xv = h32 + 16 * sub;
        float amax = 0.0f;
        for (int i = 0; i < 16; ++i) {
            float a = std::fabs(xv[i]);
            if (a > amax) amax = a;
        }
        if (amax == 0.0f) {
            std::memset(q_h + 16 * sub, 0, 16);
            d_h[sub] = 0.0f;
            for (int g = 0; g < 4; ++g) neg_bias_h[4 * sub + g] = 0;
        } else {
            float inv = 127.0f / amax;
            d_h[sub] = amax * (0.5f / 127.0f);
            for (int g = 0; g < 4; ++g) {
                int32_t sum4 = 0;
                for (int j = 0; j < 4; ++j) {
                    int32_t q = static_cast<int32_t>(std::nearbyint(xv[4 * g + j] * inv));
                    q = std::clamp(q, -127, 127);
                    q_h[16 * sub + 4 * g + j] = static_cast<int8_t>(q);
                    sum4 += q;
                }
                neg_bias_h[4 * sub + g] = -12 * sum4;
            }
        }
    }
}

static inline __m256i codes32_modelopt_avx2(const uint8_t* wp, __m256i lut, __m256i mask) {
    __m256i w16 = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wp)));
    __m256i nibbles = _mm256_and_si256(_mm256_or_si256(w16, _mm256_slli_epi16(w16, 4)), mask);
    return _mm256_shuffle_epi8(lut, nibbles);
}

static inline __m256i dpbusd_avx2(__m256i acc, __m256i u8_a, __m256i s8_b, __m256i ones16) {
    __m256i prod16 = _mm256_maddubs_epi16(u8_a, s8_b);
    __m256i prod32 = _mm256_madd_epi16(prod16, ones16);
    return _mm256_add_epi32(acc, prod32);
}

template <int NT, int NB>
static inline void nvfp4_row_dot_avx2(const uint8_t* wp, const uint8_t* sp,
                                      const int8_t* const* q_x, const float* const* d_x,
                                      const int32_t* const* neg_bias_x, float* out_f32) {
    static_assert(NB % 8 == 0, "NB must be a multiple of 8");
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
    const __m256i mask = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i exp_lo = _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0);
    const __m256i exp_hi = _mm256_set_epi32(3, 3, 3, 3, 2, 2, 2, 2);

    __m256 acc0[NT];
    __m256 acc1[NT];
    for (int t = 0; t < NT; ++t) {
        acc0[t] = _mm256_setzero_ps();
        acc1[t] = _mm256_setzero_ps();
    }

    for (int b = 0; b < NB; b += 4) {
        __m256i c0 = codes32_modelopt_avx2(wp + 8 * b, lut, mask);
        __m256i c1 = codes32_modelopt_avx2(wp + 8 * b + 16, lut, mask);
        float ws4[4] = {
            g_f8_e4m3_lut[sp[b + 0]],
            g_f8_e4m3_lut[sp[b + 1]],
            g_f8_e4m3_lut[sp[b + 2]],
            g_f8_e4m3_lut[sp[b + 3]],
        };
        __m128 ws_v = _mm_loadu_ps(ws4);
        for (int t = 0; t < NT; ++t) {
            __m128 sc4 = _mm_mul_ps(ws_v, _mm_loadu_ps(d_x[t] + b));
            __m256 sc4_256 = _mm256_castps128_ps256(sc4);

            __m256i qv0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_x[t] + 16 * b));
            __m256i nb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(neg_bias_x[t] + 4 * b));
            __m256i raw0 = dpbusd_avx2(nb0, c0, qv0, ones16);
            __m256 sc8_0 = _mm256_permutevar8x32_ps(sc4_256, exp_lo);
            acc0[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(raw0), sc8_0, acc0[t]);

            __m256i qv1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q_x[t] + 16 * b + 32));
            __m256i nb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(neg_bias_x[t] + 4 * b + 8));
            __m256i raw1 = dpbusd_avx2(nb1, c1, qv1, ones16);
            __m256 sc8_1 = _mm256_permutevar8x32_ps(sc4_256, exp_hi);
            acc1[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(raw1), sc8_1, acc1[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        __m256 sum8 = _mm256_add_ps(acc0[t], acc1[t]);
        __m128 sum4 = _mm_add_ps(_mm256_castps256_ps128(sum8), _mm256_extractf128_ps(sum8, 1));
        __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
        out_f32[t] = _mm_cvtss_f32(sum1);
    }
}

template <int NT>
static void gu_tile32_nt_avx2(CpuExpertJob& jb, int r0) {
    const uint8_t* gw_base = jb.exp_base + EXP_OFF_GATE_W + static_cast<size_t>(r0) * 1280;
    const uint8_t* gs_base = jb.exp_base + EXP_OFF_GATE_S + static_cast<size_t>(r0) * 160;
    const uint8_t* uw_base = jb.exp_base + EXP_OFF_UP_W + static_cast<size_t>(r0) * 1280;
    const uint8_t* us_base = jb.exp_base + EXP_OFF_UP_S + static_cast<size_t>(r0) * 160;

    alignas(64) float g_all[32][NT];
    for (int r = 0; r < 32; ++r) {
        nvfp4_row_dot_avx2<NT, 160>(gw_base + static_cast<size_t>(r) * 1280, gs_base + static_cast<size_t>(r) * 160,
                                    jb.q_x, jb.d_x, jb.neg_bias_x, g_all[r]);
    }

    alignas(64) float h32[NT][32];
    for (int r = 0; r < 32; ++r) {
        float u_f32[NT];
        nvfp4_row_dot_avx2<NT, 160>(uw_base + static_cast<size_t>(r) * 1280, us_base + static_cast<size_t>(r) * 160,
                                    jb.q_x, jb.d_x, jb.neg_bias_x, u_f32);
        for (int t = 0; t < NT; ++t) {
            float gf = bf16_bits_to_f32(f32_to_bf16_bits(g_all[r][t] * jb.gate_s2));
            float uf = bf16_bits_to_f32(f32_to_bf16_bits(u_f32[t] * jb.up_s2));
            float silu_g = bf16_bits_to_f32(f32_to_bf16_bits(gf / (1.0f + std::exp(-gf))));
            h32[t][r] = bf16_bits_to_f32(f32_to_bf16_bits(silu_g * uf));
        }
    }
    int b0 = r0 / 16;
    for (int t = 0; t < NT; ++t) {
        quantize_act_f32_32_avx2(h32[t], jb.q_h[t] + r0, jb.d_h[t] + b0, jb.neg_bias_h[t] + 4 * b0);
    }
}

template <int NT>
static void down_tile80_nt_avx2(CpuExpertJob& jb, int r0) {
    const uint8_t* dw_base = jb.exp_base + EXP_OFF_DOWN_W + static_cast<size_t>(r0) * 320;
    const uint8_t* ds_base = jb.exp_base + EXP_OFF_DOWN_S + static_cast<size_t>(r0) * 40;

    const int8_t* q_h_ptrs[NT];
    const float* d_h_ptrs[NT];
    const int32_t* nb_h_ptrs[NT];
    for (int t = 0; t < NT; ++t) {
        q_h_ptrs[t] = jb.q_h[t];
        d_h_ptrs[t] = jb.d_h[t];
        nb_h_ptrs[t] = jb.neg_bias_h[t];
    }

    for (int r = 0; r < 80; ++r) {
        float d_f32[NT];
        nvfp4_row_dot_avx2<NT, 40>(dw_base + static_cast<size_t>(r) * 320, ds_base + static_cast<size_t>(r) * 40,
                                   q_h_ptrs, d_h_ptrs, nb_h_ptrs, d_f32);
        for (int t = 0; t < NT; ++t) {
            jb.y_out[t][r0 + r] = f32_to_bf16_bits(d_f32[t] * jb.down_s2);
        }
    }
}

static void down_tile80_decode_avx2(const CpuExpertJob& jb, int r0) {
    const uint8_t* dw_base = jb.exp_base + EXP_OFF_DOWN_W + static_cast<size_t>(r0) * 320;
    const uint8_t* ds_base = jb.exp_base + EXP_OFF_DOWN_S + static_cast<size_t>(r0) * 40;
    const int8_t* q_h_ptr = jb.q_h[0];
    const float* d_h_ptr = jb.d_h[0];
    const int32_t* nb_h_ptr = jb.neg_bias_h[0];
    const float sd = jb.down_s2;
    const float w = jb.weight;
    float* dst = jb.y_f32 + r0;
    for (int r = 0; r < 80; ++r) {
        float d_f32 = 0.0f;
        nvfp4_row_dot_avx2<1, 40>(dw_base + static_cast<size_t>(r) * 320,
                                  ds_base + static_cast<size_t>(r) * 40,
                                  &q_h_ptr, &d_h_ptr, &nb_h_ptr, &d_f32);
        float y_bf = bf16_bits_to_f32(f32_to_bf16_bits(d_f32 * sd));
        dst[r] = bf16_bits_to_f32(f32_to_bf16_bits(y_bf * w));
    }
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#pragma GCC pop_options
#endif

static void quantize_act_bf16_2560(const uint16_t* x_bf16, int8_t* q_x, float* d_x, int32_t* neg_bias_x) {
    if (g_cpu_has_avx512vnni) {
        quantize_act_bf16_2560_avx512(x_bf16, q_x, d_x, neg_bias_x);
    } else {
        quantize_act_bf16_2560_avx2(x_bf16, q_x, d_x, neg_bias_x);
    }
}

static void dispatch_gu_tile32(CpuExpertJob& jb, int r0) {
    if (g_cpu_has_avx512vnni) {
        switch (jb.nt) {
            case 1: gu_tile32_nt_avx512<1>(jb, r0); break;
            case 2: gu_tile32_nt_avx512<2>(jb, r0); break;
            case 3: gu_tile32_nt_avx512<3>(jb, r0); break;
            default: gu_tile32_nt_avx512<4>(jb, r0); break;
        }
    } else {
        switch (jb.nt) {
            case 1: gu_tile32_nt_avx2<1>(jb, r0); break;
            case 2: gu_tile32_nt_avx2<2>(jb, r0); break;
            case 3: gu_tile32_nt_avx2<3>(jb, r0); break;
            default: gu_tile32_nt_avx2<4>(jb, r0); break;
        }
    }
}

static void dispatch_down_tile80(CpuExpertJob& jb, int r0) {
    if (g_cpu_has_avx512vnni) {
        switch (jb.nt) {
            case 1: down_tile80_nt_avx512<1>(jb, r0); break;
            case 2: down_tile80_nt_avx512<2>(jb, r0); break;
            case 3: down_tile80_nt_avx512<3>(jb, r0); break;
            default: down_tile80_nt_avx512<4>(jb, r0); break;
        }
    } else {
        switch (jb.nt) {
            case 1: down_tile80_nt_avx2<1>(jb, r0); break;
            case 2: down_tile80_nt_avx2<2>(jb, r0); break;
            case 3: down_tile80_nt_avx2<3>(jb, r0); break;
            default: down_tile80_nt_avx2<4>(jb, r0); break;
        }
    }
}

static void dispatch_down_tile80_decode(const CpuExpertJob& jb, int r0) {
    if (g_cpu_has_avx512vnni) {
        down_tile80_decode_avx512(jb, r0);
    } else {
        down_tile80_decode_avx2(jb, r0);
    }
}

class Nvfp4CpuPool {
public:
    Nvfp4CpuPool() = default;
    ~Nvfp4CpuPool() { stop(); }

    static int ccd_of_cpu(int cpu) {
        // On Ryzen 9 7950X3D, CCD 0 (96 MB 3D V-Cache) is CPUs 0..7 and SMT siblings 16..23;
        // CCD 1 (32 MB L3) is CPUs 8..15 and SMT siblings 24..31.
        return ((cpu >= 0 && cpu < 8) || (cpu >= 16 && cpu < 24)) ? 0 : 1;
    }

    void init() {
        if (inited_) return;
        init_f8_e4m3_lut();
        std::vector<int> cores;
        const char* mode = std::getenv("STRATA_NVFP4_CPU_MODE");
        if (mode && std::string(mode) == "ccd0_16") {
            for (int c = 0; c < 8; ++c) cores.push_back(c);
            for (int c = 16; c < 24; ++c) cores.push_back(c);
        } else if (mode && std::string(mode) == "ccd0_8") {
            for (int c = 0; c < 8; ++c) cores.push_back(c);
        } else if (mode && std::string(mode) == "x3d_24") {
            for (int c = 0; c < 8; ++c) cores.push_back(c);
            for (int c = 16; c < 24; ++c) cores.push_back(c);
            for (int c = 8; c < 16; ++c) cores.push_back(c);
        } else {
            // Default: 16 physical cores (1 thread per 512-bit AVX-512 unit), sorted so CCD 0 (96 MB 3D V-Cache, CPUs 0..7) comes first!
            cores = strata::kernels::cpu::physical_cores(false);
            std::stable_sort(cores.begin(), cores.end(), [](int a, int b) {
                return ccd_of_cpu(a) < ccd_of_cpu(b);
            });
        }
        int n = static_cast<int>(cores.size());
        if (n <= 0) n = 16;
        if (const char* env_t = std::getenv("STRATA_NVFP4_THREADS")) {
            int req = std::atoi(env_t);
            if (req >= 1 && req <= 32) n = req;
        }
        n_threads_ = n;
        n_ccd0_ = 0;
        n_ccd1_ = 0;
        for (int t = 0; t < n_threads_; ++t) {
            int cpu = (t < static_cast<int>(cores.size())) ? cores[t] : t;
            if (ccd_of_cpu(cpu) == 0) n_ccd0_++;
            else n_ccd1_++;
        }
        if (!cores.empty()) {
            strata::kernels::cpu::pin_current_thread(cores[0]);
        }
        workers_.reserve(n_threads_ - 1);
        for (int t = 1; t < n_threads_; ++t) {
            int cpu = (t < static_cast<int>(cores.size())) ? cores[t] : -1;
            int ccd = ccd_of_cpu(cpu >= 0 ? cpu : t);
            workers_.emplace_back([this, cpu, ccd]() { worker_loop(cpu, ccd); });
        }
        inited_ = true;
    }

    void stop() {
        if (!inited_) return;
        stop_.store(true, std::memory_order_release);
        gen_.fetch_add(1, std::memory_order_acq_rel);
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
        workers_.clear();
        inited_ = false;
    }

    void run_batch(CpuExpertJob* jobs, int n_jobs) {
        if (n_jobs <= 0) return;
        jobs_ = jobs;
        n_jobs_ = n_jobs;

        // Phase 1: 20 tiles of 32 gate+up rows per expert
        launch_phase(1, n_jobs, 20);

        // Phase 2: 32 tiles of 80 down rows per expert
        launch_phase(2, n_jobs, 32);
    }

    // Decode (M=1) batch runner: Phase 1 computes gate+up across all n_jobs experts;
    // Phase 4 computes 32 tiles of 80 down rows per expert and pre-combines into y_accum_f32 [2560]!
    void run_decode_batch(CpuExpertJob* jobs, int n_jobs, float* y_accum_f32) {
        if (n_jobs <= 0) return;
        jobs_ = jobs;
        n_jobs_ = n_jobs;

        // Phase 1: 20 tiles of 32 gate+up rows per expert
        launch_phase(1, n_jobs, 20);

        // Phase 4: 32 tiles of 80 down rows per expert (writing pre-weighted float32 into jb.y_f32)
        launch_phase(4, n_jobs, 32);

        if (n_jobs > 1) {
            const float* src0 = jobs[0].y_f32;
            for (int c = 0; c < 2560; ++c) {
                float sum = src0[c];
                for (int j = 1; j < n_jobs; ++j) {
                    sum += jobs[j].y_f32[c];
                }
                y_accum_f32[c] = sum;
            }
        }
    }

    // Parallel Queue-Depth-16 NVMe reader for PLE 160-byte rows across all 16 persistent threads
    void run_ple_reads(int fd, const uint64_t* offsets, uint8_t* const* dsts, int n_reads) {
        if (n_reads <= 0) return;
        ple_fd_ = fd;
        ple_offsets_ = offsets;
        ple_dsts_ = dsts;
        launch_phase(5, n_reads, 1);
    }

    int num_threads() const { return n_threads_; }
    int num_ccd0_threads() const { return n_ccd0_; }
    int num_ccd1_threads() const { return n_ccd1_; }

private:
    void launch_phase(int ph, int n_jobs, int tiles_per_job) {
        const int total = n_jobs * tiles_per_job;
        n_tasks_ = total;
        phase_ = ph;
        next_task_ccd0_.store(0, std::memory_order_relaxed);
        done_cnt_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_acq_rel);
        do_work(0);
        while (done_cnt_.load(std::memory_order_acquire) < n_threads_ - 1) {
            _mm_pause();
        }
    }

    void do_work(int /*ccd*/) {
        const int total = n_tasks_;
        const int ph = phase_;
        if (ph == 5) {
            const int fd = ple_fd_;
            const uint64_t* offs = ple_offsets_;
            uint8_t* const* dsts = ple_dsts_;
            while (true) {
                int idx = next_task_ccd0_.fetch_add(1, std::memory_order_relaxed);
                if (idx >= total) break;
                pread_exact(fd, dsts[idx], 160, offs[idx]);
            }
            return;
        }
        CpuExpertJob* jobs = jobs_;
        while (true) {
            int idx = next_task_ccd0_.fetch_add(1, std::memory_order_relaxed);
            if (idx >= total) break;
            if (ph == 1) {
                int j = idx / 20;
                int r0 = (idx % 20) * 32;
                dispatch_gu_tile32(jobs[j], r0);
            } else if (ph == 2) {
                int j = idx / 32;
                int r0 = (idx % 32) * 80;
                dispatch_down_tile80(jobs[j], r0);
            } else {
                int j = idx / 32;
                int r0 = (idx % 32) * 80;
                dispatch_down_tile80_decode(jobs[j], r0);
            }
        }
    }

    void worker_loop(int cpu, int ccd) {
        if (cpu >= 0) {
            strata::kernels::cpu::pin_current_thread(cpu);
        }
        uint32_t my_gen = 0;
        while (true) {
            uint32_t g = gen_.load(std::memory_order_acquire);
            if (g == my_gen) {
                _mm_pause();
                continue;
            }
            if (stop_.load(std::memory_order_acquire)) break;
            my_gen = g;
            do_work(ccd);
            done_cnt_.fetch_add(1, std::memory_order_release);
        }
    }

    bool inited_ = false;
    int n_threads_ = 1;
    int n_ccd0_ = 1;
    int n_ccd1_ = 0;
    std::vector<std::thread> workers_;
    alignas(64) std::atomic<uint32_t> gen_{0};
    alignas(64) std::atomic<int> next_task_ccd0_{0};
    alignas(64) std::atomic<int> next_task_ccd1_{0};
    alignas(64) std::atomic<int> done_cnt_{0};
    std::atomic<bool> stop_{false};
    CpuExpertJob* jobs_ = nullptr;
    int ple_fd_ = -1;
    const uint64_t* ple_offsets_ = nullptr;
    uint8_t* const* ple_dsts_ = nullptr;
    int n_jobs_ = 0;
    int n_tasks_ = 0;
    int ccd0_limit_ = 0;
    int ccd1_limit_ = 0;
    int phase_ = 0;
};

struct SamplingParams {
    float temperature = 0.0f;
    float top_p = 1.0f;
    int32_t top_k = 0;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    float repeat_penalty = 1.0f;
    uint64_t seed = 42;

    bool is_greedy() const {
        return temperature <= 0.0f || top_k == 1 || top_p <= 0.0f;
    }
};

class Qwen4ExpFlyDslEngine {
public:
    Qwen4ExpFlyDslEngine(std::string model_dir, std::string cache_dir, std::string expert_profile_path,
                         int vram_expert_layers, int max_seq, int kv_resident, bool kv_grow,
                         strata::kernels::RopeScaling rope_scaling, bool rope_auto_yarn,
                         int vram_reserve_mib)
        : model_dir_(std::move(model_dir)),
          expert_profile_path_(std::move(expert_profile_path)),
          kernels_(std::move(cache_dir)),
          vram_expert_layers_(vram_expert_layers),
          vram_reserve_mib_(std::max(256, vram_reserve_mib)),
          max_seq_(std::max(4096, (max_seq + 3) & ~3)),
          max_blocks_(max_seq_ / 4),
          kv_resident_(std::max(8192, (kv_resident + 4095) & ~4095)),
          kv_grow_(kv_grow),
          rope_scaling_(rope_scaling),
          rope_auto_yarn_(rope_auto_yarn) {
        vram_cap_ = std::min(max_seq_, kv_resident_);
        vram_prefix_ = (max_seq_ > vram_cap_) ? (vram_cap_ - 4096) : vram_cap_;
        vram_blocks_ = vram_cap_ / 4;
    }

    ~Qwen4ExpFlyDslEngine() {
        cpu_pool_.stop();
        if (ev_mixed_) (void) hipEventDestroy(ev_mixed_);
        if (ev_route_) (void) hipEventDestroy(ev_route_);
        if (ple_.ple_fd >= 0) ::close(ple_.ple_fd);
        for (void* p : host_slab_registered_) {
            (void) hipHostUnregister(p);
        }
        if (host_kv_mmap_ && host_kv_mmap_ != MAP_FAILED) {
            (void) ::munmap(host_kv_mmap_, host_kv_mmap_bytes_);
        }
        if (host_bk_mmap_ && host_bk_mmap_ != MAP_FAILED) {
            (void) ::munmap(host_bk_mmap_, host_bk_mmap_bytes_);
        }
        if (host_rope_mmap_ && host_rope_mmap_ != MAP_FAILED) {
            (void) ::munmap(host_rope_mmap_, host_rope_mmap_bytes_);
        }
        kv_vmm_range_.release();
        exp_vmm_range_.release();
        for (void* p : vram_allocs_) (void) hipFree(p);
        for (void* p : host_pinned_) (void) hipHostFree(p);
        for (void* p : host_registered_) {
            (void) hipHostUnregister(p);
            std::free(p);
        }
        for (void* p : host_mallocs_) {
            std::free(p);
        }
    }

    void init(bool serve_mode = false) {
        serve_mode_ = serve_mode;
        FILE* log_fp = serve_mode_ ? stderr : stdout;
        auto t0 = std::chrono::steady_clock::now();
        if (const char* env_pf = std::getenv("STRATA_NVFP4_PCIE_FRAC")) {
            pcie_frac_ = std::clamp(static_cast<float>(std::atof(env_pf)), 0.0f, 1.0f);
        }
        cpu_pool_.init();
        std::fprintf(log_fp, "[flydsl-nvfp4] Initializing FlyDSL kernels, %d-core AVX-512 VNNI pool (pcie_frac=%.2f), and model buffers...\n",
                     cpu_pool_.num_threads(), pcie_frac_);
        std::fflush(log_fp);

        // Pre-warm all FlyDSL decode + batched prefill kernels (loads from .flydsl_strata_cache or JIT-compiles on miss)
        const char* warm_keys[] = {
            "embed_hc_gather", "embed_hc_gather_and_ple_dequant",
            "hc_premix_c1_u0", "hc_premix_c1_u1", "hc_premix_c0_u1",
            "hc_premix_decode_c1_u0", "hc_premix_decode_c1_u1", "hc_premix_decode_c0_u1",
            "hc_mix", "hc_mix_decode",
            "hc_premix_wmma_c1_u0", "hc_premix_wmma_c1_u1", "hc_premix_wmma_c0_u1", "hc_mix_wmma",
            "hc_residual_add", "ple_embed_dequant", "ple_gate", "ple_conv_add", "ple_conv_add_decode", "ple_state_update",
            "gated_deltanet_fused", "gdn_proj_split", "qsa_proj_split", "moe_sh_gu_extract",
            "qsa_q_extract", "qk_norm_partial_rope", "qsa_cache_write_nvfp4", "qsa_prep_and_cache_write_nvfp4",
            "qsa_prefill_causal_nvfp4",
            "qsa_block_keys", "qsa_block_keys_tiered", "qsa_select", "qsa_score_blocks_tiered", "qsa_topk_blocks",
            "attn_decode_partial_nvfp4", "attn_decode_fused_nvfp4", "qsa_sparse_decode_partial_nvfp4",
            "attn_decode_combine", "silu_mul_640", "bf16_gemv_swiglu_640_2560",
            "moe_topk", "moe_topk_decode", "moe_topk_and_silu_mul_640", "moe_align", "moe_gate_up",
            "moe_down", "moe_decode_gate_up", "moe_decode_down", "moe_scatter_rows", "moe_combine", "moe_combine_decode",
            "argmax_248320",
            "bf16_gemv_16480_2560", "bf16_gemv_13952_2560", "bf16_gemv_12800_2560", "bf16_gemv_1793_2560",
            "bf16_gemv_513_2560", "bf16_gemv_10240_2560", "bf16_gemv_2560_6144", "bf16_gemv_2560_640",
            "bf16_gemv_2560_2560", "bf16_gemv_248320_2560",
            "bf16_gemv_m4_16480_2560", "bf16_gemv_m4_13952_2560", "bf16_gemv_m4_2560_6144",
            "bf16_gemv_m4_1856_2560", "bf16_gemv_m4_2560_640", "bf16_gemv_m4_248320_2560",
            "bf16_linear_16480_2560", "bf16_linear_13952_2560", "bf16_linear_12800_2560", "bf16_linear_1856_2560",
            "bf16_linear_10240_2560", "bf16_linear_2560_2560", "bf16_linear_2560_6144",
            "bf16_linear_2560_640",
        };
        for (const char* k : warm_keys) {
            (void) kernels_.get(k);
        }

        alloc_state_and_scratch();
        init_rope_tables();
        load_weights_from_safetensors();

        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(log_fp, "[flydsl-nvfp4] Model ready in %.2f s (%d profile-ranked experts in VRAM, %d in pinned host RAM; "
                     "context window: %d -> %d max [VRAM resident cap: %d, prefix: %d + 4096 ring, host overflow: %d])\n",
                     dt, total_vram_experts_, 48 * 512 - total_vram_experts_,
                     kv_mapped_seq_, max_seq_, vram_cap_, vram_prefix_, std::max(0, max_seq_ - vram_prefix_));
        std::fflush(log_fp);
    }

    int32_t step_token(int32_t token_id, int32_t prev0, int32_t prev1, int32_t pos, bool compute_logits,
                       std::vector<float>* dump_logits_out = nullptr,
                       const SamplingParams* sp = nullptr,
                       const std::vector<int32_t>* history = nullptr,
                       std::mt19937_64* rng = nullptr) {
        ensure_context_capacity(pos + 1);
        // 1. Pack 16-byte step header into pinned host memory and immediately launch token embedding gather + Layer 0 on GPU
        int32_t* meta_i32 = static_cast<int32_t*>(h_step_meta_);
        meta_i32[0] = token_id;
        meta_i32[1] = pos;
        meta_i32[2] = pos + 1;
        meta_i32[3] = 0;
        HIP_CHECK(hipMemcpyAsync(d_step_meta_, h_step_meta_, 16, hipMemcpyHostToDevice, nullptr));

        int32_t one = 1;
        float eps = 1e-6f;
        void* d_cur_ids = static_cast<uint8_t*>(d_route_bell_) + 2560 * sizeof(uint16_t);
        void* d_cur_wts = static_cast<uint8_t*>(d_cur_ids) + 10 * sizeof(int32_t);

        {
            void* args[] = {&d_tok_id_, &d_embed_tokens_, &d_x_stream_a_};
            kernels_.launch_default("embed_hc_gather", args);
        }

        void* cur_x = d_x_stream_a_;
        void* alt_x = d_x_stream_b_;
        bool has_prev_update = false;
        void* prev_out = nullptr;
        void* prev_inj = nullptr;

        const bool prof = (std::getenv("STRATA_PROFILE_FLYDSL") != nullptr && compute_logits);
        // 3. Run all 48 layers
        for (int l = 0; l < 48; ++l) {
            LayerWeights& lw = layers_[l];
            if (prof) kernels_.set_profile(true, "");

            // Layer 1 PLE runs BEFORE Layer 1's attention hyper-connection mixer
            if (l == 1) {
                HIP_CHECK(hipMemcpyAsync(d_ple_rows_u8_, static_cast<uint8_t*>(h_step_meta_) + 16, 2560,
                                         hipMemcpyHostToDevice, nullptr));
                {
                    void* args[] = {&cur_x, &prev_out, &prev_inj, &alt_x};
                    kernels_.launch_default("hc_residual_add", args);
                    std::swap(cur_x, alt_x);
                    has_prev_update = false;
                }
                {
                    void* args_dq[] = {&d_ple_rows_u8_, &ple_.weight_scale, &d_ple_embed_, &one};
                    kernels_.launch_default("ple_embed_dequant", args_dq);
                }
                void* d_ple_v_dec = static_cast<uint16_t*>(d_ple_k_) + 10240;
                {
                    void* args_kv[] = {&d_ple_embed_, &ple_.key_proj, &d_ple_k_};
                    kernels_.launch_default("bf16_gemv_12800_2560", args_kv);
                }
                {
                    void* args[] = {&d_ple_k_, &d_ple_v_dec, &cur_x, &ple_.norm_key, &ple_.norm_query,
                                    &ple_.norm_conv, &d_ple_gv_, &d_ple_gvn_, &eps};
                    kernels_.launch_default("ple_gate", args);
                }
                {
                    void* args[] = {&cur_x, &d_ple_gv_, &d_ple_gvn_, &d_ple_conv_state_, &ple_.conv_weight, &alt_x};
                    kernels_.launch_default("ple_conv_add_decode", args);
                    std::swap(cur_x, alt_x);
                }
            }

            // Attention Hyper-Connection
            if (l == 20 && ev_prof_[0]) HIP_CHECK(hipEventRecord(ev_prof_[0], nullptr));
            {
                int64_t wi_delta = lw.attn_hc.wi_delta;
                if (!has_prev_update) {
                    void* args[] = {&cur_x, &cur_x, &cur_x, &cur_x, &lw.attn_hc.norm, &lw.attn_hc.down,
                                    &d_hc_h_, &d_attn_inj_, &d_hc_rstd_, &wi_delta, &eps};
                    kernels_.launch_default("hc_premix_decode_c1_u0", args);
                } else {
                    void* args[] = {&cur_x, &prev_out, &prev_inj, &alt_x, &lw.attn_hc.norm, &lw.attn_hc.down,
                                    &d_hc_h_, &d_attn_inj_, &d_hc_rstd_, &wi_delta, &eps};
                    kernels_.launch_default("hc_premix_decode_c1_u1", args);
                    std::swap(cur_x, alt_x);
                }
                void* m_args[] = {&cur_x, &lw.attn_hc.norm, &d_hc_rstd_, &d_hc_h_, &lw.attn_hc.up, &d_mixed_, &one};
                kernels_.launch_default("hc_mix_decode", m_args);
            }
            if (l == 20 && ev_prof_[1]) HIP_CHECK(hipEventRecord(ev_prof_[1], nullptr));

            // Attention / Linear-Attention Block
            if (!lw.is_qsa) {
                {
                    void* a1[] = {&d_mixed_, &lw.gdn.in_proj_fused, &d_gdn_qkv_};
                    kernels_.launch_default("bf16_gemv_16480_2560", a1);
                }
                if (l == 20 && ev_prof_[2]) HIP_CHECK(hipEventRecord(ev_prof_[2], nullptr));
                {
                    void* csi = d_gdn_conv_in_[l];
                    void* cso = d_gdn_conv_out_[l];
                    void* st = d_gdn_state_[l];
                    void* args[] = {&d_gdn_qkv_, &d_gdn_z_, &d_gdn_a_, &d_gdn_b_, &lw.gdn.A_log, &lw.gdn.dt_bias,
                                    &lw.gdn.conv1d, &csi, &cso, &st, &lw.gdn.norm, &d_gdn_out_, &one, &eps};
                    kernels_.launch_default("gated_deltanet_fused", args);
                    std::swap(d_gdn_conv_in_[l], d_gdn_conv_out_[l]);
                }
                if (l == 20 && ev_prof_[3]) HIP_CHECK(hipEventRecord(ev_prof_[3], nullptr));
                {
                    void* args[] = {&d_gdn_out_, &lw.gdn.out_proj, &d_attn_out_};
                    kernels_.launch_default("bf16_gemv_2560_6144", args);
                }
                if (l == 20 && ev_prof_[4]) HIP_CHECK(hipEventRecord(ev_prof_[4], nullptr));
            } else {
                const int qsa_idx = l / 4;
                {
                    void* a1[] = {&d_mixed_, &lw.qsa.in_proj_fused, &d_qsa_qg_};
                    kernels_.launch_default("bf16_gemv_13952_2560", a1);
                }
                void* kc = d_k_cache_[qsa_idx];
                void* vc = d_v_cache_[qsa_idx];
                void* rkc = d_raw_k_cache_[qsa_idx];
                void* bkc = d_block_k_cache_[qsa_idx];
                int64_t kc_delta = kc_host_delta_[qsa_idx];
                int64_t vc_delta = vc_host_delta_[qsa_idx];
                int64_t bk_delta = bk_host_delta_elems_[qsa_idx];
                {
                    void* args[] = {&d_qsa_qg_, &lw.qsa.q_norm, &lw.qsa.k_norm, &d_rope_cos_f32_, &d_rope_sin_f32_,
                                    &d_qsa_q_, &kc, &vc, &rkc, &kc_delta, &vc_delta, &pos, &vram_prefix_, &eps};
                    kernels_.launch_default("qsa_prep_and_cache_write_nvfp4", args);
                }
                if ((pos + 1) % 4 == 0) {
                    int32_t b_start = pos / 4;
                    int32_t count = 1;
                    void* args[] = {&rkc, &lw.qsa.idx_k_ln, &d_rope_cos_bf16_, &d_rope_sin_bf16_, &bkc,
                                    &bk_delta, &b_start, &count, &vram_blocks_, &eps};
                    kernels_.launch("qsa_block_keys_tiered", 1, 1, 1, args);
                }
                int32_t Lk = pos + 1;
                int32_t chunk = 256;
                if (Lk <= 256) {
                    void* af[] = {&d_qsa_q_, &kc, &vc, &d_qsa_qg_, &d_qsa_attn_raw_, &d_seq_k_i32_};
                    kernels_.launch_default("attn_decode_fused_nvfp4", af);
                } else if (Lk <= 2048) {
                    int32_t splits = (Lk + chunk - 1) / chunk;
                    void* ap[] = {&d_qsa_q_, &kc, &vc, &d_zero_mask_, &d_attn_pm_, &d_attn_pl_, &d_attn_pa_,
                                  &d_seq_k_i32_, &chunk, &splits};
                    kernels_.launch("attn_decode_partial_nvfp4", splits * 24, 1, 1, ap);
                    void* ac[] = {&d_attn_pm_, &d_attn_pl_, &d_attn_pa_, &d_qsa_qg_, &d_qsa_attn_raw_, &splits};
                    kernels_.launch("attn_decode_combine", 24, 1, 1, ac);
                } else {
                    int32_t nb = (pos + 1) / 4;
                    int32_t score_grid = std::max(1, (nb + 255) / 256);
                    void* asc[] = {&d_qsa_idx_qk_, &lw.qsa.idx_q_ln, &d_rope_cos_bf16_, &d_rope_sin_bf16_,
                                   &bkc, &d_qsa_scores_, &bk_delta, &pos, &vram_blocks_, &max_blocks_, &eps};
                    kernels_.launch("qsa_score_blocks_tiered", score_grid, 1, 1, asc);
                    void* atk[] = {&d_qsa_scores_, &d_qsa_sel_ids_, &pos, &max_blocks_};
                    kernels_.launch("qsa_topk_blocks", 1, 1, 1, atk);
                    int32_t splits = 9;
                    void* ap[] = {&d_qsa_q_, &kc, &vc, &d_qsa_sel_ids_, &d_pos_i32_, &d_attn_pm_, &d_attn_pl_,
                                  &d_attn_pa_, &kc_delta, &vc_delta, &vram_prefix_, &chunk, &splits};
                    kernels_.launch("qsa_sparse_decode_partial_nvfp4", splits * 24, 1, 1, ap);
                    void* ac[] = {&d_attn_pm_, &d_attn_pl_, &d_attn_pa_, &d_qsa_qg_, &d_qsa_attn_raw_, &splits};
                    kernels_.launch("attn_decode_combine", 24, 1, 1, ac);
                }
                {
                    void* args[] = {&d_qsa_attn_raw_, &lw.qsa.o_proj, &d_attn_out_};
                    kernels_.launch_default("bf16_gemv_2560_6144", args);
                }
            }

            // MLP Hyper-Connection
            {
                int64_t wi_delta = lw.mlp_hc.wi_delta;
                void* args[] = {&cur_x, &d_attn_out_, &d_attn_inj_, &alt_x, &lw.mlp_hc.norm, &lw.mlp_hc.down,
                                &d_hc_h_, &d_mlp_inj_, &d_hc_rstd_, &wi_delta, &eps};
                kernels_.launch_default("hc_premix_decode_c1_u1", args);
                std::swap(cur_x, alt_x);
                void* m_args[] = {&cur_x, &lw.mlp_hc.norm, &d_hc_rstd_, &d_hc_h_, &lw.mlp_hc.up, &d_mixed_, &one};
                kernels_.launch_default("hc_mix_decode", m_args);
            }
            if (l == 20 && ev_prof_[5]) HIP_CHECK(hipEventRecord(ev_prof_[5], nullptr));

            // MoE Block (Hybrid CPU+GPU: Fused Shared Gate+Up+Router -> TopK+SwiGLU -> Doorbell -> Dynamic Balanced GPU/CPU Experts)
            {
                void* a_gu[] = {&d_mixed_, &lw.moe.in_proj_fused, &d_sh_gu_};
                kernels_.launch_default("bf16_gemv_1793_2560", a_gu);

                void* ptrs = lw.moe.d_ptrs;
                void* ptrs_vram = lw.moe.d_ptrs_prefill;
                void* a_ts[] = {&d_sh_gu_, &ptrs, &ptrs_vram, &d_cur_ids, &d_cur_wts, &d_sh_inter_};
                kernels_.launch_default("moe_topk_and_silu_mul_640", a_ts);

                // Single D2H doorbell copy: [d_mixed_ (5120 B) | d_cur_ids (40 B) | d_cur_wts (40 B)] = 5200 B contiguous
                HIP_CHECK(hipMemcpyAsync(h_route_bell_, d_route_bell_,
                                         2560 * sizeof(uint16_t) + 10 * sizeof(int32_t) + 10 * sizeof(float),
                                         hipMemcpyDeviceToHost, nullptr));
                HIP_CHECK(hipEventRecord(ev_route_, nullptr));
                if (l == 20 && ev_prof_[6]) HIP_CHECK(hipEventRecord(ev_prof_[6], nullptr));

                // Immediately enqueue Shared Down + GPU Routed Experts on the GPU stream
                void* a_d[] = {&d_sh_inter_, &lw.moe.sh_down, &d_sh_out_};
                kernels_.launch_default("bf16_gemv_2560_640", a_d);

                void* s2 = lw.moe.d_s2;
                void* a_mgu[] = {&d_mixed_, &d_cur_ids, &ptrs, &s2, &d_moe_inter_};
                kernels_.launch_default("moe_decode_gate_up", a_mgu);
                void* a_dn[] = {&d_moe_inter_, &d_cur_ids, &ptrs, &s2, &d_moe_y_};
                kernels_.launch_default("moe_decode_down", a_dn);
                if (l == 20 && ev_prof_[7]) HIP_CHECK(hipEventRecord(ev_prof_[7], nullptr));

                // While Layer 0 GPU kernels are running, gather all 16 PLE FP8 rows at Queue Depth 16 on the CPU pool!
                if (l == 0) {
                    uint32_t rows16[16]{};
                    const int32_t prev_pair[2] = {prev0, prev1};
                    strata::kernels::ngram_rows(&token_id, prev_pair, 1, ple_consts_, rows16);
                    uint8_t* h_ple_rows = static_cast<uint8_t*>(h_step_meta_) + 16;
                    uint64_t ple_offs[16];
                    uint8_t* ple_dsts[16];
                    for (int h = 0; h < 16; ++h) {
                        uint32_t r = rows16[h];
                        uint32_t shard = r / 2500012u;
                        uint32_t row_in_shard = r % 2500012u;
                        ple_offs[h] = ple_.shard_offsets[shard] + static_cast<uint64_t>(row_in_shard) * 160ULL;
                        ple_dsts[h] = h_ple_rows + h * 160;
                    }
                    cpu_pool_.run_ple_reads(ple_.ple_fd, ple_offs, ple_dsts, 16);
                }

                // Run CPU-assigned cold experts concurrently on the 16-core AVX-512 VNNI pool and pre-combine on host
                int32_t has_cpu = run_cpu_moe_layer(l, 1, /*is_prefill=*/false);

                void* a_cb[] = {&d_moe_y_, &d_cur_ids, &ptrs, &d_cur_wts, &d_cpu_upload_,
                                &d_sh_out_, &d_sh_scale_, &d_mlp_out_, &has_cpu};
                kernels_.launch_default("moe_combine_decode", a_cb);
                if (l == 20 && ev_prof_[8]) HIP_CHECK(hipEventRecord(ev_prof_[8], nullptr));
            }

            has_prev_update = true;
            prev_out = d_mlp_out_;
            prev_inj = d_mlp_inj_;
        }

        if (!compute_logits) {
            adapt_vram_experts(h_all_routed_ids_.data(), 1, /*min_hits=*/2, /*max_swaps=*/2);
            t_wait_acc_ms_ = 0.0;
            t_cpu_acc_ms_ = 0.0;
            n_cpu_exp_acc_ = 0;
            n_gpu_pcie_acc_ = 0;
            n_vram_exp_acc_ = 0;
            return -1;
        }
        if (prof) kernels_.set_profile(true, "head:");

        // 4. Final Hyper-Connection Mixer (`combine=False, update=True`)
        {
            int64_t zero_delta = 0;
            void* args[] = {&cur_x, &prev_out, &prev_inj, &alt_x, &final_hc_.norm, &final_hc_.down,
                            &d_hc_h_, &d_hc_h_, &d_hc_rstd_, &zero_delta, &eps};
            kernels_.launch_default("hc_premix_decode_c0_u1", args);
            std::swap(cur_x, alt_x);
            void* m_args[] = {&cur_x, &final_hc_.norm, &d_hc_rstd_, &d_hc_h_, &final_hc_.up, &d_final_hidden_, &one};
            kernels_.launch_default("hc_mix_decode", m_args);
        }

        // 5. LM Head + Argmax / Non-Greedy Sampling
        const bool use_sampling = (sp && !sp->is_greedy() && rng);
        {
            void* a_lm[] = {&d_final_hidden_, &d_lm_head_, &d_logits_bf16_};
            kernels_.launch_default("bf16_gemv_248320_2560", a_lm);
            if (!use_sampling) {
                void* a_am[] = {&d_logits_bf16_, &d_next_tok_};
                kernels_.launch_default("argmax_248320", a_am);
            }
        }

        int32_t next_id = -1;
        auto t_sync0 = std::chrono::steady_clock::now();
        if (use_sampling) {
            next_id = sample_from_device_logits(*sp, history, *rng);
        } else {
            HIP_CHECK(hipMemcpy(&next_id, d_next_tok_, sizeof(int32_t), hipMemcpyDeviceToHost));
        }
        auto t_adapt0 = std::chrono::steady_clock::now();
        adapt_vram_experts(h_all_routed_ids_.data(), 1, /*min_hits=*/2, /*max_swaps=*/2);
        auto t_adapt1 = std::chrono::steady_clock::now();
        if (std::getenv("STRATA_NVFP4_TIMING") != nullptr && compute_logits) {
            double sync_ms = std::chrono::duration<double, std::milli>(t_adapt0 - t_sync0).count();
            double adapt_ms = std::chrono::duration<double, std::milli>(t_adapt1 - t_adapt0).count();
            float g_hc1 = 0, g_in = 0, g_mid = 0, g_out = 0, g_hc2 = 0, g_rt = 0, g_moe = 0, g_cb = 0;
            if (ev_prof_[0]) {
                (void) hipEventElapsedTime(&g_hc1, ev_prof_[0], ev_prof_[1]);
                (void) hipEventElapsedTime(&g_in,  ev_prof_[1], ev_prof_[2]);
                (void) hipEventElapsedTime(&g_mid, ev_prof_[2], ev_prof_[3]);
                (void) hipEventElapsedTime(&g_out, ev_prof_[3], ev_prof_[4]);
                (void) hipEventElapsedTime(&g_hc2, ev_prof_[4], ev_prof_[5]);
                (void) hipEventElapsedTime(&g_rt,  ev_prof_[5], ev_prof_[6]);
                (void) hipEventElapsedTime(&g_moe, ev_prof_[6], ev_prof_[7]);
                (void) hipEventElapsedTime(&g_cb,  ev_prof_[7], ev_prof_[8]);
            }
            std::printf("[timing] wait=%.2f ms | cpu_moe=%.2f ms (cpu=%d, pcie=%d, vram=%d) | sync=%.2f | adapt=%.2f | L20(us): hc1=%.0f in=%.0f gdn=%.0f out=%.0f hc2=%.0f rt=%.0f moe=%.0f cb=%.0f\n",
                        t_wait_acc_ms_, t_cpu_acc_ms_, n_cpu_exp_acc_, n_gpu_pcie_acc_, n_vram_exp_acc_, sync_ms, adapt_ms,
                        g_hc1 * 1000.0f, g_in * 1000.0f, g_mid * 1000.0f, g_out * 1000.0f,
                        g_hc2 * 1000.0f, g_rt * 1000.0f, g_moe * 1000.0f, g_cb * 1000.0f);
            t_wait_acc_ms_ = 0.0;
            t_cpu_acc_ms_ = 0.0;
            n_cpu_exp_acc_ = 0;
            n_gpu_pcie_acc_ = 0;
            n_vram_exp_acc_ = 0;
        }

        if (prof) {
            kernels_.set_profile(false);
            kernels_.dump_profile();
        }

        if (dump_logits_out) {
            std::vector<uint16_t> bf(248320);
            HIP_CHECK(hipMemcpy(bf.data(), d_logits_bf16_, 248320 * sizeof(uint16_t), hipMemcpyDeviceToHost));
            dump_logits_out->resize(248320);
            for (size_t i = 0; i < 248320; ++i) {
                (*dump_logits_out)[i] = bf16_bits_to_f32(bf[i]);
            }
        }
        return next_id;
    }

    int32_t step_prefill_batch(const int32_t* tokens, const int32_t* prev0s, const int32_t* prev1s,
                               int32_t M, int32_t pos0, bool compute_logits,
                               std::vector<float>* dump_logits_out = nullptr,
                               int32_t* verify_next_toks_out = nullptr,
                               const SamplingParams* sp = nullptr,
                               const std::vector<int32_t>* history = nullptr,
                               std::mt19937_64* rng = nullptr) {
        if (M == 1 && !verify_next_toks_out) {
            return step_token(tokens[0], prev0s[0], prev1s[0], pos0, compute_logits, dump_logits_out, sp, history, rng);
        }
        ensure_context_capacity(pos0 + M);

        // 1. Copy M token IDs to GPU and immediately launch token embedding gather + Layer 0 GPU kernels
        //    so Layer 0 GPU WMMA runs concurrently with the CPU's Queue-Depth-16 PLE NVMe reads!
        uint8_t* h_ple_rows = static_cast<uint8_t*>(h_step_meta_) + 16;
        uint8_t* h_batch_toks = h_ple_rows + static_cast<size_t>(M) * 2560;
        std::memcpy(h_batch_toks, tokens, static_cast<size_t>(M) * sizeof(int32_t));
        void* d_batch_toks = static_cast<uint8_t*>(d_ple_rows_u8_) + static_cast<size_t>(M) * 2560;
        HIP_CHECK(hipMemcpyAsync(d_batch_toks, h_batch_toks, static_cast<size_t>(M) * sizeof(int32_t),
                                 hipMemcpyHostToDevice, nullptr));

        int32_t one = 1;
        float eps = 1e-6f;
        const int32_t m_tiles16 = (M + 15) / 16;
        const int32_t m_tiles64 = (M + 63) / 64;
        const bool use_wmma_hc = (M >= 16);
        void* d_cur_ids = static_cast<uint8_t*>(d_route_bell_) + static_cast<size_t>(M) * 2560 * sizeof(uint16_t);

        // 2. Embed M tokens -> d_x_stream_a_ [M, 10240] BF16
        {
            void* args[] = {&d_batch_toks, &d_embed_tokens_, &d_x_stream_a_};
            kernels_.launch("embed_hc_gather", M, 1, 1, args);
        }

        void* cur_x = d_x_stream_a_;
        void* alt_x = d_x_stream_b_;
        bool has_prev_update = false;
        void* prev_out = nullptr;
        void* prev_inj = nullptr;

        // 3. Run all 48 layers over M tokens simultaneously
        for (int l = 0; l < 48; ++l) {
            LayerWeights& lw = layers_[l];

            if (l == 1) {
                HIP_CHECK(hipMemcpyAsync(d_ple_rows_u8_, h_ple_rows, static_cast<size_t>(M) * 2560,
                                         hipMemcpyHostToDevice, nullptr));
                {
                    void* args[] = {&cur_x, &prev_out, &prev_inj, &alt_x};
                    kernels_.launch("hc_residual_add", 40, M, 1, args);
                    std::swap(cur_x, alt_x);
                    has_prev_update = false;
                }
                {
                    void* args_dq[] = {&d_ple_rows_u8_, &ple_.weight_scale, &d_ple_embed_, &M};
                    kernels_.launch("ple_embed_dequant", M, 1, 1, args_dq);
                }
                {
                    void* args_k[] = {&d_ple_embed_, &ple_.key_proj, &d_ple_k_, &M};
                    kernels_.launch("bf16_linear_10240_2560", 160, m_tiles16, 1, args_k);
                    void* args_v[] = {&d_ple_embed_, &ple_.value_proj, &d_ple_v_, &M};
                    kernels_.launch("bf16_linear_2560_2560", 40, m_tiles16, 1, args_v);
                }
                {
                    void* args[] = {&d_ple_k_, &d_ple_v_, &cur_x, &ple_.norm_key, &ple_.norm_query,
                                    &ple_.norm_conv, &d_ple_gv_, &d_ple_gvn_, &eps};
                    kernels_.launch("ple_gate", M, 1, 1, args);
                }
                {
                    void* args[] = {&cur_x, &d_ple_gv_, &d_ple_gvn_, &d_ple_conv_state_, &ple_.conv_weight, &alt_x, &M};
                    kernels_.launch("ple_conv_add", (M * 1280 + 255) / 256, 1, 1, args);
                    std::swap(cur_x, alt_x);
                }
                for (int32_t t = 0; t < M; ++t) {
                    void* gvn_t = static_cast<uint16_t*>(d_ple_gvn_) + static_cast<size_t>(t) * 10240;
                    void* args[] = {&d_ple_conv_state_, &gvn_t};
                    kernels_.launch_default("ple_state_update", args);
                }
            }

            // Attention Hyper-Connection
            {
                int64_t wi_delta = lw.attn_hc.wi_delta;
                if (!has_prev_update) {
                    void* args[] = {&cur_x, &cur_x, &cur_x, &cur_x, &lw.attn_hc.norm, &lw.attn_hc.down,
                                    &d_hc_h_, &d_attn_inj_, &d_hc_rstd_, &wi_delta, &M, &eps};
                    if (use_wmma_hc) {
                        kernels_.launch("hc_premix_wmma_c1_u0", 21, m_tiles64, 1, args);
                    } else {
                        kernels_.launch("hc_premix_c1_u0", 81, M, 1, args);
                    }
                } else {
                    void* args[] = {&cur_x, &prev_out, &prev_inj, &alt_x, &lw.attn_hc.norm, &lw.attn_hc.down,
                                    &d_hc_h_, &d_attn_inj_, &d_hc_rstd_, &wi_delta, &M, &eps};
                    if (use_wmma_hc) {
                        kernels_.launch("hc_premix_wmma_c1_u1", 21, m_tiles64, 1, args);
                    } else {
                        kernels_.launch("hc_premix_c1_u1", 81, M, 1, args);
                    }
                    std::swap(cur_x, alt_x);
                }
                void* m_args[] = {&cur_x, &lw.attn_hc.norm, &d_hc_rstd_, &d_hc_h_, &lw.attn_hc.up, &d_mixed_, &M};
                if (use_wmma_hc) {
                    kernels_.launch("hc_mix_wmma", 40, m_tiles64, 1, m_args);
                } else {
                    kernels_.launch("hc_mix", 80, M, 1, m_args);
                }
            }

            // Attention / Linear-Attention Block
            const bool is_small_batch = (M <= 4);
            if (!lw.is_qsa) {
                void* qkv_buf = d_gdn_qkv_;
                void* z_buf = d_gdn_z_b_;
                void* a_buf = d_gdn_a_b_;
                void* b_buf = d_gdn_b_b_;
                if (is_small_batch) {
                    qkv_buf = d_gdn_verify_qkv_[l];
                    z_buf = static_cast<uint16_t*>(qkv_buf) + 4 * 10240;
                    a_buf = static_cast<uint16_t*>(qkv_buf) + 4 * 16384;
                    b_buf = static_cast<uint16_t*>(qkv_buf) + 4 * 16432;
                }
                {
                    void* a1[] = {&d_mixed_, &lw.gdn.in_proj_fused, &d_proj_scratch_, &M};
                    kernels_.launch("bf16_linear_16480_2560", 515, m_tiles16, 1, a1);
                    void* as[] = {&d_proj_scratch_, &qkv_buf, &z_buf, &a_buf, &b_buf};
                    kernels_.launch("gdn_proj_split", 65, M, 1, as);
                }
                {
                    void* csi = d_gdn_conv_in_[l];
                    void* cso = d_gdn_conv_out_[l];
                    void* st = d_gdn_state_[l];
                    void* args[] = {&qkv_buf, &z_buf, &a_buf, &b_buf, &lw.gdn.A_log, &lw.gdn.dt_bias,
                                    &lw.gdn.conv1d, &csi, &cso, &st, &lw.gdn.norm, &d_gdn_out_, &M, &eps};
                    kernels_.launch_default("gated_deltanet_fused", args);
                    std::swap(d_gdn_conv_in_[l], d_gdn_conv_out_[l]);
                }
                {
                    void* args[] = {&d_gdn_out_, &lw.gdn.out_proj, &d_attn_out_, &M};
                    kernels_.launch("bf16_linear_2560_6144", 40, m_tiles16, 1, args);
                }
            } else {
                const int qsa_idx = l / 4;
                {
                    void* a1[] = {&d_mixed_, &lw.qsa.in_proj_fused, &d_proj_scratch_, &M};
                    kernels_.launch("bf16_linear_13952_2560", 218, m_tiles16, 1, a1);
                    void* as[] = {&d_proj_scratch_, &d_qsa_qg_, &d_qsa_k_b_, &d_qsa_v_b_, &d_qsa_idx_qk_b_};
                    kernels_.launch("qsa_proj_split", 55, M, 1, as);
                }
                {
                    void* args[] = {&d_qsa_qg_, &d_qsa_q_};
                    kernels_.launch("qsa_q_extract", 24, M, 1, args);
                }
                {
                    void* cos_p = static_cast<uint8_t*>(d_rope_cos_f32_) + static_cast<size_t>(pos0) * 32 * sizeof(float);
                    void* sin_p = static_cast<uint8_t*>(d_rope_sin_f32_) + static_cast<size_t>(pos0) * 32 * sizeof(float);
                    int32_t r24 = M * 24, h24 = 24;
                    void* aq[] = {&d_qsa_q_, &lw.qsa.q_norm, &cos_p, &sin_p, &d_qsa_q_, &r24, &h24, &eps};
                    kernels_.launch("qk_norm_partial_rope", M * 3, 1, 1, aq);
                    int32_t r2 = M * 2, h2 = 2;
                    void* ak[] = {&d_qsa_k_b_, &lw.qsa.k_norm, &cos_p, &sin_p, &d_qsa_k_b_, &r2, &h2, &eps};
                    kernels_.launch("qk_norm_partial_rope", (M * 2 + 7) / 8, 1, 1, ak);
                }
                void* kc = d_k_cache_[qsa_idx];
                void* vc = d_v_cache_[qsa_idx];
                void* rkc = d_raw_k_cache_[qsa_idx];
                void* bkc = d_block_k_cache_[qsa_idx];
                int64_t kc_delta = kc_host_delta_[qsa_idx];
                int64_t vc_delta = vc_host_delta_[qsa_idx];
                int64_t bk_delta = bk_host_delta_elems_[qsa_idx];
                {
                    void* args[] = {&d_qsa_k_b_, &d_qsa_v_b_, &d_qsa_idx_qk_b_, &kc, &vc, &rkc,
                                    &kc_delta, &vc_delta, &pos0, &vram_prefix_};
                    kernels_.launch("qsa_cache_write_nvfp4", 1, M, 1, args);
                }
                int32_t b_start = pos0 / 4;
                int32_t b_end = (pos0 + M) / 4;
                int32_t b_cnt = b_end - b_start;
                if (b_cnt > 0) {
                    void* args[] = {&rkc, &lw.qsa.idx_k_ln, &d_rope_cos_bf16_, &d_rope_sin_bf16_, &bkc,
                                    &bk_delta, &b_start, &b_cnt, &vram_blocks_, &eps};
                    kernels_.launch("qsa_block_keys_tiered", (b_cnt + 3) / 4, 1, 1, args);
                }
                if (pos0 + M > 2048) {
                    int32_t t_first = std::max(0, 2048 - pos0);
                    int32_t max_nb = (pos0 + M) / 4;
                    int32_t score_grid = std::max(1, (max_nb + 255) / 256);
                    int32_t score_stride = score_grid * 256;
                    int32_t max_m_per_launch = std::max(1, static_cast<int32_t>(4194304 / score_stride));
                    for (int32_t t0 = t_first; t0 < M; t0 += max_m_per_launch) {
                        int32_t m_cur = std::min(M - t0, max_m_per_launch);
                        int32_t pos_cur = pos0 + t0;
                        void* qk_cur = static_cast<uint16_t*>(d_qsa_idx_qk_b_) + static_cast<size_t>(t0) * 640;
                        void* sel_cur = static_cast<int32_t*>(d_qsa_sel_ids_) + static_cast<size_t>(t0) * 512;
                        void* asc[] = {&qk_cur, &lw.qsa.idx_q_ln, &d_rope_cos_bf16_, &d_rope_sin_bf16_,
                                       &bkc, &d_qsa_scores_, &bk_delta, &pos_cur, &vram_blocks_, &score_stride, &eps};
                        kernels_.launch("qsa_score_blocks_tiered", score_grid, m_cur, 1, asc);
                        void* atk[] = {&d_qsa_scores_, &sel_cur, &pos_cur, &score_stride};
                        kernels_.launch("qsa_topk_blocks", m_cur, 1, 1, atk);
                    }
                }
                {
                    void* ap[] = {&d_qsa_q_, &kc, &vc, &d_qsa_qg_, &d_qsa_sel_ids_, &d_qsa_attn_raw_,
                                  &kc_delta, &vc_delta, &pos0, &M, &vram_prefix_};
                    kernels_.launch("qsa_prefill_causal_nvfp4", 24, M, 1, ap);
                }
                {
                    void* args[] = {&d_qsa_attn_raw_, &lw.qsa.o_proj, &d_attn_out_, &M};
                    kernels_.launch("bf16_linear_2560_6144", 40, m_tiles16, 1, args);
                }
            }

            // MLP Hyper-Connection
            {
                int64_t wi_delta = lw.mlp_hc.wi_delta;
                void* args[] = {&cur_x, &d_attn_out_, &d_attn_inj_, &alt_x, &lw.mlp_hc.norm, &lw.mlp_hc.down,
                                &d_hc_h_, &d_mlp_inj_, &d_hc_rstd_, &wi_delta, &M, &eps};
                if (use_wmma_hc) {
                    kernels_.launch("hc_premix_wmma_c1_u1", 21, m_tiles64, 1, args);
                } else {
                    kernels_.launch("hc_premix_c1_u1", 81, M, 1, args);
                }
                std::swap(cur_x, alt_x);
                void* m_args[] = {&cur_x, &lw.mlp_hc.norm, &d_hc_rstd_, &d_hc_h_, &lw.mlp_hc.up, &d_mixed_, &M};
                if (use_wmma_hc) {
                    kernels_.launch("hc_mix_wmma", 40, m_tiles64, 1, m_args);
                } else {
                    kernels_.launch("hc_mix", 80, M, 1, m_args);
                }
            }

            // MoE Block (Hybrid CPU+GPU: GPU runs VRAM experts via WMMA while 16-Core AVX-512 VNNI pool runs cold host experts)
            {
                void* a_in[] = {&d_mixed_, &lw.moe.in_proj_fused, &d_proj_scratch_, &M};
                kernels_.launch("bf16_linear_1856_2560", 29, m_tiles16, 1, a_in);

                void* r_logits = static_cast<uint16_t*>(d_proj_scratch_) + 1280;
                void* sh_sc = static_cast<uint16_t*>(d_proj_scratch_) + 1792;
                int32_t stride1856 = 1856;
                void* a_tk[] = {&r_logits, &d_cur_ids, &d_moe_wts_, &M, &stride1856};
                kernels_.launch("moe_topk", (M + 3) / 4, 1, 1, a_tk);

                // Async copy [d_mixed_ (M*5120 B) | d_cur_ids (M*40 B)] to pinned host memory & record doorbell event
                HIP_CHECK(hipMemcpyAsync(h_route_bell_, d_route_bell_,
                                         static_cast<size_t>(M) * (2560 * sizeof(uint16_t) + 10 * sizeof(int32_t)),
                                         hipMemcpyDeviceToHost, nullptr));
                HIP_CHECK(hipEventRecord(ev_route_, nullptr));

                // Immediately enqueue Shared Expert + VRAM/PCIe GPU Routed Experts on the GPU stream
                void* a_ex[] = {&d_proj_scratch_, &d_sh_gu_};
                kernels_.launch("moe_sh_gu_extract", 5, M, 1, a_ex);
                void* a_sm[] = {&d_sh_gu_, &d_sh_inter_, &M};
                kernels_.launch("silu_mul_640", (M * 80 + 255) / 256, 1, 1, a_sm);
                void* a_d[] = {&d_sh_inter_, &lw.moe.sh_down, &d_sh_out_, &M};
                kernels_.launch("bf16_linear_2560_640", 40, m_tiles16, 1, a_d);

                int32_t pairs = M * 10;
                int32_t max_tiles = (pairs + 15) / 16 + std::min(512, pairs);
                void* a_al[] = {&d_cur_ids, &d_moe_sorted_, &d_moe_tile_e_, &pairs, &max_tiles};
                kernels_.launch("moe_align", 1, 1, 1, a_al);

                void* ptrs = lw.moe.d_ptrs_prefill;
                void* s2 = lw.moe.d_s2;
                int32_t topk10 = 10;
                void* a_gu[] = {&d_mixed_, &d_moe_sorted_, &d_moe_tile_e_, &ptrs, &s2, &d_moe_inter_, &topk10};
                kernels_.launch("moe_gate_up", 5, max_tiles, 1, a_gu);
                void* a_dn[] = {&d_moe_inter_, &d_moe_sorted_, &d_moe_tile_e_, &ptrs, &s2, &d_moe_y_};
                kernels_.launch("moe_down", 10, max_tiles, 1, a_dn);

                // While Layer 0 GPU WMMA kernels are executing, gather all M x 16 PLE rows at Queue Depth 16!
                if (l == 0) {
                    struct PleReq {
                        uint64_t off;
                        uint8_t* dst;
                    };
                    std::vector<PleReq> reqs(static_cast<size_t>(M) * 16);
                    for (int32_t t = 0; t < M; ++t) {
                        uint32_t rows16[16]{};
                        const int32_t prev_pair[2] = {prev0s[t], prev1s[t]};
                        strata::kernels::ngram_rows(&tokens[t], prev_pair, 1, ple_consts_, rows16);
                        uint8_t* dst_row = h_ple_rows + static_cast<size_t>(t) * 2560;
                        for (int h = 0; h < 16; ++h) {
                            uint32_t r = rows16[h];
                            uint32_t shard = r / 2500012u;
                            uint32_t row_in_shard = r % 2500012u;
                            reqs[static_cast<size_t>(t) * 16 + h] = PleReq{
                                ple_.shard_offsets[shard] + static_cast<uint64_t>(row_in_shard) * 160ULL,
                                dst_row + h * 160,
                            };
                        }
                    }
                    std::sort(reqs.begin(), reqs.end(), [](const PleReq& a, const PleReq& b) {
                        return a.off < b.off;
                    });
                    std::vector<uint64_t> uniq_offs;
                    std::vector<uint8_t*> uniq_dsts;
                    uniq_offs.reserve(reqs.size());
                    uniq_dsts.reserve(reqs.size());
                    for (size_t i = 0; i < reqs.size(); ++i) {
                        if (i == 0 || reqs[i].off != reqs[i - 1].off) {
                            uniq_offs.push_back(reqs[i].off);
                            uniq_dsts.push_back(reqs[i].dst);
                        }
                    }
                    cpu_pool_.run_ple_reads(ple_.ple_fd, uniq_offs.data(), uniq_dsts.data(),
                                            static_cast<int>(uniq_offs.size()));
                    for (size_t i = 1; i < reqs.size(); ++i) {
                        if (reqs[i].off == reqs[i - 1].off) {
                            std::memcpy(reqs[i].dst, reqs[i - 1].dst, 160);
                        }
                    }
                }

                // Run all host-resident (!in_vram) experts on the 16-core AVX-512 VNNI pool and scatter into d_moe_y_
                (void) run_cpu_moe_layer(l, M, /*is_prefill=*/true);

                void* a_cb[] = {&d_moe_y_, &d_moe_wts_, &d_sh_out_, &sh_sc, &d_mlp_out_, &stride1856};
                kernels_.launch("moe_combine", M, 1, 1, a_cb);
            }

            has_prev_update = true;
            prev_out = d_mlp_out_;
            prev_inj = d_mlp_inj_;
        }

        if (!compute_logits) {
            adapt_vram_experts(h_all_routed_ids_.data(), M, /*min_hits=*/2, /*max_swaps=*/12);
            return -1;
        }

        if (verify_next_toks_out) {
            // Multi-token speculative verification: compute final HC mixer + 1-pass M-token LM head + argmax
            int64_t zero_delta = 0;
            void* args[] = {&cur_x, &prev_out, &prev_inj, &alt_x, &final_hc_.norm, &final_hc_.down,
                            &d_hc_h_, &d_hc_h_, &d_hc_rstd_, &zero_delta, &M, &eps};
            kernels_.launch("hc_premix_c0_u1", 80, M, 1, args);
            std::swap(cur_x, alt_x);
            void* m_args[] = {&cur_x, &final_hc_.norm, &d_hc_rstd_, &d_hc_h_, &final_hc_.up, &d_attn_out_, &M};
            kernels_.launch("hc_mix", 80, M, 1, m_args);
            void* a_lm[] = {&d_attn_out_, &d_lm_head_, &d_logits_bf16_, &M};
            kernels_.launch_default("bf16_gemv_m4_248320_2560", a_lm);
            for (int32_t t = 0; t < M; ++t) {
                void* log_t = static_cast<uint16_t*>(d_logits_bf16_) + static_cast<size_t>(t) * 248320;
                void* out_t = static_cast<int32_t*>(d_next_tok_) + t;
                void* a_am[] = {&log_t, &out_t};
                kernels_.launch_default("argmax_248320", a_am);
            }
            HIP_CHECK(hipMemcpy(verify_next_toks_out, d_next_tok_, static_cast<size_t>(M) * sizeof(int32_t),
                                hipMemcpyDeviceToHost));
            return verify_next_toks_out[M - 1];
        }

        // 4. Final Hyper-Connection Mixer on the last token of the batch (t = M - 1)
        {
            void* last_x = static_cast<uint16_t*>(cur_x) + static_cast<size_t>(M - 1) * 10240;
            void* last_alt = static_cast<uint16_t*>(alt_x) + static_cast<size_t>(M - 1) * 10240;
            void* last_out = static_cast<uint16_t*>(prev_out) + static_cast<size_t>(M - 1) * 2560;
            void* last_inj = static_cast<uint16_t*>(prev_inj) + static_cast<size_t>(M - 1) * 4;
            int64_t zero_delta = 0;
            void* args[] = {&last_x, &last_out, &last_inj, &last_alt, &final_hc_.norm, &final_hc_.down,
                            &d_hc_h_, &d_hc_h_, &d_hc_rstd_, &zero_delta, &one, &eps};
            kernels_.launch_default("hc_premix_c0_u1", args);
            void* m_args[] = {&last_alt, &final_hc_.norm, &d_hc_rstd_, &d_hc_h_, &final_hc_.up, &d_final_hidden_, &one};
            kernels_.launch_default("hc_mix_decode", m_args);
        }

        // 5. LM Head + Argmax / Non-Greedy Sampling on the last token
        const bool use_sampling = (sp && !sp->is_greedy() && rng);
        {
            void* a_lm[] = {&d_final_hidden_, &d_lm_head_, &d_logits_bf16_};
            kernels_.launch_default("bf16_gemv_248320_2560", a_lm);
            if (!use_sampling) {
                void* a_am[] = {&d_logits_bf16_, &d_next_tok_};
                kernels_.launch_default("argmax_248320", a_am);
            }
        }

        int32_t next_id = -1;
        if (use_sampling) {
            next_id = sample_from_device_logits(*sp, history, *rng);
        } else {
            HIP_CHECK(hipMemcpy(&next_id, d_next_tok_, sizeof(int32_t), hipMemcpyDeviceToHost));
        }
        adapt_vram_experts(h_all_routed_ids_.data(), M, /*min_hits=*/2, /*max_swaps=*/12);

        if (dump_logits_out) {
            std::vector<uint16_t> bf(248320);
            HIP_CHECK(hipMemcpy(bf.data(), d_logits_bf16_, 248320 * sizeof(uint16_t), hipMemcpyDeviceToHost));
            dump_logits_out->resize(248320);
            for (size_t i = 0; i < 248320; ++i) {
                (*dump_logits_out)[i] = bf16_bits_to_f32(bf[i]);
            }
        }
        return next_id;
    }

    int32_t sample_from_device_logits(const SamplingParams& sp,
                                      const std::vector<int32_t>* history,
                                      std::mt19937_64& rng) {
        static constexpr int32_t kVocab = 248320;
        uint16_t* h_bf = static_cast<uint16_t*>(h_cpu_upload_);
        HIP_CHECK(hipMemcpy(h_bf, d_logits_bf16_, kVocab * sizeof(uint16_t), hipMemcpyDeviceToHost));

        std::vector<float> logits(kVocab);
        for (int32_t i = 0; i < kVocab; ++i) {
            logits[i] = bf16_bits_to_f32(h_bf[i]);
        }

        if (history && !history->empty() &&
            (sp.presence_penalty != 0.0f || sp.frequency_penalty != 0.0f || sp.repeat_penalty != 1.0f)) {
            std::unordered_map<int32_t, int32_t> counts;
            const size_t n_hist = history->size();
            const size_t start = (n_hist > 1024) ? (n_hist - 1024) : 0;
            for (size_t i = start; i < n_hist; ++i) {
                int32_t tid = (*history)[i];
                if (tid >= 0 && tid < kVocab) counts[tid]++;
            }
            for (const auto& kv : counts) {
                int32_t tid = kv.first;
                int32_t cnt = kv.second;
                float v = logits[tid];
                if (sp.repeat_penalty != 1.0f && sp.repeat_penalty > 0.0f) {
                    v = (v < 0.0f) ? (v * sp.repeat_penalty) : (v / sp.repeat_penalty);
                }
                v -= sp.presence_penalty;
                v -= sp.frequency_penalty * static_cast<float>(cnt);
                logits[tid] = v;
            }
        }

        // Keep top-K candidates (default 256 when top_k <= 0 for fast O(V) partial_sort; exact when top_k > 0)
        int32_t K = (sp.top_k > 0) ? std::min<int32_t>(sp.top_k, kVocab) : 256;
        struct Cand {
            float val;
            int32_t id;
        };
        float max_l = -1e30f;
        int32_t best_id = 0;
        for (int32_t i = 0; i < kVocab; ++i) {
            if (logits[i] > max_l) {
                max_l = logits[i];
                best_id = i;
            }
        }
        if (K <= 1 || sp.temperature <= 0.0f) return best_id;

        // First filter tokens within 20.0 logit units of max_l (exp(-20) ~ 2e-9) to avoid sorting 248K dead tokens
        const float cutoff = max_l - 20.0f * std::max(1.0f, sp.temperature);
        std::vector<Cand> cands;
        cands.reserve(2048);
        for (int32_t i = 0; i < kVocab; ++i) {
            if (logits[i] >= cutoff) {
                cands.push_back(Cand{logits[i], i});
            }
        }
        if (static_cast<int32_t>(cands.size()) < K && sp.top_k > 0) {
            cands.resize(kVocab);
            for (int32_t i = 0; i < kVocab; ++i) cands[i] = Cand{logits[i], i};
        }
        if (cands.empty()) return best_id;

        auto cmp = [](const Cand& a, const Cand& b) {
            return (a.val != b.val) ? (a.val > b.val) : (a.id < b.id);
        };
        if (static_cast<int32_t>(cands.size()) > K) {
            std::partial_sort(cands.begin(), cands.begin() + K, cands.end(), cmp);
            cands.resize(K);
        } else {
            std::sort(cands.begin(), cands.end(), cmp);
        }

        const float inv_temp = 1.0f / std::max(1e-5f, sp.temperature);
        const float top_val = cands[0].val;
        double sum_p = 0.0;
        std::vector<double> probs(cands.size());
        for (size_t i = 0; i < cands.size(); ++i) {
            double p = std::exp(static_cast<double>((cands[i].val - top_val) * inv_temp));
            probs[i] = p;
            sum_p += p;
        }
        if (sum_p <= 0.0) return cands[0].id;

        // Nucleus (top-p) truncation
        size_t keep = cands.size();
        if (sp.top_p > 0.0f && sp.top_p < 1.0f) {
            double cum = 0.0;
            const double thresh = static_cast<double>(sp.top_p) * sum_p;
            for (size_t i = 0; i < cands.size(); ++i) {
                cum += probs[i];
                if (cum >= thresh) {
                    keep = i + 1;
                    sum_p = cum;
                    break;
                }
            }
        }

        std::uniform_real_distribution<double> dist(0.0, sum_p);
        double r = dist(rng);
        double acc = 0.0;
        for (size_t i = 0; i < keep; ++i) {
            acc += probs[i];
            if (r <= acc) return cands[i].id;
        }
        return cands[keep - 1].id;
    }

    void save_recurrent_ckpt() {
        HIP_CHECK(hipMemcpyAsync(d_recurrent_ckpt_, d_recurrent_arena_, recurrent_arena_bytes_,
                                 hipMemcpyDeviceToDevice, nullptr));
        for (int l = 0; l < 48; ++l) {
            ckpt_gdn_conv_in_[l] = d_gdn_conv_in_[l];
            ckpt_gdn_conv_out_[l] = d_gdn_conv_out_[l];
        }
    }

    void restore_recurrent_ckpt() {
        HIP_CHECK(hipMemcpyAsync(d_recurrent_arena_, d_recurrent_ckpt_, recurrent_arena_bytes_,
                                 hipMemcpyDeviceToDevice, nullptr));
        for (int l = 0; l < 48; ++l) {
            d_gdn_conv_in_[l] = ckpt_gdn_conv_in_[l];
            d_gdn_conv_out_[l] = ckpt_gdn_conv_out_[l];
        }
    }

    // Zero-re-run speculative verification finalizer: when accepted < K (n_adv < M), restores the recurrent
    // checkpoint (0.17 ms) and replays ONLY the lightweight PLE + 36 GDN recurrence kernels (~1.8 ms total)
    // using the saved d_ple_gvn_ and d_gdn_verify_qkv_ inputs without re-running projections, QSA, or MoE!
    void finish_verify_step(int32_t M, int32_t n_adv) {
        if (n_adv < M) {
            restore_recurrent_ckpt();
            for (int32_t t = 0; t < n_adv; ++t) {
                void* gvn_t = static_cast<uint16_t*>(d_ple_gvn_) + static_cast<size_t>(t) * 10240;
                void* args[] = {&d_ple_conv_state_, &gvn_t};
                kernels_.launch_default("ple_state_update", args);
            }
            float eps = 1e-6f;
            for (int l = 0; l < 48; ++l) {
                LayerWeights& lw = layers_[l];
                if (!lw.is_qsa) {
                    void* v_qkv = d_gdn_verify_qkv_[l];
                    void* v_z = static_cast<uint16_t*>(v_qkv) + 4 * 10240;
                    void* v_a = static_cast<uint16_t*>(v_qkv) + 4 * 16384;
                    void* v_b = static_cast<uint16_t*>(v_qkv) + 4 * 16432;
                    void* csi = d_gdn_conv_in_[l];
                    void* cso = d_gdn_conv_out_[l];
                    void* st = d_gdn_state_[l];
                    void* args[] = {&v_qkv, &v_z, &v_a, &v_b, &lw.gdn.A_log, &lw.gdn.dt_bias,
                                    &lw.gdn.conv1d, &csi, &cso, &st, &lw.gdn.norm, &d_gdn_out_, &n_adv, &eps};
                    kernels_.launch_default("gated_deltanet_fused", args);
                    std::swap(d_gdn_conv_in_[l], d_gdn_conv_out_[l]);
                }
            }
        }
    }

    void reset_session() {
        HIP_CHECK(hipMemsetAsync(d_recurrent_arena_, 0, recurrent_arena_bytes_, nullptr));
        for (int l = 0; l < 48; ++l) {
            d_gdn_conv_in_[l] = init_gdn_conv_in_[l];
            d_gdn_conv_out_[l] = init_gdn_conv_out_[l];
        }
    }

    void save_turn_ckpt(const std::vector<int32_t>& ids, int32_t next_tok) {
        if (!h_turn_ckpt_) return;
        HIP_CHECK(hipMemcpy(h_turn_ckpt_, d_recurrent_arena_, recurrent_arena_bytes_, hipMemcpyDeviceToHost));
        for (int l = 0; l < 48; ++l) {
            turn_ckpt_gdn_conv_in_[l] = d_gdn_conv_in_[l];
            turn_ckpt_gdn_conv_out_[l] = d_gdn_conv_out_[l];
        }
        turn_ckpt_ids_ = ids;
        turn_ckpt_next_tok_ = next_tok;
    }

    void restore_turn_ckpt(std::vector<int32_t>& history_out, int32_t& next_tok_out) {
        if (!h_turn_ckpt_) return;
        HIP_CHECK(hipMemcpy(d_recurrent_arena_, h_turn_ckpt_, recurrent_arena_bytes_, hipMemcpyHostToDevice));
        for (int l = 0; l < 48; ++l) {
            d_gdn_conv_in_[l] = turn_ckpt_gdn_conv_in_[l];
            d_gdn_conv_out_[l] = turn_ckpt_gdn_conv_out_[l];
        }
        history_out = turn_ckpt_ids_;
        next_tok_out = turn_ckpt_next_tok_;
    }

    const std::vector<int32_t>& turn_ckpt_ids() const { return turn_ckpt_ids_; }
    int total_vram_experts() const { return total_vram_experts_; }
    int32_t max_seq() const { return max_seq_; }
    int32_t vram_cap() const { return vram_cap_; }

private:
    static constexpr int32_t kMaxBatch = 256;

    int32_t run_cpu_moe_layer(int l, int32_t M, bool is_prefill) {
        auto t0 = std::chrono::steady_clock::now();
        while (hipEventQuery(ev_route_) == hipErrorNotReady) {
            _mm_pause();
        }
        auto t1 = std::chrono::steady_clock::now();
        t_wait_acc_ms_ += std::chrono::duration<double, std::milli>(t1 - t0).count();
        const uint16_t* h_mixed = reinterpret_cast<const uint16_t*>(h_route_bell_);
        const int32_t* h_ids = reinterpret_cast<const int32_t*>(
            static_cast<const uint8_t*>(h_route_bell_) + static_cast<size_t>(M) * 2560 * sizeof(uint16_t));
        const int32_t pairs = M * 10;
        int32_t* dst_routed = h_all_routed_ids_.data() + static_cast<size_t>(l) * pairs;

        const MoeWeights& mw = layers_[l].moe;
        struct MissItem {
            int32_t e;
            int32_t pair;
        };
        MissItem misses[kMaxBatch * 10];
        bool tok_needed[kMaxBatch]{};
        int32_t n_miss = 0;
        const bool use_gpu_cold = !is_prefill;
        for (int32_t p = 0; p < pairs; ++p) {
            int32_t raw_id = h_ids[p];
            int32_t e = (raw_id >= 0) ? (raw_id & 0xFFFF) : -1;
            dst_routed[p] = e;
            if (e < 0 || e >= 512) continue;
            bool forced_cpu = (raw_id & 0x10000) != 0;
            if (mw.in_vram[e]) {
                n_vram_exp_acc_++;
            } else if (use_gpu_cold && !forced_cpu && mw.in_gpu_cold[e]) {
                n_gpu_pcie_acc_++;
            }
            bool on_cpu = !mw.in_vram[e] && (!use_gpu_cold || forced_cpu || !mw.in_gpu_cold[e]);
            if (on_cpu) {
                misses[n_miss++] = MissItem{e, p};
                tok_needed[p / 10] = true;
            }
        }
        n_cpu_exp_acc_ += n_miss;
        if (n_miss == 0) return 0;

        if (!is_prefill) {
            // Decode (M == 1): quantize activation once and pre-combine all CPU experts into float32 [2560]
            quantize_act_bf16_2560(h_mixed, act_q_[0], act_d_[0], act_nb_[0]);
            const float* h_wts = reinterpret_cast<const float*>(h_ids + 10);
            float* h_accum_f32 = reinterpret_cast<float*>(h_cpu_upload_);
            for (int32_t i = 0; i < n_miss; ++i) {
                int32_t e = misses[i].e;
                int32_t p = misses[i].pair;
                CpuExpertJob& jb = cpu_jobs_[i];
                jb.exp_base = mw.host_cpu_ptr[e];
                jb.gate_s2 = mw.h_s2[e * 3 + 0];
                jb.up_s2 = mw.h_s2[e * 3 + 1];
                jb.down_s2 = mw.h_s2[e * 3 + 2];
                jb.weight = h_wts[p];
                jb.nt = 1;
                jb.in_vcache = mw.in_vcache[e];
                jb.q_x[0] = act_q_[0];
                jb.d_x[0] = act_d_[0];
                jb.neg_bias_x[0] = act_nb_[0];
                jb.y_f32 = (n_miss == 1) ? h_accum_f32 : (h_accum_f32 + static_cast<size_t>(i + 1) * 2560);
            }
            cpu_pool_.run_decode_batch(cpu_jobs_.data(), n_miss, h_accum_f32);
            HIP_CHECK(hipMemcpyAsync(d_cpu_upload_, h_accum_f32, 2560 * sizeof(float), hipMemcpyHostToDevice, nullptr));
            auto t2 = std::chrono::steady_clock::now();
            t_cpu_acc_ms_ += std::chrono::duration<double, std::milli>(t2 - t1).count();
            return 1;
        }

        // Prefill (M >= 1): sort missed pairs by expert ID so multiple tokens hitting the same expert are grouped into 1 multi-token job!
        if (n_miss > 1) {
            std::sort(misses, misses + n_miss, [](const MissItem& a, const MissItem& b) {
                return (a.e != b.e) ? (a.e < b.e) : (a.pair < b.pair);
            });
        }

        for (int32_t t = 0; t < M; ++t) {
            if (tok_needed[t]) {
                quantize_act_bf16_2560(h_mixed + static_cast<size_t>(t) * 2560,
                                       act_q_[t], act_d_[t], act_nb_[t]);
            }
        }

        uint16_t* h_y_out = reinterpret_cast<uint16_t*>(h_cpu_upload_);
        int32_t upload_pairs[kMaxBatch * 10];
        int32_t n_jobs = 0;
        for (int32_t i = 0; i < n_miss;) {
            int32_t e = misses[i].e;
            int nt = 0;
            CpuExpertJob& jb = cpu_jobs_[n_jobs++];
            jb.exp_base = mw.host_cpu_ptr[e];
            jb.gate_s2 = mw.h_s2[e * 3 + 0];
            jb.up_s2 = mw.h_s2[e * 3 + 1];
            jb.down_s2 = mw.h_s2[e * 3 + 2];
            jb.in_vcache = mw.in_vcache[e];
            while (i < n_miss && misses[i].e == e && nt < kMaxJobToks) {
                int32_t p = misses[i].pair;
                int32_t t = p / 10;
                jb.q_x[nt] = act_q_[t];
                jb.d_x[nt] = act_d_[t];
                jb.neg_bias_x[nt] = act_nb_[t];
                jb.y_out[nt] = h_y_out + static_cast<size_t>(i) * 2560;
                upload_pairs[i] = p;
                nt++;
                i++;
            }
            jb.nt = nt;
        }

        cpu_pool_.run_batch(cpu_jobs_.data(), n_jobs);

        if (n_miss == 1) {
            void* dst = static_cast<uint16_t*>(d_moe_y_) + static_cast<size_t>(upload_pairs[0]) * 2560;
            HIP_CHECK(hipMemcpyAsync(dst, h_y_out, 2560 * sizeof(uint16_t), hipMemcpyHostToDevice, nullptr));
        } else {
            size_t y_bytes = static_cast<size_t>(n_miss) * 2560 * sizeof(uint16_t);
            size_t idx_bytes = static_cast<size_t>(n_miss) * sizeof(int32_t);
            std::memcpy(static_cast<uint8_t*>(h_cpu_upload_) + y_bytes, upload_pairs, idx_bytes);
            HIP_CHECK(hipMemcpyAsync(d_cpu_upload_, h_cpu_upload_, y_bytes + idx_bytes, hipMemcpyHostToDevice, nullptr));
            void* d_src = d_cpu_upload_;
            void* d_idx = static_cast<uint8_t*>(d_cpu_upload_) + y_bytes;
            void* s_args[] = {&d_src, &d_idx, &d_moe_y_, &n_miss};
            kernels_.launch("moe_scatter_rows", n_miss, 1, 1, s_args);
        }
        return 1;
    }

    // Zero-D2H Adaptive VRAM Expert Cache: routes hot experts into dynamic VRAM slots and cold ones back to host RAM.
    void adapt_vram_experts(const int32_t* routed_ids, int32_t M, uint32_t min_hits, int max_swaps) {
        step_counter_++;
        int swaps_done = 0;
        const int32_t pairs_per_layer = M * 10;
        for (int l = 0; l < 48; ++l) {
            MoeWeights& mw = layers_[l].moe;
            if (mw.n_dyn <= 0) continue;
            const int32_t* l_ids = routed_ids + static_cast<size_t>(l) * pairs_per_layer;
            bool ptrs_dirty = false;

            for (int32_t i = 0; i < pairs_per_layer; ++i) {
                int32_t e = l_ids[i];
                if (e < 0 || e >= 512) continue;
                mw.hit_count[e]++;
                if (mw.in_vram[e]) {
                    int16_t s = mw.dyn_slot[e];
                    if (s >= 0) {
                        mw.slot_last_used[s] = step_counter_;
                    }
                } else if (swaps_done < max_swaps && mw.hit_count[e] >= min_hits) {
                    // Find LRU dynamic slot not used in the current step whose occupant has strictly fewer hits
                    int best_s = -1;
                    uint32_t best_age = step_counter_;
                    for (int s = 0; s < mw.n_dyn; ++s) {
                        int32_t cand_old = mw.slot_exp[s];
                        if (mw.slot_last_used[s] < best_age &&
                            (cand_old < 0 || mw.hit_count[e] >= mw.hit_count[cand_old] + min_hits)) {
                            best_age = mw.slot_last_used[s];
                            best_s = s;
                        }
                    }
                    if (best_s >= 0) {
                        int32_t old_e = mw.slot_exp[best_s];
                        // Evict old_e back to host (0 in prefill table; mapped host ptr in decode table iff in_gpu_cold)
                        for (int k = 0; k < 6; ++k) mw.h_ptrs_prefill[old_e * 6 + k] = 0;
                        if (mw.in_gpu_cold[old_e] && mw.host_dev_ptr[old_e]) {
                            uint8_t* old_hdev = mw.host_dev_ptr[old_e];
                            mw.h_ptrs[old_e * 6 + 0] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_GATE_W);
                            mw.h_ptrs[old_e * 6 + 1] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_GATE_S);
                            mw.h_ptrs[old_e * 6 + 2] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_UP_W);
                            mw.h_ptrs[old_e * 6 + 3] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_UP_S);
                            mw.h_ptrs[old_e * 6 + 4] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_DOWN_W);
                            mw.h_ptrs[old_e * 6 + 5] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_DOWN_S);
                        } else {
                            for (int k = 0; k < 6; ++k) mw.h_ptrs[old_e * 6 + k] = 0;
                        }
                        mw.in_vram[old_e] = false;
                        mw.dyn_slot[old_e] = -1;

                        // Promote hot expert e into VRAM slot best_s asynchronously on default stream
                        uint8_t* vslot = mw.dyn_vram_base
                                             ? (mw.dyn_vram_base + static_cast<size_t>(best_s) * EXP_STRIDE_BYTES)
                                             : (mw.vram_base + static_cast<size_t>(mw.n_pinned + best_s) * EXP_STRIDE_BYTES);
                        HIP_CHECK(hipMemcpyAsync(vslot, mw.host_cpu_ptr[e], EXP_STRIDE_BYTES,
                                                 hipMemcpyHostToDevice, nullptr));
                        mw.h_ptrs[e * 6 + 0] = reinterpret_cast<int64_t>(vslot + EXP_OFF_GATE_W);
                        mw.h_ptrs[e * 6 + 1] = reinterpret_cast<int64_t>(vslot + EXP_OFF_GATE_S);
                        mw.h_ptrs[e * 6 + 2] = reinterpret_cast<int64_t>(vslot + EXP_OFF_UP_W);
                        mw.h_ptrs[e * 6 + 3] = reinterpret_cast<int64_t>(vslot + EXP_OFF_UP_S);
                        mw.h_ptrs[e * 6 + 4] = reinterpret_cast<int64_t>(vslot + EXP_OFF_DOWN_W);
                        mw.h_ptrs[e * 6 + 5] = reinterpret_cast<int64_t>(vslot + EXP_OFF_DOWN_S);
                        for (int k = 0; k < 6; ++k) mw.h_ptrs_prefill[e * 6 + k] = mw.h_ptrs[e * 6 + k];
                        mw.in_vram[e] = true;
                        mw.dyn_slot[e] = static_cast<int16_t>(best_s);
                        mw.slot_exp[best_s] = e;
                        mw.slot_last_used[best_s] = step_counter_;
                        ptrs_dirty = true;
                        swaps_done++;
                    }
                }
            }
            if (ptrs_dirty) {
                HIP_CHECK(hipMemcpyAsync(mw.d_ptrs, mw.h_ptrs.data(), mw.h_ptrs.size() * sizeof(int64_t),
                                         hipMemcpyHostToDevice, nullptr));
                HIP_CHECK(hipMemcpyAsync(mw.d_ptrs_prefill, mw.h_ptrs_prefill.data(),
                                         mw.h_ptrs_prefill.size() * sizeof(int64_t), hipMemcpyHostToDevice, nullptr));
            }
        }
    }

    void* gpu_alloc(size_t bytes, bool zero = false) {
        void* p = nullptr;
        HIP_CHECK(hipMalloc(&p, bytes));
        if (zero) HIP_CHECK(hipMemset(p, 0, bytes));
        vram_allocs_.push_back(p);
        return p;
    }

    void* host_pinned_alloc(size_t bytes) {
        void* p = nullptr;
        HIP_CHECK(hipHostMalloc(&p, bytes, hipHostMallocDefault));
        std::memset(p, 0, bytes);
        host_pinned_.push_back(p);
        return p;
    }

    void alloc_state_and_scratch() {
        const size_t B = static_cast<size_t>(kMaxBatch);
        HIP_CHECK(hipEventCreateWithFlags(&ev_mixed_, hipEventDisableTiming));
        HIP_CHECK(hipEventCreateWithFlags(&ev_route_, hipEventDisableTiming));
        if (std::getenv("STRATA_NVFP4_TIMING") != nullptr) {
            for (int i = 0; i < 9; ++i) HIP_CHECK(hipEventCreate(&ev_prof_[i]));
        }
        cpu_jobs_.resize(B * 10);
        h_all_routed_ids_.resize(48 * B * 10, 0);

        d_step_meta_ = gpu_alloc(16 + B * 2560 + B * sizeof(int32_t));
        h_step_meta_ = host_pinned_alloc(16 + B * 2560 + B * sizeof(int32_t));
        d_tok_id_ = d_step_meta_;
        d_pos_i32_ = static_cast<uint8_t*>(d_step_meta_) + 4;
        d_seq_k_i32_ = static_cast<uint8_t*>(d_step_meta_) + 8;
        d_ple_rows_u8_ = static_cast<uint8_t*>(d_step_meta_) + 16;

        d_step_out_ = gpu_alloc((B + 48 * B * 10) * sizeof(int32_t));
        d_next_tok_ = d_step_out_;
        d_moe_ids_ = static_cast<int32_t*>(d_step_out_) + B;

        // Contiguous [d_mixed_ (M * 2560 BF16) | d_cur_ids (M * 10 int32) | d_cur_wts (M * 10 float32)] for 1-transfer D2H routing doorbell
        const size_t bell_bytes = B * (2560 * sizeof(uint16_t) + 10 * sizeof(int32_t) + 10 * sizeof(float));
        d_route_bell_ = gpu_alloc(bell_bytes);
        h_route_bell_ = host_pinned_alloc(bell_bytes);
        d_mixed_ = d_route_bell_;

        // Staging buffer for uploading CPU-computed expert outputs + pair indices to GPU
        const size_t upload_bytes = B * 10 * (2560 * sizeof(uint16_t) + sizeof(int32_t));
        d_cpu_upload_ = gpu_alloc(upload_bytes);
        h_cpu_upload_ = host_pinned_alloc(upload_bytes);

        d_x_stream_a_ = gpu_alloc(B * 10240 * sizeof(uint16_t));
        d_x_stream_b_ = gpu_alloc(B * 10240 * sizeof(uint16_t));
        d_hc_h_ = gpu_alloc(B * 320 * sizeof(uint16_t));
        d_hc_rstd_ = gpu_alloc(B * 4 * sizeof(float));
        d_attn_inj_ = gpu_alloc(B * 4 * sizeof(uint16_t));
        d_mlp_inj_ = gpu_alloc(B * 4 * sizeof(uint16_t));
        d_attn_out_ = gpu_alloc(B * 2560 * sizeof(uint16_t));
        d_mlp_out_ = gpu_alloc(B * 2560 * sizeof(uint16_t));
        d_final_hidden_ = gpu_alloc(2560 * sizeof(uint16_t));
        d_logits_bf16_ = gpu_alloc(4ULL * 248320 * sizeof(uint16_t));
        d_proj_scratch_ = gpu_alloc(B * 16480 * sizeof(uint16_t));

        // PLE state & scratch
        d_ple_embed_ = gpu_alloc(B * 2560 * sizeof(uint16_t));
        d_ple_k_ = gpu_alloc(B * 12800 * sizeof(uint16_t));
        d_ple_v_ = gpu_alloc(B * 2560 * sizeof(uint16_t));
        d_ple_gv_ = gpu_alloc(B * 10240 * sizeof(uint16_t));
        d_ple_gvn_ = gpu_alloc(B * 10240 * sizeof(uint16_t));

        // Contiguous recurrent state arena (PLE conv state + 36 GDN conv_in/conv_out/state + 12 QSA raw_k ring buffers)
        const size_t ple_conv_bytes = 9 * 10240 * sizeof(uint16_t);
        const size_t gdn_conv_bytes = 3 * 10240 * sizeof(uint16_t);
        const size_t gdn_state_bytes = 48 * 128 * 128 * sizeof(float);
        const size_t raw_k_bytes = 1024ULL * 128 * sizeof(uint16_t);
        recurrent_arena_bytes_ = ple_conv_bytes + 36ULL * (2 * gdn_conv_bytes + gdn_state_bytes) + 12ULL * raw_k_bytes;
        d_recurrent_arena_ = gpu_alloc(recurrent_arena_bytes_, true);
        d_recurrent_ckpt_ = gpu_alloc(recurrent_arena_bytes_, false);
        void* h_turn = nullptr;
        if (::posix_memalign(&h_turn, 2 * 1024 * 1024, recurrent_arena_bytes_) == 0 && h_turn) {
            (void) ::madvise(h_turn, recurrent_arena_bytes_, MADV_HUGEPAGE);
            h_turn_ckpt_ = h_turn;
            host_mallocs_.push_back(h_turn);
        }
        uint8_t* rec_cur = static_cast<uint8_t*>(d_recurrent_arena_);
        d_ple_conv_state_ = rec_cur;
        rec_cur += ple_conv_bytes;

        // GDN state & scratch
        d_gdn_qkv_ = gpu_alloc(B * 16480 * sizeof(uint16_t));
        d_gdn_z_ = static_cast<uint16_t*>(d_gdn_qkv_) + 10240;
        d_gdn_a_ = static_cast<uint16_t*>(d_gdn_qkv_) + 16384;
        d_gdn_b_ = static_cast<uint16_t*>(d_gdn_qkv_) + 16432;
        d_gdn_z_b_ = static_cast<uint16_t*>(d_gdn_qkv_) + B * 10240;
        d_gdn_a_b_ = static_cast<uint16_t*>(d_gdn_qkv_) + B * 16384;
        d_gdn_b_b_ = static_cast<uint16_t*>(d_gdn_qkv_) + B * 16432;
        d_gdn_out_ = gpu_alloc(B * 6144 * sizeof(uint16_t));
        const size_t gdn_verify_layer_bytes = 4ULL * 16480 * sizeof(uint16_t);
        d_gdn_verify_arena_ = gpu_alloc(36ULL * gdn_verify_layer_bytes);
        uint8_t* ver_cur = static_cast<uint8_t*>(d_gdn_verify_arena_);
        for (int l = 0; l < 48; ++l) {
            if (l % 4 != 3) {
                d_gdn_conv_in_[l] = rec_cur;
                init_gdn_conv_in_[l] = rec_cur;
                rec_cur += gdn_conv_bytes;
                d_gdn_conv_out_[l] = rec_cur;
                init_gdn_conv_out_[l] = rec_cur;
                rec_cur += gdn_conv_bytes;
                d_gdn_state_[l] = rec_cur;
                rec_cur += gdn_state_bytes;
                d_gdn_verify_qkv_[l] = ver_cur;
                ver_cur += gdn_verify_layer_bytes;
            }
        }

        // QSA state & scratch
        d_qsa_qg_ = gpu_alloc(B * 13952 * sizeof(uint16_t));
        d_qsa_k_ = static_cast<uint16_t*>(d_qsa_qg_) + 12288;
        d_qsa_v_ = static_cast<uint16_t*>(d_qsa_qg_) + 12800;
        d_qsa_idx_qk_ = static_cast<uint16_t*>(d_qsa_qg_) + 13312;
        d_qsa_k_b_ = static_cast<uint16_t*>(d_qsa_qg_) + B * 12288;
        d_qsa_v_b_ = static_cast<uint16_t*>(d_qsa_qg_) + B * 12800;
        d_qsa_idx_qk_b_ = static_cast<uint16_t*>(d_qsa_qg_) + B * 13312;
        d_qsa_q_ = gpu_alloc(B * 6144 * sizeof(uint16_t));
        d_qsa_attn_raw_ = gpu_alloc(B * 6144 * sizeof(uint16_t));
        d_qsa_sel_ids_ = gpu_alloc(B * 512 * sizeof(int32_t));
        d_zero_mask_ = gpu_alloc(2048 * sizeof(float), true);
        int max_splits = 10; // dense Lk <= 2048 uses <= 8 splits; sparse QSA uses 9 splits (2052 keys)
        d_attn_pm_ = gpu_alloc(24 * max_splits * sizeof(float));
        d_attn_pl_ = gpu_alloc(24 * max_splits * sizeof(float));
        d_attn_pa_ = gpu_alloc(24 * max_splits * 256 * sizeof(float));

        // 1. Raw K ring buffer: only the most recent 1,024 tokens are ever needed to pool 4-token blocks!
        //    12 layers x 1024 x 128 x 2 B = 3.0 MiB total in VRAM (placed inside d_recurrent_arena_ for turn checkpointing).
        for (int q = 0; q < 12; ++q) {
            d_raw_k_cache_[q] = rec_cur;
            rec_cur += raw_k_bytes;
        }

        // 2. Elastic VRAM Resident KV + Block-K Arena via Strata VmmRange (2 MiB granularity):
        //    Each QSA layer gets a contiguous virtual address space in kv_vmm_range_ for up to vram_cap_ tokens
        //    (vram_prefix_ prefix tokens + 4,096 local sliding-window ring tokens), plus vram_blocks_ block keys.
        const size_t gran = strata::core::vmm_available()
                                ? static_cast<size_t>(strata::core::vmm_granularity())
                                : (2ULL * 1024 * 1024);
        vram_kv_stride_bytes_ = ((static_cast<size_t>(vram_cap_) * 2 * 160) + gran - 1) & ~(gran - 1);
        vram_bk_stride_bytes_ = ((static_cast<size_t>(vram_blocks_) * 128 * sizeof(uint16_t)) + gran - 1) & ~(gran - 1);
        const size_t total_kv_vmm_bytes = 12ULL * (2 * vram_kv_stride_bytes_ + vram_bk_stride_bytes_);

        kv_vmm_ok_ = strata::core::vmm_available() && kv_vmm_range_.reserve(total_kv_vmm_bytes);
        const int32_t init_vram_seq = kv_grow_ ? std::min<int32_t>(vram_cap_, 6553) : vram_cap_;
        const size_t init_kv_map = ((static_cast<size_t>(init_vram_seq) * 2 * 160) + gran - 1) & ~(gran - 1);
        const size_t init_bk_map = ((static_cast<size_t>(std::min<int32_t>(vram_blocks_, (init_vram_seq + 3) / 4)) * 128 * sizeof(uint16_t)) + gran - 1) & ~(gran - 1);

        auto no_take = []() -> strata::core::VmmChunk { return 0; };
        if (kv_vmm_ok_) {
            uint8_t* vbase = kv_vmm_range_.base();
            for (int q = 0; q < 12; ++q) {
                size_t k_off = static_cast<size_t>(q) * 2 * vram_kv_stride_bytes_;
                size_t v_off = k_off + vram_kv_stride_bytes_;
                size_t b_off = 24ULL * vram_kv_stride_bytes_ + static_cast<size_t>(q) * vram_bk_stride_bytes_;
                d_k_cache_[q] = vbase + k_off;
                d_v_cache_[q] = vbase + v_off;
                d_block_k_cache_[q] = vbase + b_off;
                int64_t k_lo = static_cast<int64_t>(k_off / gran);
                int64_t k_hi = static_cast<int64_t>((k_off + init_kv_map) / gran);
                int64_t v_lo = static_cast<int64_t>(v_off / gran);
                int64_t v_hi = static_cast<int64_t>((v_off + init_kv_map) / gran);
                int64_t b_lo = static_cast<int64_t>(b_off / gran);
                int64_t b_hi = static_cast<int64_t>((b_off + init_bk_map) / gran);
                if (!kv_vmm_range_.map_range(k_lo, k_hi, no_take) ||
                    !kv_vmm_range_.map_range(v_lo, v_hi, no_take) ||
                    (init_bk_map > 0 && !kv_vmm_range_.map_range(b_lo, b_hi, no_take))) {
                    std::fprintf(stderr, "[flydsl-nvfp4] VmmRange initial KV map failed\n");
                    std::exit(1);
                }
                HIP_CHECK(hipMemset(d_k_cache_[q], 0, init_kv_map));
                HIP_CHECK(hipMemset(d_v_cache_[q], 0, init_kv_map));
                if (init_bk_map > 0) HIP_CHECK(hipMemset(d_block_k_cache_[q], 0, init_bk_map));
            }
            vram_kv_mapped_bytes_ = init_kv_map;
            vram_bk_mapped_bytes_ = init_bk_map;
            kv_mapped_seq_ = static_cast<int32_t>(std::min<size_t>(vram_prefix_, (init_kv_map / 320)));
        } else {
            for (int q = 0; q < 12; ++q) {
                d_k_cache_[q] = gpu_alloc(vram_kv_stride_bytes_, true);
                d_v_cache_[q] = gpu_alloc(vram_kv_stride_bytes_, true);
                if (vram_bk_stride_bytes_ > 0) {
                    d_block_k_cache_[q] = gpu_alloc(vram_bk_stride_bytes_, true);
                }
            }
            vram_kv_mapped_bytes_ = vram_kv_stride_bytes_;
            vram_bk_mapped_bytes_ = vram_bk_stride_bytes_;
            kv_mapped_seq_ = vram_prefix_;
        }

        // 3. Elastic Host RAM Overflow KV + Block-K Arena (for tokens beyond vram_prefix_ up to max_seq_):
        //    Reserved lazily via mmap(MAP_NORESERVE) so startup physical RAM usage is 0 B!
        //    As context grows past vram_prefix_, 2 MiB slabs are faulted and mapped via hipHostRegisterMapped.
        const int32_t host_overflow_toks = std::max<int32_t>(0, max_seq_ - vram_prefix_);
        const int32_t host_overflow_blks = std::max<int32_t>(0, max_blocks_ - vram_blocks_);
        host_kv_stride_bytes_ = (host_overflow_toks > 0)
                                    ? (((static_cast<size_t>(host_overflow_toks) * 2 * 160) + gran - 1) & ~(gran - 1))
                                    : 0;
        host_bk_stride_bytes_ = (host_overflow_blks > 0)
                                    ? (((static_cast<size_t>(host_overflow_blks) * 128 * sizeof(uint16_t)) + gran - 1) & ~(gran - 1))
                                    : 0;

        if (host_kv_stride_bytes_ > 0) {
            host_kv_mmap_bytes_ = 24ULL * host_kv_stride_bytes_;
            void* p = ::mmap(nullptr, host_kv_mmap_bytes_, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (p == MAP_FAILED) {
                std::fprintf(stderr, "[flydsl-nvfp4] mmap failed for elastic host KV overflow range\n");
                std::exit(1);
            }
            host_kv_mmap_ = static_cast<uint8_t*>(p);
            for (int q = 0; q < 12; ++q) {
                uint8_t* hk = host_kv_mmap_ + static_cast<size_t>(2 * q) * host_kv_stride_bytes_;
                uint8_t* hv = host_kv_mmap_ + static_cast<size_t>(2 * q + 1) * host_kv_stride_bytes_;
                kc_host_delta_[q] = reinterpret_cast<int64_t>(hk) - reinterpret_cast<int64_t>(d_k_cache_[q]);
                vc_host_delta_[q] = reinterpret_cast<int64_t>(hv) - reinterpret_cast<int64_t>(d_v_cache_[q]);
            }
        }
        if (host_bk_stride_bytes_ > 0) {
            host_bk_mmap_bytes_ = 12ULL * host_bk_stride_bytes_;
            void* p = ::mmap(nullptr, host_bk_mmap_bytes_, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (p == MAP_FAILED) {
                std::fprintf(stderr, "[flydsl-nvfp4] mmap failed for elastic host block-key range\n");
                std::exit(1);
            }
            host_bk_mmap_ = static_cast<uint8_t*>(p);
            for (int q = 0; q < 12; ++q) {
                uint8_t* hb = host_bk_mmap_ + static_cast<size_t>(q) * host_bk_stride_bytes_;
                if (!d_block_k_cache_[q]) d_block_k_cache_[q] = hb;
                bk_host_delta_elems_[q] = (reinterpret_cast<int64_t>(hb) - reinterpret_cast<int64_t>(d_block_k_cache_[q])) / 2;
            }
        }
        // Shared QSA block-score buffer in VRAM (16.0 MiB for batched multi-token long-context prefill + 1M-token decode)
        d_qsa_scores_ = gpu_alloc(std::max<size_t>(4194304, static_cast<size_t>(max_blocks_)) * sizeof(float), true);

        // MoE scratch
        d_sh_gu_ = gpu_alloc(B * 1856 * sizeof(uint16_t), true);
        d_router_logits_ = static_cast<uint16_t*>(d_sh_gu_) + 1280;
        d_sh_scale_ = static_cast<uint16_t*>(d_sh_gu_) + 1792;
        d_sh_inter_ = gpu_alloc(B * 640 * sizeof(uint16_t));
        d_sh_out_ = gpu_alloc(B * 2560 * sizeof(uint16_t));
        d_moe_wts_ = gpu_alloc(B * 16 * sizeof(float));
        const size_t max_moe_tiles = (B * 10 + 15) / 16 + 512;
        d_moe_sorted_ = gpu_alloc(max_moe_tiles * 16 * sizeof(int32_t));
        d_moe_tile_e_ = gpu_alloc(max_moe_tiles * sizeof(int32_t));
        d_moe_inter_ = gpu_alloc(max_moe_tiles * 16 * 640 * sizeof(uint16_t));
        d_moe_y_ = gpu_alloc(B * 16 * 2560 * sizeof(uint16_t));
    }

    void populate_rope_range(int32_t p_begin, int32_t p_end) {
        if (p_end <= p_begin) return;
        double inv[32];
        for (int i = 0; i < 32; ++i) {
            inv[i] = std::pow(rope_scaling_.freq_base, -static_cast<double>(2 * i) / 64.0);
        }
        float* cf32 = static_cast<float*>(d_rope_cos_f32_);
        float* sf32 = static_cast<float*>(d_rope_sin_f32_);
        uint16_t* cbf16 = static_cast<uint16_t*>(d_rope_cos_bf16_);
        uint16_t* sbf16 = static_cast<uint16_t*>(d_rope_sin_bf16_);

        const strata::kernels::RopeKernelArgs static_ka = rope_scaling_.kernel_args(64);
        const int32_t orig_ctx = static_cast<int32_t>(rope_scaling_.orig_ctx);

        for (int32_t p = p_begin; p < p_end; ++p) {
            // In auto mode (Dynamic YaRN), positions < orig_ctx (262,144) use exact unscaled angles (scale = 1.0),
            // and positions >= orig_ctx dynamically scale via YaRN with factor = (p + 1) / orig_ctx.
            strata::kernels::RopeKernelArgs ka = static_ka;
            bool unscaled = (rope_scaling_.type == strata::kernels::RopeScalingType::None);
            if (rope_auto_yarn_) {
                if (p < orig_ctx) {
                    unscaled = true;
                } else {
                    unscaled = false;
                    strata::kernels::RopeScaling dyn_rs = rope_scaling_;
                    dyn_rs.type = strata::kernels::RopeScalingType::YaRN;
                    dyn_rs.factor = std::max(1.0, static_cast<double>(p + 1) / static_cast<double>(orig_ctx));
                    dyn_rs.ext_factor = 1.0;
                    ka = dyn_rs.kernel_args(64);
                }
            }
            for (int i = 0; i < 32; ++i) {
                float c = 1.0f, s = 0.0f;
                if (unscaled) {
                    double ang = static_cast<double>(p) * inv[i];
                    c = static_cast<float>(std::cos(ang));
                    s = static_cast<float>(std::sin(ang));
                } else {
                    float theta_extrap = static_cast<float>(static_cast<double>(p) * inv[i]);
                    strata::kernels::rope_scaled_angle(theta_extrap, ka.freq_scale, ka.corr_low, ka.corr_high,
                                                       ka.ext_factor, ka.attn_factor, i, c, s);
                }
                cf32[static_cast<size_t>(p) * 32 + i] = c;
                sf32[static_cast<size_t>(p) * 32 + i] = s;
                uint16_t cb = f32_to_bf16_bits(c);
                uint16_t sb = f32_to_bf16_bits(s);
                cbf16[static_cast<size_t>(p) * 64 + i] = cb;
                cbf16[static_cast<size_t>(p) * 64 + 32 + i] = cb;
                sbf16[static_cast<size_t>(p) * 64 + i] = sb;
                sbf16[static_cast<size_t>(p) * 64 + 32 + i] = sb;
            }
        }
    }

    void init_rope_tables() {
        const size_t gran = strata::core::vmm_available()
                                ? static_cast<size_t>(strata::core::vmm_granularity())
                                : (2ULL * 1024 * 1024);
        rope_stride_f32_bytes_ = ((static_cast<size_t>(max_seq_) * 32 * sizeof(float)) + gran - 1) & ~(gran - 1);
        rope_stride_bf16_bytes_ = ((static_cast<size_t>(max_seq_) * 64 * sizeof(uint16_t)) + gran - 1) & ~(gran - 1);
        host_rope_mmap_bytes_ = 2 * rope_stride_f32_bytes_ + 2 * rope_stride_bf16_bytes_;

        void* p = ::mmap(nullptr, host_rope_mmap_bytes_, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) {
            std::fprintf(stderr, "[flydsl-nvfp4] mmap failed for elastic RoPE tables\n");
            std::exit(1);
        }
        host_rope_mmap_ = static_cast<uint8_t*>(p);
        d_rope_cos_f32_ = host_rope_mmap_;
        d_rope_sin_f32_ = host_rope_mmap_ + rope_stride_f32_bytes_;
        d_rope_cos_bf16_ = host_rope_mmap_ + 2 * rope_stride_f32_bytes_;
        d_rope_sin_bf16_ = host_rope_mmap_ + 2 * rope_stride_f32_bytes_ + rope_stride_bf16_bytes_;

        rope_populated_seq_ = 0;
        rope_mapped_bytes_ = 0;
        ensure_rope_capacity(kv_mapped_seq_);
    }

    void ensure_rope_capacity(int32_t needed_seq) {
        int32_t target_seq = std::min<int32_t>(max_seq_, std::max<int32_t>(needed_seq, 16384));
        const size_t gran = strata::core::vmm_available()
                                ? static_cast<size_t>(strata::core::vmm_granularity())
                                : (2ULL * 1024 * 1024);
        size_t need_bytes = ((static_cast<size_t>(target_seq) * 32 * sizeof(float)) + gran - 1) & ~(gran - 1);
        need_bytes = std::min(need_bytes, rope_stride_f32_bytes_);
        if (need_bytes > rope_mapped_bytes_) {
            size_t add_bytes = need_bytes - rope_mapped_bytes_;
            uint8_t* ptrs[4] = {
                static_cast<uint8_t*>(d_rope_cos_f32_) + rope_mapped_bytes_,
                static_cast<uint8_t*>(d_rope_sin_f32_) + rope_mapped_bytes_,
                static_cast<uint8_t*>(d_rope_cos_bf16_) + rope_mapped_bytes_,
                static_cast<uint8_t*>(d_rope_sin_bf16_) + rope_mapped_bytes_,
            };
            for (int k = 0; k < 4; ++k) {
                (void) ::madvise(ptrs[k], add_bytes, MADV_HUGEPAGE);
                std::memset(ptrs[k], 0, add_bytes);
                HIP_CHECK(hipHostRegister(ptrs[k], add_bytes, hipHostRegisterMapped));
                host_slab_registered_.push_back(ptrs[k]);
            }
            rope_mapped_bytes_ = need_bytes;
        }
        int32_t max_pop = static_cast<int32_t>(std::min<size_t>(max_seq_, rope_mapped_bytes_ / (32 * sizeof(float))));
        if (rope_populated_seq_ < max_pop) {
            populate_rope_range(rope_populated_seq_, max_pop);
            rope_populated_seq_ = max_pop;
        }
    }

    // Reclaims 1 physical 2 MiB VMM chunk from the tail of the dynamic VRAM expert pool if free VRAM is tight
    // when growing the VRAM resident KV window from 6,553 -> vram_cap_ tokens, returning the unmapped VmmChunk handle.
    strata::core::VmmChunk take_one_dynamic_expert_chunk() {
        if (!exp_vmm_ok_ || exp_vmm_mapped_bytes_ == 0) return 0;
        const size_t gran = strata::core::vmm_available()
                                ? static_cast<size_t>(strata::core::vmm_granularity())
                                : (2ULL * 1024 * 1024);
        if (exp_vmm_mapped_bytes_ < gran) return 0;
        const size_t new_mapped_bytes = exp_vmm_mapped_bytes_ - gran;
        const size_t max_valid_slots = new_mapped_bytes / EXP_STRIDE_BYTES;

        // Evict any dynamic VRAM expert slot across the 48 layers whose global slot index >= max_valid_slots
        for (int l = 0; l < 48; ++l) {
            MoeWeights& mw = layers_[l].moe;
            bool dirty = false;
            for (int s = mw.n_dyn - 1; s >= 0; --s) {
                size_t g_slot = exp_layer_dyn_slot_off_[l] + static_cast<size_t>(s);
                if (g_slot >= max_valid_slots) {
                    int32_t old_e = mw.slot_exp[s];
                    if (old_e >= 0 && old_e < 512) {
                        for (int k = 0; k < 6; ++k) mw.h_ptrs_prefill[old_e * 6 + k] = 0;
                        if (mw.in_gpu_cold[old_e]) {
                            uint8_t* old_hdev = mw.host_dev_ptr[old_e];
                            mw.h_ptrs[old_e * 6 + 0] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_GATE_W);
                            mw.h_ptrs[old_e * 6 + 1] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_GATE_S);
                            mw.h_ptrs[old_e * 6 + 2] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_UP_W);
                            mw.h_ptrs[old_e * 6 + 3] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_UP_S);
                            mw.h_ptrs[old_e * 6 + 4] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_DOWN_W);
                            mw.h_ptrs[old_e * 6 + 5] = reinterpret_cast<int64_t>(old_hdev + EXP_OFF_DOWN_S);
                        } else {
                            for (int k = 0; k < 6; ++k) mw.h_ptrs[old_e * 6 + k] = 0;
                        }
                        mw.in_vram[old_e] = false;
                        mw.dyn_slot[old_e] = -1;
                        mw.slot_exp[s] = -1;
                        total_vram_experts_--;
                        dirty = true;
                    }
                    mw.n_dyn = s;
                    mw.vram_experts = mw.n_pinned + mw.n_dyn;
                }
            }
            if (dirty) {
                HIP_CHECK(hipMemcpy(mw.d_ptrs, mw.h_ptrs.data(), mw.h_ptrs.size() * sizeof(int64_t),
                                    hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(mw.d_ptrs_prefill, mw.h_ptrs_prefill.data(),
                                    mw.h_ptrs_prefill.size() * sizeof(int64_t), hipMemcpyHostToDevice));
            }
        }
        HIP_CHECK(hipDeviceSynchronize());
        int64_t chunk_idx = static_cast<int64_t>(new_mapped_bytes / gran);
        strata::core::VmmChunk h = exp_vmm_range_.unmap(chunk_idx);
        if (h == 0) return 0;
        exp_vmm_mapped_bytes_ = new_mapped_bytes;
        return h;
    }

    bool map_kv_vmm_with_reclaim(size_t offset, size_t length) {
        const size_t gran = strata::core::vmm_available()
                                ? static_cast<size_t>(strata::core::vmm_granularity())
                                : (2ULL * 1024 * 1024);
        int64_t lo = static_cast<int64_t>(offset / gran);
        int64_t hi = static_cast<int64_t>((offset + length) / gran);
        auto take_fn = [&]() -> strata::core::VmmChunk {
            size_t free_vram = 0, total_vram = 0;
            const size_t min_free = static_cast<size_t>(std::max(512, vram_reserve_mib_ - 256)) * 1024ULL * 1024ULL;
            if (hipMemGetInfo(&free_vram, &total_vram) == hipSuccess &&
                free_vram >= gran + min_free) {
                return 0; // Let VmmRange::map_range allocate a fresh chunk via vmm_chunk_new()
            }
            return take_one_dynamic_expert_chunk();
        };
        if (kv_vmm_range_.map_range(lo, hi, take_fn)) return true;
        // Fallback if vmm_chunk_new() failed despite hipMemGetInfo reporting free space: force-take from dynamic experts
        return kv_vmm_range_.map_range(lo, hi, [&]() { return take_one_dynamic_expert_chunk(); });
    }

    // Dynamically grows the active context capacity (VRAM VmmRange chunks + Host RAM overflow slabs + RoPE tables)
    // on demand as `needed_seq = pos + M` increases, up to `max_seq_`.
    void ensure_context_capacity(int32_t needed_seq) {
        if (needed_seq > max_seq_) {
            std::fprintf(stderr, "[flydsl-nvfp4] Context limit reached (%d > max_seq %d)\n", needed_seq, max_seq_);
            std::exit(1);
        }
        ensure_rope_capacity(needed_seq);
        if (needed_seq <= kv_mapped_seq_) return;

        const size_t gran = strata::core::vmm_available()
                                ? static_cast<size_t>(strata::core::vmm_granularity())
                                : (2ULL * 1024 * 1024);
        // 1. If VRAM resident KV cache is not yet fully mapped up to vram_cap_, grow kv_vmm_range_ in 2 MiB steps
        if (kv_vmm_ok_ && vram_kv_mapped_bytes_ < vram_kv_stride_bytes_) {
            // Once needed_seq exceeds vram_prefix_, the 4,096-token ring buffer at [vram_prefix_, vram_prefix_ + 4096)
            // is active, so we need all vram_cap_ tokens mapped in VRAM.
            int32_t want_vram_seq = (needed_seq > vram_prefix_) ? vram_cap_ : std::min<int32_t>(vram_cap_, needed_seq);
            size_t want_kv_bytes = ((static_cast<size_t>(want_vram_seq) * 2 * 160) + gran - 1) & ~(gran - 1);
            want_kv_bytes = std::min(want_kv_bytes, vram_kv_stride_bytes_);
            if (want_kv_bytes > vram_kv_mapped_bytes_) {
                size_t add_kv = want_kv_bytes - vram_kv_mapped_bytes_;
                uint8_t* vbase = kv_vmm_range_.base();
                for (int q = 0; q < 12; ++q) {
                    size_t k_off = static_cast<size_t>(q) * 2 * vram_kv_stride_bytes_ + vram_kv_mapped_bytes_;
                    size_t v_off = static_cast<size_t>(q) * 2 * vram_kv_stride_bytes_ + vram_kv_stride_bytes_ + vram_kv_mapped_bytes_;
                    if (!map_kv_vmm_with_reclaim(k_off, add_kv) || !map_kv_vmm_with_reclaim(v_off, add_kv)) {
                        std::fprintf(stderr, "[flydsl-nvfp4] Failed to grow VRAM KV VmmRange to %zu bytes\n", want_kv_bytes);
                        std::exit(1);
                    }
                    HIP_CHECK(hipMemset(vbase + k_off, 0, add_kv));
                    HIP_CHECK(hipMemset(vbase + v_off, 0, add_kv));
                }
                vram_kv_mapped_bytes_ = want_kv_bytes;
            }

            int32_t want_vram_blks = std::min<int32_t>(vram_blocks_, (needed_seq + 3) / 4);
            size_t want_bk_bytes = ((static_cast<size_t>(want_vram_blks) * 128 * sizeof(uint16_t)) + gran - 1) & ~(gran - 1);
            want_bk_bytes = std::min(want_bk_bytes, vram_bk_stride_bytes_);
            if (want_bk_bytes > vram_bk_mapped_bytes_) {
                size_t add_bk = want_bk_bytes - vram_bk_mapped_bytes_;
                uint8_t* vbase = kv_vmm_range_.base();
                for (int q = 0; q < 12; ++q) {
                    size_t b_off = 24ULL * vram_kv_stride_bytes_ + static_cast<size_t>(q) * vram_bk_stride_bytes_ + vram_bk_mapped_bytes_;
                    if (!map_kv_vmm_with_reclaim(b_off, add_bk)) {
                        std::fprintf(stderr, "[flydsl-nvfp4] Failed to grow VRAM Block-K VmmRange to %zu bytes\n", want_bk_bytes);
                        std::exit(1);
                    }
                    HIP_CHECK(hipMemset(vbase + b_off, 0, add_bk));
                }
                vram_bk_mapped_bytes_ = want_bk_bytes;
            }
        }

        // 2. If needed_seq exceeds vram_prefix_, grow the Host RAM overflow KV + Block-K slabs in 2 MiB increments
        if (needed_seq > vram_prefix_ && host_kv_mmap_) {
            int32_t over_toks = needed_seq - vram_prefix_;
            size_t want_hkv_bytes = ((static_cast<size_t>(over_toks) * 2 * 160) + gran - 1) & ~(gran - 1);
            want_hkv_bytes = std::min(want_hkv_bytes, host_kv_stride_bytes_);
            if (want_hkv_bytes > host_kv_mapped_bytes_) {
                size_t add_hkv = want_hkv_bytes - host_kv_mapped_bytes_;
                for (int q = 0; q < 12; ++q) {
                    uint8_t* hk = host_kv_mmap_ + static_cast<size_t>(2 * q) * host_kv_stride_bytes_ + host_kv_mapped_bytes_;
                    uint8_t* hv = host_kv_mmap_ + static_cast<size_t>(2 * q + 1) * host_kv_stride_bytes_ + host_kv_mapped_bytes_;
                    (void) ::madvise(hk, add_hkv, MADV_HUGEPAGE);
                    (void) ::madvise(hv, add_hkv, MADV_HUGEPAGE);
                    std::memset(hk, 0, add_hkv);
                    std::memset(hv, 0, add_hkv);
                    HIP_CHECK(hipHostRegister(hk, add_hkv, hipHostRegisterMapped));
                    HIP_CHECK(hipHostRegister(hv, add_hkv, hipHostRegisterMapped));
                    host_slab_registered_.push_back(hk);
                    host_slab_registered_.push_back(hv);
                }
                host_kv_mapped_bytes_ = want_hkv_bytes;
            }
        }
        if (needed_seq > vram_blocks_ * 4 && host_bk_mmap_ && host_bk_stride_bytes_ > 0) {
            int32_t over_blks = ((needed_seq + 3) / 4) - vram_blocks_;
            size_t want_hbk_bytes = ((static_cast<size_t>(over_blks) * 128 * sizeof(uint16_t)) + gran - 1) & ~(gran - 1);
            want_hbk_bytes = std::min(want_hbk_bytes, host_bk_stride_bytes_);
            if (want_hbk_bytes > host_bk_mapped_bytes_) {
                size_t add_hbk = want_hbk_bytes - host_bk_mapped_bytes_;
                for (int q = 0; q < 12; ++q) {
                    uint8_t* hb = host_bk_mmap_ + static_cast<size_t>(q) * host_bk_stride_bytes_ + host_bk_mapped_bytes_;
                    (void) ::madvise(hb, add_hbk, MADV_HUGEPAGE);
                    std::memset(hb, 0, add_hbk);
                    HIP_CHECK(hipHostRegister(hb, add_hbk, hipHostRegisterMapped));
                    host_slab_registered_.push_back(hb);
                }
                host_bk_mapped_bytes_ = want_hbk_bytes;
            }
        }

        int32_t vram_cov = static_cast<int32_t>(std::min<size_t>(vram_prefix_, vram_kv_mapped_bytes_ / 320));
        int32_t host_cov = static_cast<int32_t>(host_kv_mapped_bytes_ / 320);
        int32_t bk_cov = (vram_blocks_ + static_cast<int32_t>(host_bk_mapped_bytes_ / 256)) * 4;
        kv_mapped_seq_ = std::min(bk_cov, (host_cov > 0) ? (vram_prefix_ + host_cov) : vram_cov);
    }

    void load_weights_from_safetensors() {
        ple_consts_ = strata::kernels::ple_artifact_consts();

        // 1. Open PLE shard (model-fp8-mtp-ple.safetensors) and record the 128 shard file offsets
        std::string ple_path = model_dir_ + "/model-fp8-mtp-ple.safetensors";
        ple_.ple_fd = ::open(ple_path.c_str(), O_RDONLY);
        if (ple_.ple_fd < 0) {
            std::fprintf(stderr, "Cannot open PLE safetensors shard: %s\n", ple_path.c_str());
            std::exit(1);
        }
        std::unordered_map<std::string, TensorSlice> ple_index;
        parse_safetensors_header(ple_.ple_fd, 0, ple_index);
        for (int s = 0; s < 128; ++s) {
            std::string tname = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_" +
                                std::to_string(s) + ".weight";
            auto it = ple_index.find(tname);
            if (it == ple_index.end()) {
                std::fprintf(stderr, "Missing PLE shard tensor: %s\n", tname.c_str());
                std::exit(1);
            }
            ple_.shard_offsets[s] = it->second.file_offset;
        }
        {
            auto it = ple_index.find("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale");
            if (it == ple_index.end()) {
                std::fprintf(stderr, "Missing PLE weight_scale\n");
                std::exit(1);
            }
            uint16_t sc_bf16 = 0;
            pread_exact(ple_.ple_fd, &sc_bf16, sizeof(uint16_t), it->second.file_offset);
            ple_.weight_scale = gpu_alloc(sizeof(uint16_t));
            HIP_CHECK(hipMemcpy(ple_.weight_scale, &sc_bf16, sizeof(uint16_t), hipMemcpyHostToDevice));
        }

        // 2. Parse headers of model-00001-of-00010.safetensors .. model-00010-of-00010.safetensors
        std::unordered_map<std::string, TensorSlice> main_index;
        int shard_fds[10]{};
        for (int s = 0; s < 10; ++s) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "/model-%05d-of-00010.safetensors", s + 1);
            std::string spath = model_dir_ + buf;
            shard_fds[s] = ::open(spath.c_str(), O_RDONLY);
            if (shard_fds[s] < 0) {
                std::fprintf(stderr, "Cannot open shard: %s\n", spath.c_str());
                std::exit(1);
            }
            parse_safetensors_header(shard_fds[s], s, main_index);
            (void) ::posix_fadvise(shard_fds[s], 0, 0, POSIX_FADV_DONTNEED);
        }

        std::vector<uint8_t> stage_buf(16 * 1024 * 1024);
        auto load_vram_tensor = [&](const std::string& name) -> void* {
            auto it = main_index.find(name);
            if (it == main_index.end()) {
                std::fprintf(stderr, "Missing tensor in safetensors: %s\n", name.c_str());
                std::exit(1);
            }
            const TensorSlice& ts = it->second;
            void* d_ptr = gpu_alloc(ts.byte_len);
            if (stage_buf.size() < ts.byte_len) stage_buf.resize(ts.byte_len);
            pread_exact(shard_fds[ts.shard_idx], stage_buf.data(), ts.byte_len, ts.file_offset);
            HIP_CHECK(hipMemcpy(d_ptr, stage_buf.data(), ts.byte_len, hipMemcpyHostToDevice));
            return d_ptr;
        };

        auto load_vram_tensor_concat = [&](const std::vector<std::string>& names, size_t pad_bytes = 0) -> void* {
            size_t total_bytes = pad_bytes;
            for (const auto& name : names) {
                auto it = main_index.find(name);
                if (it == main_index.end()) {
                    std::fprintf(stderr, "Missing tensor in safetensors: %s\n", name.c_str());
                    std::exit(1);
                }
                total_bytes += it->second.byte_len;
            }
            void* d_ptr = gpu_alloc(total_bytes, pad_bytes > 0);
            size_t cur_off = 0;
            for (const auto& name : names) {
                const TensorSlice& ts = main_index.at(name);
                if (stage_buf.size() < ts.byte_len) stage_buf.resize(ts.byte_len);
                pread_exact(shard_fds[ts.shard_idx], stage_buf.data(), ts.byte_len, ts.file_offset);
                HIP_CHECK(hipMemcpy(static_cast<uint8_t*>(d_ptr) + cur_off, stage_buf.data(), ts.byte_len,
                                    hipMemcpyHostToDevice));
                cur_off += ts.byte_len;
            }
            return d_ptr;
        };

        auto load_hc = [&](const std::string& pfx, bool combine) -> HcWeights {
            HcWeights hc;
            hc.norm = load_vram_tensor(pfx + ".hc_norm.weight");
            hc.down = load_vram_tensor(pfx + ".input_mix_weight_down.weight");
            hc.up = load_vram_tensor(pfx + ".input_mix_weight_up.weight");
            if (combine) {
                hc.inj = load_vram_tensor(pfx + ".block_inject_weight.weight");
                hc.wi_delta = (reinterpret_cast<int64_t>(hc.inj) - reinterpret_cast<int64_t>(hc.down)) / 2;
            }
            return hc;
        };

        // 3. Load top-level & dense layer weights into VRAM
        d_embed_tokens_ = load_vram_tensor("model.language_model.embed_tokens.weight");
        d_lm_head_ = load_vram_tensor("lm_head.weight");
        final_hc_ = load_hc("model.language_model.hyper_connection_mixer", false);

        ple_.key_proj = load_vram_tensor_concat({
            "model.language_model.layers.1.ple.key_proj.weight",
            "model.language_model.layers.1.ple.value_proj.weight",
        });
        ple_.value_proj = static_cast<uint16_t*>(ple_.key_proj) + 10240ULL * 2560;
        ple_.norm_key = load_vram_tensor("model.language_model.layers.1.ple.norm_key.weight");
        ple_.norm_query = load_vram_tensor("model.language_model.layers.1.ple.norm_query.weight");
        ple_.norm_conv = load_vram_tensor("model.language_model.layers.1.ple.norm_conv.weight");
        ple_.conv_weight = load_vram_tensor("model.language_model.layers.1.ple.conv1d.weight");

        for (int l = 0; l < 48; ++l) {
            LayerWeights& lw = layers_[l];
            lw.is_qsa = (l % 4 == 3);
            std::string lpfx = "model.language_model.layers." + std::to_string(l);
            lw.attn_hc = load_hc(lpfx + ".attn_hyper_connection", true);
            lw.mlp_hc = load_hc(lpfx + ".mlp_hyper_connection", true);

            if (!lw.is_qsa) {
                std::string apfx = lpfx + ".linear_attn";
                lw.gdn.in_proj_fused = load_vram_tensor_concat({
                    apfx + ".in_proj_qkv.weight",
                    apfx + ".in_proj_z.weight",
                    apfx + ".in_proj_a.weight",
                    apfx + ".in_proj_b.weight",
                });
                lw.gdn.out_proj = load_vram_tensor(apfx + ".out_proj.weight");
                lw.gdn.conv1d = load_vram_tensor(apfx + ".conv1d.weight");
                lw.gdn.A_log = load_vram_tensor(apfx + ".A_log");
                lw.gdn.dt_bias = load_vram_tensor(apfx + ".dt_bias");
                lw.gdn.norm = load_vram_tensor(apfx + ".norm.weight");
            } else {
                std::string apfx = lpfx + ".self_attn";
                lw.qsa.in_proj_fused = load_vram_tensor_concat({
                    apfx + ".q_proj.weight",
                    apfx + ".k_proj.weight",
                    apfx + ".v_proj.weight",
                    apfx + ".indexer.index_qk_proj.weight",
                });
                lw.qsa.o_proj = load_vram_tensor(apfx + ".o_proj.weight");
                lw.qsa.q_norm = load_vram_tensor(apfx + ".q_norm.weight");
                lw.qsa.k_norm = load_vram_tensor(apfx + ".k_norm.weight");
                lw.qsa.idx_q_ln = load_vram_tensor(apfx + ".indexer.q_layernorm.weight");
                lw.qsa.idx_k_ln = load_vram_tensor(apfx + ".indexer.k_layernorm.weight");
            }

            std::string mpfx = lpfx + ".mlp";
            // Pad 1793 rows to 1856 rows (+63 rows = 63 * 2560 * 2 B) so both GEMV(1793) and WMMA Linear(1856) work
            lw.moe.in_proj_fused = load_vram_tensor_concat(
                {
                    mpfx + ".shared_expert.gate_proj.weight",
                    mpfx + ".shared_expert.up_proj.weight",
                    mpfx + ".gate.weight",
                    mpfx + ".shared_expert_gate.weight",
                },
                63ULL * 2560 * sizeof(uint16_t));
            lw.moe.sh_down = load_vram_tensor(mpfx + ".shared_expert.down_proj.weight");
        }
        for (int s = 0; s < 10; ++s) {
            (void) ::posix_fadvise(shard_fds[s], 0, 0, POSIX_FADV_DONTNEED);
        }

        // 4. Load 48 layers x 512 NVFP4 experts using Strata's expert-profile.bin ranking + free VRAM query
        int32_t vram_rank[48][512];
        for (int l = 0; l < 48; ++l) {
            for (int e = 0; e < 512; ++e) vram_rank[l][e] = -1;
        }
        int layer_vram_cnt[48]{};
        size_t free_vram = 0, total_vram = 0;
        HIP_CHECK(hipMemGetInfo(&free_vram, &total_vram));
        // All state, KV caches, GDN states, recurrent checkpoint arena, and dense BF16 weights are ALREADY allocated before this point!
        // Reserve vram_reserve_mib_ (default 1536 MiB = 1.5 GiB) for display server / compositor + elastic KV growth headroom.
        const size_t reserve_vram = static_cast<size_t>(vram_reserve_mib_) * 1024ULL * 1024ULL;
        int max_vram_experts = (free_vram > reserve_vram)
                                   ? static_cast<int>((free_vram - reserve_vram) / EXP_STRIDE_BYTES)
                                   : (vram_expert_layers_ * 512);
        if (vram_expert_layers_ > 0 && vram_expert_layers_ != 14) {
            max_vram_experts = std::min(max_vram_experts, vram_expert_layers_ * 512);
        }

        int vcache_cap = -1; // Default to temporal _MM_HINT_T0 into CCD 0's 96 MB 3D V-Cache
        if (const char* env_vc = std::getenv("STRATA_NVFP4_VCACHE_EXPERTS")) {
            vcache_cap = std::atoi(env_vc);
        }
        for (int l = 0; l < 48; ++l) {
            for (int e = 0; e < 512; ++e) {
                layers_[l].moe.prof_rank[e] = l * 512 + e;
                layers_[l].moe.in_vcache[e] = (vcache_cap < 0);
            }
        }

        std::vector<std::pair<int32_t, int32_t>> ranked_pairs;
        int64_t prof_slots = 0;
        std::string prof_err;
        if (!expert_profile_path_.empty() &&
            strata::core::read_expert_profile(expert_profile_path_, 48, 512, ranked_pairs, prof_slots, prof_err)) {
            int placed = 0;
            int vcache_placed = 0;
            int rank_idx = 0;
            double cold_acc = 0.0;
            for (const auto& [ly, ex] : ranked_pairs) {
                if (ly < 0 || ly >= 48 || ex < 0 || ex >= 512 || vram_rank[ly][ex] >= 0) continue;
                layers_[ly].moe.prof_rank[ex] = rank_idx++;
                if (placed < max_vram_experts) {
                    vram_rank[ly][ex] = layer_vram_cnt[ly]++;
                    placed++;
                } else {
                    cold_acc += static_cast<double>(pcie_frac_);
                    if (cold_acc >= 1.0) {
                        cold_acc -= 1.0;
                        layers_[ly].moe.in_gpu_cold[ex] = true;
                    } else if (vcache_cap < 0 || vcache_placed < vcache_cap) {
                        layers_[ly].moe.in_vcache[ex] = true;
                        vcache_placed++;
                    }
                }
            }
            total_vram_experts_ = placed;
        } else {
            int placed = 0;
            double cold_acc = 0.0;
            for (int l = 0; l < 48; ++l) {
                for (int e = 0; e < 512; ++e) {
                    if (placed < max_vram_experts) {
                        vram_rank[l][e] = layer_vram_cnt[l]++;
                        placed++;
                    } else {
                        cold_acc += static_cast<double>(pcie_frac_);
                        if (cold_acc >= 1.0) {
                            cold_acc -= 1.0;
                            layers_[l].moe.in_gpu_cold[e] = true;
                        }
                    }
                }
            }
            total_vram_experts_ = placed;
        }

        // Partition host-backed experts into:
        //  (1) gpu_cold_host_base (~11.6 GB, only for in_gpu_cold[e] experts) -> registered with hipHostRegisterMapped
        //  (2) cpu_host_base (~37.0 GB, for CPU-only cold experts + dynamic VRAM backup slots) -> hugepage RAM, no GTT pin!
        size_t total_gpu_cold_experts = 0;
        size_t total_cpu_host_experts = 0;
        int max_layer_pinned = 0;
        size_t layer_gpu_cold_off[48]{};
        size_t layer_cpu_host_off[48]{};
        for (int l = 0; l < 48; ++l) {
            int n_vram = layer_vram_cnt[l];
            int n_dyn = std::min(32, std::max(0, n_vram / 4));
            int n_pinned = n_vram - n_dyn;
            layer_gpu_cold_off[l] = total_gpu_cold_experts;
            layer_cpu_host_off[l] = total_cpu_host_experts;
            for (int e = 0; e < 512; ++e) {
                int32_t rk = vram_rank[l][e];
                bool is_pinned = (rk >= 0 && rk < n_pinned);
                if (is_pinned) continue;
                if (layers_[l].moe.in_gpu_cold[e]) {
                    total_gpu_cold_experts++;
                } else {
                    total_cpu_host_experts++;
                }
            }
            max_layer_pinned = std::max(max_layer_pinned, n_pinned);
        }

        const int n_io_threads = std::max(1, std::min(16, cpu_pool_.num_threads()));
        auto prefault_hugepages = [&](uint8_t* base, size_t total_bytes) {
            if (!base || total_bytes == 0) return;
            const size_t step = 2 * 1024 * 1024; // 2 MB Transparent Huge Page stride
            const size_t n_pages = (total_bytes + step - 1) / step;
            std::vector<std::thread> pf_workers;
            pf_workers.reserve(n_io_threads - 1);
            auto touch_range = [&](int tid) {
                size_t p0 = (static_cast<size_t>(tid) * n_pages) / static_cast<size_t>(n_io_threads);
                size_t p1 = (static_cast<size_t>(tid + 1) * n_pages) / static_cast<size_t>(n_io_threads);
                for (size_t p = p0; p < p1; ++p) {
                    base[p * step] = 0;
                }
            };
            for (int t = 1; t < n_io_threads; ++t) {
                pf_workers.emplace_back(touch_range, t);
            }
            touch_range(0);
            for (auto& w : pf_workers) w.join();
        };

        uint8_t* gpu_cold_host_base = nullptr;
        const size_t total_gpu_cold_bytes = total_gpu_cold_experts * EXP_STRIDE_BYTES;
        if (total_gpu_cold_bytes > 0) {
            void* hptr = nullptr;
            if (::posix_memalign(&hptr, 2 * 1024 * 1024, total_gpu_cold_bytes) != 0 || !hptr) {
                std::fprintf(stderr, "posix_memalign failed for %zu MB GPU-cold host arena\n",
                             total_gpu_cold_bytes / (1024 * 1024));
                std::exit(1);
            }
            (void) ::madvise(hptr, total_gpu_cold_bytes, MADV_HUGEPAGE);
            gpu_cold_host_base = static_cast<uint8_t*>(hptr);
            prefault_hugepages(gpu_cold_host_base, total_gpu_cold_bytes);
        }

        uint8_t* cpu_host_base = nullptr;
        const size_t total_cpu_host_bytes = total_cpu_host_experts * EXP_STRIDE_BYTES;
        if (total_cpu_host_bytes > 0) {
            void* hptr = nullptr;
            if (::posix_memalign(&hptr, 2 * 1024 * 1024, total_cpu_host_bytes) != 0 || !hptr) {
                std::fprintf(stderr, "posix_memalign failed for %zu MB CPU-only host arena\n",
                             total_cpu_host_bytes / (1024 * 1024));
                std::exit(1);
            }
            (void) ::madvise(hptr, total_cpu_host_bytes, MADV_HUGEPAGE);
            cpu_host_base = static_cast<uint8_t*>(hptr);
            host_mallocs_.push_back(cpu_host_base);
            prefault_hugepages(cpu_host_base, total_cpu_host_bytes);
        }

        uint8_t* pin_vram_stage = nullptr;
        if (max_layer_pinned > 0) {
            HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&pin_vram_stage),
                                    static_cast<size_t>(max_layer_pinned) * EXP_STRIDE_BYTES,
                                    hipHostMallocDefault));
        }

        struct ExpSlice9 {
            TensorSlice gw, gs, gs2, uw, us, us2, dw, ds, ds2;
        };
        std::vector<ExpSlice9> exp_slices(512);
        int32_t gpu_cold_slot[48][512];

        // Reserve a single contiguous VmmRange for all 48 layers' dynamic VRAM expert slots so that if the
        // VRAM resident KV cache grows from 6,553 -> vram_cap_ tokens and needs up to ~208 MiB of VRAM,
        // `reclaim_one_dynamic_expert_chunk()` can cleanly unmap 2 MiB chunks from the tail of `exp_vmm_range_`.
        size_t total_dyn_experts = 0;
        for (int l = 0; l < 48; ++l) {
            int n_vram = layer_vram_cnt[l];
            int n_dyn = std::min(32, std::max(0, n_vram / 4));
            exp_layer_dyn_slot_off_[l] = total_dyn_experts;
            total_dyn_experts += static_cast<size_t>(n_dyn);
        }
        const size_t gran = strata::core::vmm_available()
                                ? static_cast<size_t>(strata::core::vmm_granularity())
                                : (2ULL * 1024 * 1024);
        const size_t total_dyn_vmm_bytes = ((total_dyn_experts * EXP_STRIDE_BYTES) + gran - 1) & ~(gran - 1);
        exp_vmm_ok_ = false;
        exp_vmm_mapped_bytes_ = 0;
        if (total_dyn_vmm_bytes > 0 && strata::core::vmm_available() && exp_vmm_range_.reserve(total_dyn_vmm_bytes)) {
            int64_t n_chunks = static_cast<int64_t>(total_dyn_vmm_bytes / gran);
            if (exp_vmm_range_.map_range(0, n_chunks, []() -> strata::core::VmmChunk { return 0; })) {
                exp_vmm_ok_ = true;
                exp_vmm_mapped_bytes_ = total_dyn_vmm_bytes;
            } else {
                exp_vmm_range_.release();
            }
        }

        for (int l = 0; l < 48; ++l) {
            LayerWeights& lw = layers_[l];
            int n_vram = layer_vram_cnt[l];
            int n_dyn = std::min(32, std::max(0, n_vram / 4));
            int n_pinned = n_vram - n_dyn;

            lw.moe.vram_experts = n_vram;
            lw.moe.n_pinned = n_pinned;
            lw.moe.n_dyn = n_dyn;

            uint8_t* vram_base = nullptr;
            uint8_t* dyn_vram_base = nullptr;
            if (exp_vmm_ok_) {
                if (n_pinned > 0) {
                    vram_base = static_cast<uint8_t*>(gpu_alloc(static_cast<size_t>(n_pinned) * EXP_STRIDE_BYTES));
                }
                if (n_dyn > 0) {
                    dyn_vram_base = static_cast<uint8_t*>(exp_vmm_range_.base()) +
                                    exp_layer_dyn_slot_off_[l] * EXP_STRIDE_BYTES;
                }
            } else {
                if (n_vram > 0) {
                    vram_base = static_cast<uint8_t*>(gpu_alloc(static_cast<size_t>(n_vram) * EXP_STRIDE_BYTES));
                }
                if (n_dyn > 0 && vram_base) {
                    dyn_vram_base = vram_base + static_cast<size_t>(n_pinned) * EXP_STRIDE_BYTES;
                }
            }
            lw.moe.vram_base = vram_base;
            lw.moe.dyn_vram_base = dyn_vram_base;

            uint8_t* layer_gpu_cold = gpu_cold_host_base + layer_gpu_cold_off[l] * EXP_STRIDE_BYTES;
            uint8_t* layer_cpu_host = cpu_host_base + layer_cpu_host_off[l] * EXP_STRIDE_BYTES;
            lw.moe.h_ptrs.assign(512 * 6, 0);
            lw.moe.h_ptrs_prefill.assign(512 * 6, 0);
            lw.moe.h_s2.resize(512 * 3);
            std::string epfx_base = "model.language_model.layers." + std::to_string(l) + ".mlp.experts.";

            int sorted_e[512];
            for (int e = 0; e < 512; ++e) {
                std::string epfx = epfx_base + std::to_string(e) + ".";
                exp_slices[e] = ExpSlice9{
                    main_index.at(epfx + "gate_proj.weight"),
                    main_index.at(epfx + "gate_proj.weight_scale"),
                    main_index.at(epfx + "gate_proj.weight_scale_2"),
                    main_index.at(epfx + "up_proj.weight"),
                    main_index.at(epfx + "up_proj.weight_scale"),
                    main_index.at(epfx + "up_proj.weight_scale_2"),
                    main_index.at(epfx + "down_proj.weight"),
                    main_index.at(epfx + "down_proj.weight_scale"),
                    main_index.at(epfx + "down_proj.weight_scale_2"),
                };
                sorted_e[e] = e;
                gpu_cold_slot[l][e] = -1;
            }
            // Sort experts by on-disk (shard_idx, file_offset) order so NVMe reads + hugepage writes are strictly sequential!
            std::sort(sorted_e, sorted_e + 512, [&](int a, int b) {
                const auto& da = exp_slices[a].dw;
                const auto& db = exp_slices[b].dw;
                return (da.shard_idx != db.shard_idx) ? (da.shard_idx < db.shard_idx)
                                                      : (da.file_offset < db.file_offset);
            });

            uint8_t* dst_ptr[512]{};
            int gc_idx = 0;
            int ch_idx = 0;
            for (int e = 0; e < 512; ++e) {
                int32_t rk = vram_rank[l][e];
                bool is_pinned = (rk >= 0 && rk < n_pinned);
                if (is_pinned) {
                    dst_ptr[e] = pin_vram_stage + static_cast<size_t>(rk) * EXP_STRIDE_BYTES;
                } else if (lw.moe.in_gpu_cold[e]) {
                    int s = gc_idx++;
                    gpu_cold_slot[l][e] = s;
                    dst_ptr[e] = layer_gpu_cold + static_cast<size_t>(s) * EXP_STRIDE_BYTES;
                    lw.moe.host_cpu_ptr[e] = dst_ptr[e];
                } else {
                    int s = ch_idx++;
                    dst_ptr[e] = layer_cpu_host + static_cast<size_t>(s) * EXP_STRIDE_BYTES;
                    lw.moe.host_cpu_ptr[e] = dst_ptr[e];
                }
            }

            // Pass A: Read all 512 experts' 4-byte weight_scale_2 values (grouped at start of shard) in sorted order
            for (int idx = 0; idx < 512; ++idx) {
                int e = sorted_e[idx];
                const ExpSlice9& es = exp_slices[e];
                pread_exact(shard_fds[es.gs2.shard_idx], &lw.moe.h_s2[e * 3 + 0], sizeof(float), es.gs2.file_offset);
                pread_exact(shard_fds[es.us2.shard_idx], &lw.moe.h_s2[e * 3 + 1], sizeof(float), es.us2.file_offset);
                pread_exact(shard_fds[es.ds2.shard_idx], &lw.moe.h_s2[e * 3 + 2], sizeof(float), es.ds2.file_offset);
            }

            // Pass B & C: 16-thread sequential NVMe streaming of scales (ds, gs, us) and weights (dw, gw, uw)
            // in strictly ascending file-offset order while placing each matrix's W and S adjacent in dst!
            std::vector<std::thread> io_workers;
            io_workers.reserve(n_io_threads - 1);
            auto load_expert_chunk = [&](int tid) {
                int idx_begin = (tid * 512) / n_io_threads;
                int idx_end = ((tid + 1) * 512) / n_io_threads;
                for (int idx = idx_begin; idx < idx_end; ++idx) {
                    int e = sorted_e[idx];
                    const ExpSlice9& es = exp_slices[e];
                    uint8_t* dst = dst_ptr[e];
                    pread_exact(shard_fds[es.ds.shard_idx], dst + EXP_OFF_DOWN_S, EXP_S_BYTES, es.ds.file_offset);
                    pread_exact(shard_fds[es.gs.shard_idx], dst + EXP_OFF_GATE_S, EXP_S_BYTES, es.gs.file_offset);
                    pread_exact(shard_fds[es.us.shard_idx], dst + EXP_OFF_UP_S, EXP_S_BYTES, es.us.file_offset);
                }
                for (int idx = idx_begin; idx < idx_end; ++idx) {
                    int e = sorted_e[idx];
                    const ExpSlice9& es = exp_slices[e];
                    uint8_t* dst = dst_ptr[e];
                    pread_exact(shard_fds[es.dw.shard_idx], dst + EXP_OFF_DOWN_W, EXP_W_BYTES, es.dw.file_offset);
                    pread_exact(shard_fds[es.gw.shard_idx], dst + EXP_OFF_GATE_W, EXP_W_BYTES, es.gw.file_offset);
                    pread_exact(shard_fds[es.uw.shard_idx], dst + EXP_OFF_UP_W, EXP_W_BYTES, es.uw.file_offset);
                }
            };
            for (int t = 1; t < n_io_threads; ++t) {
                io_workers.emplace_back(load_expert_chunk, t);
            }
            load_expert_chunk(0);
            for (auto& w : io_workers) w.join();

            // Immediately drop clean page-cache pages for all shards so OS page cache never exceeds ~1.4 GB or triggers swap!
            for (int s = 0; s < 10; ++s) {
                (void) ::posix_fadvise(shard_fds[s], 0, 0, POSIX_FADV_DONTNEED);
            }

            // 1 single contiguous pinned DMA transfer for all n_pinned experts of layer l
            if (n_pinned > 0) {
                HIP_CHECK(hipMemcpy(vram_base, pin_vram_stage,
                                    static_cast<size_t>(n_pinned) * EXP_STRIDE_BYTES,
                                    hipMemcpyHostToDevice));
            }
            for (int e = 0; e < 512; ++e) {
                int32_t rk = vram_rank[l][e];
                bool is_pinned = (rk >= 0 && rk < n_pinned);
                bool is_dyn = (rk >= n_pinned && rk < n_vram);
                if (is_pinned) {
                    lw.moe.in_vram[e] = true;
                    lw.moe.dyn_slot[e] = -1;
                } else if (is_dyn) {
                    int s = rk - n_pinned;
                    uint8_t* edev = lw.moe.dyn_vram_base + static_cast<size_t>(s) * EXP_STRIDE_BYTES;
                    HIP_CHECK(hipMemcpy(edev, lw.moe.host_cpu_ptr[e], EXP_STRIDE_BYTES, hipMemcpyHostToDevice));
                    lw.moe.in_vram[e] = true;
                    lw.moe.dyn_slot[e] = static_cast<int16_t>(s);
                    lw.moe.slot_exp[s] = e;
                    lw.moe.slot_last_used[s] = 0;
                    lw.moe.hit_count[e] = 4;
                } else {
                    lw.moe.in_vram[e] = false;
                    lw.moe.dyn_slot[e] = -1;
                }
            }

            if ((l + 1) % 8 == 0 || l == 47) {
                FILE* log_fp = serve_mode_ ? stderr : stdout;
                std::fprintf(log_fp, "[flydsl-nvfp4] Loaded layers 0..%d / 47 (layer %d: %d VRAM [%d pinned + %d dynamic], %d host)\n",
                             l, l, n_vram, n_pinned, n_dyn, 512 - n_vram);
                std::fflush(log_fp);
            }
        }

        if (pin_vram_stage) {
            HIP_CHECK(hipHostFree(pin_vram_stage));
        }

        // Register ONLY the GPU-cold host arena (~11.6 GB instead of ~48.6 GB) with ROCm in 1 call!
        uint8_t* gpu_cold_host_dev = nullptr;
        if (total_gpu_cold_bytes > 0) {
            HIP_CHECK(hipHostRegister(gpu_cold_host_base, total_gpu_cold_bytes, hipHostRegisterMapped));
            host_registered_.push_back(gpu_cold_host_base);
            void* dptr = nullptr;
            HIP_CHECK(hipHostGetDevicePointer(&dptr, gpu_cold_host_base, 0));
            gpu_cold_host_dev = static_cast<uint8_t*>(dptr);
        }

        for (int l = 0; l < 48; ++l) {
            LayerWeights& lw = layers_[l];
            int n_vram = lw.moe.vram_experts;
            int n_pinned = lw.moe.n_pinned;
            uint8_t* vram_base = lw.moe.vram_base;
            uint8_t* layer_gpu_cold_dev = gpu_cold_host_dev + layer_gpu_cold_off[l] * EXP_STRIDE_BYTES;
            for (int e = 0; e < 512; ++e) {
                int32_t rk = vram_rank[l][e];
                bool is_pinned = (rk >= 0 && rk < n_pinned);
                bool is_dyn = (rk >= n_pinned && rk < n_vram);
                uint8_t* edev = nullptr;
                if (is_pinned) {
                    edev = vram_base + static_cast<size_t>(rk) * EXP_STRIDE_BYTES;
                } else if (is_dyn) {
                    edev = lw.moe.dyn_vram_base + static_cast<size_t>(rk - n_pinned) * EXP_STRIDE_BYTES;
                } else if (lw.moe.in_gpu_cold[e]) {
                    uint8_t* hdev = layer_gpu_cold_dev + static_cast<size_t>(gpu_cold_slot[l][e]) * EXP_STRIDE_BYTES;
                    lw.moe.host_dev_ptr[e] = hdev;
                    edev = hdev;
                }
                if (lw.moe.in_vram[e] || lw.moe.in_gpu_cold[e]) {
                    lw.moe.h_ptrs[e * 6 + 0] = reinterpret_cast<int64_t>(edev + EXP_OFF_GATE_W);
                    lw.moe.h_ptrs[e * 6 + 1] = reinterpret_cast<int64_t>(edev + EXP_OFF_GATE_S);
                    lw.moe.h_ptrs[e * 6 + 2] = reinterpret_cast<int64_t>(edev + EXP_OFF_UP_W);
                    lw.moe.h_ptrs[e * 6 + 3] = reinterpret_cast<int64_t>(edev + EXP_OFF_UP_S);
                    lw.moe.h_ptrs[e * 6 + 4] = reinterpret_cast<int64_t>(edev + EXP_OFF_DOWN_W);
                    lw.moe.h_ptrs[e * 6 + 5] = reinterpret_cast<int64_t>(edev + EXP_OFF_DOWN_S);
                }
                if (lw.moe.in_vram[e]) {
                    for (int k = 0; k < 6; ++k) lw.moe.h_ptrs_prefill[e * 6 + k] = lw.moe.h_ptrs[e * 6 + k];
                }
            }
            lw.moe.d_ptrs = gpu_alloc(lw.moe.h_ptrs.size() * sizeof(int64_t));
            lw.moe.d_ptrs_prefill = gpu_alloc(lw.moe.h_ptrs_prefill.size() * sizeof(int64_t));
            lw.moe.d_s2 = gpu_alloc(lw.moe.h_s2.size() * sizeof(float));
            HIP_CHECK(hipMemcpy(lw.moe.d_ptrs, lw.moe.h_ptrs.data(), lw.moe.h_ptrs.size() * sizeof(int64_t),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(lw.moe.d_ptrs_prefill, lw.moe.h_ptrs_prefill.data(),
                                lw.moe.h_ptrs_prefill.size() * sizeof(int64_t), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(lw.moe.d_s2, lw.moe.h_s2.data(), lw.moe.h_s2.size() * sizeof(float),
                                hipMemcpyHostToDevice));
        }

        for (int s = 0; s < 10; ++s) {
            ::close(shard_fds[s]);
        }

        // Pre-warm the top-ranked CPU experts (up to 32 * 2.76 MB = 88.5 MB) into CCD 0's 96 MB 3D V-Cache!
        if (vcache_cap > 0) {
            for (int l = 0; l < 48; ++l) {
                for (int e = 0; e < 512; ++e) {
                    if (layers_[l].moe.in_vcache[e] && layers_[l].moe.host_cpu_ptr[e]) {
                        const uint8_t* p = layers_[l].moe.host_cpu_ptr[e];
                        for (size_t off = 0; off < EXP_STRIDE_BYTES; off += 64) {
                            _mm_prefetch(reinterpret_cast<const char*>(p + off), _MM_HINT_T0);
                        }
                    }
                }
            }
        }
    }

    std::string model_dir_;
    std::string expert_profile_path_;
    FlyDslKernelCache kernels_;
    Nvfp4CpuPool cpu_pool_;
    float pcie_frac_ = 0.60f;
    int vram_expert_layers_ = 14;
    int vram_reserve_mib_ = 1536;
    int total_vram_experts_ = 0;
    int32_t max_seq_ = 1048576;
    int32_t max_blocks_ = 262144;
    int32_t kv_resident_ = 32768;
    int32_t vram_cap_ = 32768;
    int32_t vram_prefix_ = 28672;
    int32_t vram_blocks_ = 7168;
    int32_t kv_mapped_seq_ = 0;
    bool kv_grow_ = true;
    uint32_t step_counter_ = 0;

    strata::kernels::RopeScaling rope_scaling_{};
    bool rope_auto_yarn_ = true;
    int32_t rope_populated_seq_ = 0;
    size_t rope_stride_f32_bytes_ = 0;
    size_t rope_stride_bf16_bytes_ = 0;
    size_t rope_mapped_bytes_ = 0;
    uint8_t* host_rope_mmap_ = nullptr;
    size_t host_rope_mmap_bytes_ = 0;

    strata::core::VmmRange kv_vmm_range_;
    bool kv_vmm_ok_ = false;
    size_t vram_kv_stride_bytes_ = 0;
    size_t vram_bk_stride_bytes_ = 0;
    size_t vram_kv_mapped_bytes_ = 0;
    size_t vram_bk_mapped_bytes_ = 0;

    strata::core::VmmRange exp_vmm_range_;
    bool exp_vmm_ok_ = false;
    size_t exp_vmm_mapped_bytes_ = 0;
    size_t exp_layer_dyn_slot_off_[48]{};

    uint8_t* host_kv_mmap_ = nullptr;
    size_t host_kv_mmap_bytes_ = 0;
    size_t host_kv_stride_bytes_ = 0;
    size_t host_kv_mapped_bytes_ = 0;

    uint8_t* host_bk_mmap_ = nullptr;
    size_t host_bk_mmap_bytes_ = 0;
    size_t host_bk_stride_bytes_ = 0;
    size_t host_bk_mapped_bytes_ = 0;

    int64_t kc_host_delta_[12]{};
    int64_t vc_host_delta_[12]{};
    int64_t bk_host_delta_elems_[12]{};
    void* d_qsa_scores_ = nullptr;
    double t_wait_acc_ms_ = 0.0;
    double t_cpu_acc_ms_ = 0.0;
    int n_cpu_exp_acc_ = 0;
    int n_gpu_pcie_acc_ = 0;
    int n_vram_exp_acc_ = 0;
    hipEvent_t ev_prof_[9]{};

    strata::kernels::PleConsts ple_consts_{};
    PleWeights ple_;
    void* d_embed_tokens_ = nullptr;
    void* d_lm_head_ = nullptr;
    HcWeights final_hc_;
    LayerWeights layers_[48];

    // State & Scratch
    hipEvent_t ev_mixed_ = nullptr;
    hipEvent_t ev_route_ = nullptr;
    void* d_route_bell_ = nullptr;
    void* h_route_bell_ = nullptr;
    void* d_cpu_upload_ = nullptr;
    void* h_cpu_upload_ = nullptr;
    std::vector<CpuExpertJob> cpu_jobs_;
    std::vector<int32_t> h_all_routed_ids_;
    alignas(64) int8_t act_q_[kMaxBatch][2560];
    alignas(64) float act_d_[kMaxBatch][160];
    alignas(64) int32_t act_nb_[kMaxBatch][640];

    void* d_step_meta_ = nullptr;
    void* h_step_meta_ = nullptr;
    void* d_tok_id_ = nullptr;
    void* d_pos_i32_ = nullptr;
    void* d_seq_k_i32_ = nullptr;
    void* d_step_out_ = nullptr;
    void* d_next_tok_ = nullptr;
    void* d_x_stream_a_ = nullptr;
    void* d_x_stream_b_ = nullptr;
    void* d_hc_h_ = nullptr;
    void* d_hc_rstd_ = nullptr;
    void* d_attn_inj_ = nullptr;
    void* d_mlp_inj_ = nullptr;
    void* d_mixed_ = nullptr;
    void* d_attn_out_ = nullptr;
    void* d_mlp_out_ = nullptr;
    void* d_final_hidden_ = nullptr;
    void* d_logits_bf16_ = nullptr;
    void* d_proj_scratch_ = nullptr;

    void* d_ple_rows_u8_ = nullptr;
    void* d_ple_embed_ = nullptr;
    void* d_ple_k_ = nullptr;
    void* d_ple_v_ = nullptr;
    void* d_ple_gv_ = nullptr;
    void* d_ple_gvn_ = nullptr;
    void* d_ple_conv_state_ = nullptr;

    void* d_gdn_qkv_ = nullptr;
    void* d_gdn_z_ = nullptr;
    void* d_gdn_a_ = nullptr;
    void* d_gdn_b_ = nullptr;
    void* d_gdn_z_b_ = nullptr;
    void* d_gdn_a_b_ = nullptr;
    void* d_gdn_b_b_ = nullptr;
    void* d_gdn_out_ = nullptr;
    void* d_gdn_conv_in_[48]{};
    void* d_gdn_conv_out_[48]{};
    void* d_gdn_state_[48]{};
    void* d_recurrent_arena_ = nullptr;
    void* d_recurrent_ckpt_ = nullptr;
    void* h_turn_ckpt_ = nullptr;
    size_t recurrent_arena_bytes_ = 0;
    void* init_gdn_conv_in_[48]{};
    void* init_gdn_conv_out_[48]{};
    void* ckpt_gdn_conv_in_[48]{};
    void* ckpt_gdn_conv_out_[48]{};
    void* turn_ckpt_gdn_conv_in_[48]{};
    void* turn_ckpt_gdn_conv_out_[48]{};
    std::vector<int32_t> turn_ckpt_ids_;
    int32_t turn_ckpt_next_tok_ = -1;
    bool serve_mode_ = false;
    void* d_gdn_verify_arena_ = nullptr;
    void* d_gdn_verify_qkv_[48]{};

    void* d_qsa_qg_ = nullptr;
    void* d_qsa_q_ = nullptr;
    void* d_qsa_k_ = nullptr;
    void* d_qsa_v_ = nullptr;
    void* d_qsa_idx_qk_ = nullptr;
    void* d_qsa_k_b_ = nullptr;
    void* d_qsa_v_b_ = nullptr;
    void* d_qsa_idx_qk_b_ = nullptr;
    void* d_qsa_attn_raw_ = nullptr;
    void* d_qsa_sel_ids_ = nullptr;
    void* d_zero_mask_ = nullptr;
    void* d_attn_pm_ = nullptr;
    void* d_attn_pl_ = nullptr;
    void* d_attn_pa_ = nullptr;
    void* d_k_cache_[12]{};
    void* d_v_cache_[12]{};
    void* d_raw_k_cache_[12]{};
    void* d_block_k_cache_[12]{};

    void* d_rope_cos_f32_ = nullptr;
    void* d_rope_sin_f32_ = nullptr;
    void* d_rope_cos_bf16_ = nullptr;
    void* d_rope_sin_bf16_ = nullptr;

    void* d_sh_gu_ = nullptr;
    void* d_sh_inter_ = nullptr;
    void* d_sh_out_ = nullptr;
    void* d_sh_scale_ = nullptr;
    void* d_router_logits_ = nullptr;
    void* d_moe_ids_ = nullptr;
    void* d_moe_wts_ = nullptr;
    void* d_moe_sorted_ = nullptr;
    void* d_moe_tile_e_ = nullptr;
    void* d_moe_inter_ = nullptr;
    void* d_moe_y_ = nullptr;

    std::vector<void*> vram_allocs_;
    std::vector<void*> host_pinned_;
    std::vector<void*> host_registered_;
    std::vector<void*> host_slab_registered_;
    std::vector<void*> host_mallocs_;
};

}  // namespace

int run(int argc, char** argv) {
    std::string model_dir = "/mnt/models/nvidia--Qwen3.8-Flash-Next-NVFP4";
    if (const char* env_md = std::getenv("STRATA_MODEL_DIR")) {
        if (env_md[0] != '\0') model_dir = env_md;
    }
    std::string cache_dir; // Empty by default: uses embedded gfx1201 .hsaco kernels
    std::string expert_profile_path;
    if (const char* env_ep = std::getenv("STRATA_EXPERT_PROFILE")) {
        if (env_ep[0] != '\0') expert_profile_path = env_ep;
    }
    std::string prompt = "The capital of France is";
    std::string tokens_csv;
    std::string dump_logits_path;
    int max_new_tokens = 16;
    int vram_expert_layers = 14;
    int vram_reserve_mib = 1536; // Default 1.5 GiB reserved for display / compositor + elastic KV headroom on R9700
    if (const char* env_vr = std::getenv("STRATA_NVFP4_RESERVE_VRAM_MB")) {
        vram_reserve_mib = std::max(256, std::atoi(env_vr));
    }
    int max_seq = 1048576; // 1M context window by default (grows elastically from 6,553 tokens via VmmRange + lazy host slabs)
    int kv_resident_tokens = 32768; // 32K tokens resident in VRAM (28,672 prefix + 4,096 sliding window ring = ~240 MB max)
    bool kv_grow = true;
    std::string rope_scaling_str = "auto";
    float rope_scale_arg = 0.0f;
    int yarn_orig_ctx = 262144;
    bool chat_mode = false;
    bool think_mode = false;
    bool serve_mode = false;
    bool spec_ngram = true;
    if (const char* env_sp = std::getenv("STRATA_NVFP4_SPEC")) {
        spec_ngram = (std::atoi(env_sp) != 0);
    }
    SamplingParams cli_sp{};

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--flydsl-nvfp4") continue;
        if (a == "--serve") { serve_mode = true; continue; }
        if (a == "--think") { think_mode = true; chat_mode = true; continue; }
        if (a == "--no-think") { think_mode = false; continue; }
        if (a == "--spec-ngram") { spec_ngram = true; continue; }
        if (a == "--no-spec-ngram") { spec_ngram = false; continue; }
        if (a == "--kv-grow") { kv_grow = true; continue; }
        if (a == "--no-kv-grow") { kv_grow = false; continue; }
        if ((a == "--model" || a == "--native" || a == "--safetensors" || a == "-m") && i + 1 < argc) {
            model_dir = argv[++i];
            // If user passed a shard file path inside the directory, strip to parent directory
            struct stat st {};
            if (::stat(model_dir.c_str(), &st) == 0 && !S_ISDIR(st.st_mode)) {
                auto slash = model_dir.find_last_of('/');
                if (slash != std::string::npos) model_dir = model_dir.substr(0, slash);
            }
        } else if ((a == "--prompt" || a == "-p") && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((a == "--prompt-file" || a == "-f") && i + 1 < argc) {
            std::ifstream pf(argv[++i]);
            if (pf) {
                prompt.assign((std::istreambuf_iterator<char>(pf)), std::istreambuf_iterator<char>());
            }
        } else if (a == "--tokens" && i + 1 < argc) {
            tokens_csv = argv[++i];
        } else if ((a == "--n-predict" || a == "-n" || a == "--max-new" || a == "--tokens-out") && i + 1 < argc) {
            max_new_tokens = std::atoi(argv[++i]);
        } else if (a == "--vram-expert-layers" && i + 1 < argc) {
            vram_expert_layers = std::atoi(argv[++i]);
        } else if ((a == "--vram-reserve-mib" || a == "--vram-reserve") && i + 1 < argc) {
            vram_reserve_mib = std::max(256, std::atoi(argv[++i]));
        } else if (a == "--cache-dir" && i + 1 < argc) {
            cache_dir = argv[++i];
        } else if (a == "--expert-profile" && i + 1 < argc) {
            expert_profile_path = argv[++i];
        } else if ((a == "--max-context" || a == "-c" || a == "--ctx-size") && i + 1 < argc) {
            max_seq = std::atoi(argv[++i]);
        } else if ((a == "--kv-resident" || a == "--vram-kv-tokens") && i + 1 < argc) {
            kv_resident_tokens = std::atoi(argv[++i]);
        } else if (a == "--rope-scaling" && i + 1 < argc) {
            rope_scaling_str = argv[++i];
        } else if (a == "--rope-scale" && i + 1 < argc) {
            rope_scale_arg = static_cast<float>(std::atof(argv[++i]));
        } else if (a == "--yarn-orig-ctx" && i + 1 < argc) {
            yarn_orig_ctx = std::atoi(argv[++i]);
        } else if (a == "--dump-logits" && i + 1 < argc) {
            dump_logits_path = argv[++i];
        } else if (a == "--chat") {
            chat_mode = true;
        } else if ((a == "--temp" || a == "--temperature") && i + 1 < argc) {
            cli_sp.temperature = static_cast<float>(std::atof(argv[++i]));
        } else if (a == "--top-p" && i + 1 < argc) {
            cli_sp.top_p = static_cast<float>(std::atof(argv[++i]));
        } else if (a == "--top-k" && i + 1 < argc) {
            cli_sp.top_k = std::atoi(argv[++i]);
        } else if (a == "--presence-penalty" && i + 1 < argc) {
            cli_sp.presence_penalty = static_cast<float>(std::atof(argv[++i]));
        } else if (a == "--frequency-penalty" && i + 1 < argc) {
            cli_sp.frequency_penalty = static_cast<float>(std::atof(argv[++i]));
        } else if (a == "--repeat-penalty" && i + 1 < argc) {
            cli_sp.repeat_penalty = static_cast<float>(std::atof(argv[++i]));
        } else if (a == "--seed" && i + 1 < argc) {
            cli_sp.seed = static_cast<uint64_t>(std::strtoull(argv[++i], nullptr, 10));
        }
    }

    // Auto-resolve expert-profile.bin relative to executable if not explicitly passed
    if (expert_profile_path.empty()) {
        std::string exe_dir = get_executable_dir();
        if (!exe_dir.empty()) {
            const std::string cand1 = exe_dir + "/data/expert-profile.bin";
            const std::string cand2 = exe_dir + "/../data/expert-profile.bin";
            struct stat st {};
            if (::stat(cand1.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
                expert_profile_path = cand1;
            } else if (::stat(cand2.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
                expert_profile_path = cand2;
            }
        }
    }

    // Startup GPU architecture check: FlyDSL NVFP4 kernels are compiled specifically for RDNA4 (gfx1200 / gfx1201)
    {
        int dev_count = 0;
        hipError_t herr = hipGetDeviceCount(&dev_count);
        if (herr != hipSuccess || dev_count <= 0) {
            std::fprintf(stderr, "[flydsl-nvfp4] Error: No HIP GPU devices detected (%s).\n",
                         hipGetErrorString(herr));
            if (serve_mode) {
                std::printf("ERR No HIP GPU devices detected\n");
                std::fflush(stdout);
            }
            return 1;
        }
        hipDeviceProp_t prop{};
        herr = hipGetDeviceProperties(&prop, 0);
        if (herr != hipSuccess) {
            std::fprintf(stderr, "[flydsl-nvfp4] Error: hipGetDeviceProperties failed (%s).\n",
                         hipGetErrorString(herr));
            if (serve_mode) {
                std::printf("ERR hipGetDeviceProperties failed\n");
                std::fflush(stdout);
            }
            return 1;
        }
        std::string arch(prop.gcnArchName);
        if (arch.rfind("gfx120", 0) != 0) {
            std::fprintf(stderr,
                         "[flydsl-nvfp4] Error: Unsupported GPU architecture '%s' on device '%s'. "
                         "The FlyDSL NVFP4 engine requires an RDNA4 (gfx1200 / gfx1201) GPU.\n",
                         arch.c_str(), prop.name);
            if (serve_mode) {
                std::printf("ERR Unsupported GPU architecture %s (requires gfx1200/gfx1201)\n", arch.c_str());
                std::fflush(stdout);
            }
            return 1;
        }
    }

    QwenTokenizer tokenizer;
    if (!tokenizer.load(model_dir)) {
        std::fprintf(stderr, "[flydsl-nvfp4] Failed to load tokenizer from %s\n", model_dir.c_str());
        if (serve_mode) {
            std::printf("ERR Failed to load tokenizer from %s\n", model_dir.c_str());
            std::fflush(stdout);
        }
        return 1;
    }

    std::vector<int32_t> prompt_ids;
    if (!serve_mode) {
        if (!tokens_csv.empty()) {
            std::stringstream ss(tokens_csv);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) prompt_ids.push_back(std::stoi(item));
            }
        } else {
            std::string full_prompt = prompt;
            if (think_mode) {
                full_prompt = "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n";
            } else if (chat_mode) {
                full_prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            }
            prompt_ids = tokenizer.encode(full_prompt);
        }

        if (prompt_ids.empty()) {
            std::fprintf(stderr, "[flydsl-nvfp4] Prompt produced 0 tokens.\n");
            return 1;
        }

        std::printf("[flydsl-nvfp4] Prompt tokens (%zu):", prompt_ids.size());
        for (int32_t id : prompt_ids) std::printf(" %d", id);
        std::printf("\n");
    }

    strata::kernels::RopeScaling rope_cfg{};
    rope_cfg.freq_base = 1.0e7;
    rope_cfg.orig_ctx = static_cast<double>(std::max(4096, yarn_orig_ctx));
    bool rope_auto_yarn = false;
    if (rope_scaling_str == "none") {
        rope_cfg.type = strata::kernels::RopeScalingType::None;
    } else if (rope_scaling_str == "linear") {
        rope_cfg.type = strata::kernels::RopeScalingType::Linear;
        rope_cfg.factor = (rope_scale_arg > 0.0f)
                              ? static_cast<double>(rope_scale_arg)
                              : std::max(1.0, static_cast<double>(max_seq) / rope_cfg.orig_ctx);
        rope_cfg.ext_factor = 0.0;
    } else if (rope_scaling_str == "yarn") {
        rope_cfg.type = strata::kernels::RopeScalingType::YaRN;
        rope_cfg.factor = (rope_scale_arg > 0.0f)
                              ? static_cast<double>(rope_scale_arg)
                              : std::max(1.0, static_cast<double>(max_seq) / rope_cfg.orig_ctx);
        rope_cfg.ext_factor = 1.0;
    } else {
        // "auto": Dynamic YaRN — exact unscaled RoPE up to orig_ctx (262,144), and YaRN beyond 262,144 up to 1M+
        rope_auto_yarn = true;
        rope_cfg.type = strata::kernels::RopeScalingType::YaRN;
        rope_cfg.factor = (rope_scale_arg > 0.0f)
                              ? static_cast<double>(rope_scale_arg)
                              : std::max(1.0, static_cast<double>(max_seq) / rope_cfg.orig_ctx);
        rope_cfg.ext_factor = 1.0;
    }

    Qwen4ExpFlyDslEngine engine(model_dir, cache_dir, expert_profile_path, vram_expert_layers, max_seq,
                                kv_resident_tokens, kv_grow, rope_cfg, rope_auto_yarn, vram_reserve_mib);
    engine.init(serve_mode);

    std::vector<int32_t> history;
    int32_t next_tok = -1;

    // Per-generation rejection memory so rejected continuations are never re-drafted in the same context
    std::unordered_map<uint64_t, uint8_t> rejected_set;
    auto make_rej_key = [](int32_t p2, int32_t p1, int32_t cand) -> uint64_t {
        return (static_cast<uint64_t>(static_cast<uint32_t>(p2) & 0x1FFFFFu) << 42) |
               (static_cast<uint64_t>(static_cast<uint32_t>(p1) & 0x1FFFFFu) << 21) |
               static_cast<uint64_t>(static_cast<uint32_t>(cand) & 0x1FFFFFu);
    };
    auto is_rejected = [&](int32_t p2, int32_t p1, int32_t cand) -> bool {
        if (rejected_set.find(make_rej_key(p2, p1, cand)) != rejected_set.end()) return true;
        if (p2 == -1 && rejected_set.find(make_rej_key(-1, p1, cand)) != rejected_set.end()) return true;
        return false;
    };

    std::unordered_map<int32_t, std::string> tok_str_cache;
    auto get_tok_str = [&](int32_t id) -> const std::string& {
        auto it = tok_str_cache.find(id);
        if (it != tok_str_cache.end()) return it->second;
        return tok_str_cache.emplace(id, tokenizer.decode_token(id)).first->second;
    };

    auto is_clause_break = [&](int32_t cand_id, int32_t next_id) -> bool {
        if (cand_id < 0 || cand_id >= 248040) return true;
        const std::string& s = get_tok_str(cand_id);
        if (s.empty()) return true;
        char c0 = s[0];
        if (c0 == '\n' || c0 == '\r' || c0 == ',' || c0 == ';' || c0 == ')' || c0 == ']' || c0 == '}') return true;
        if (c0 == '.') {
            if (s.size() == 1 && next_id >= 0 && next_id < 248040) {
                const std::string& ns = get_tok_str(next_id);
                if (!ns.empty() && std::isdigit(static_cast<unsigned char>(ns[0]))) return false;
            }
            return true;
        }
        return false;
    };

    auto is_subword_cont = [&](int32_t prev_id, int32_t cand_id, int32_t next_id) -> bool {
        if (prev_id < 0 || cand_id < 0 || cand_id >= 248040) return false;
        const std::string& ps = get_tok_str(prev_id);
        const std::string& cs = get_tok_str(cand_id);
        if (ps.empty() || cs.empty()) return false;
        char pb = ps.back();
        if (!std::isalnum(static_cast<unsigned char>(pb)) && pb != '-') return false;
        char c0 = cs[0];
        if (std::isalnum(static_cast<unsigned char>(c0))) return true;
        if ((c0 == '-' || c0 == '/') && cs.size() >= 2 && std::isalnum(static_cast<unsigned char>(cs[1]))) return true;
        if ((cs == "-" || cs == ".") && next_id >= 0 && next_id < 248040) {
            const std::string& ns = get_tok_str(next_id);
            if (!ns.empty() && std::isalnum(static_cast<unsigned char>(ns[0]))) return true;
        }
        return false;
    };

    auto is_digit_tok = [&](int32_t id) -> bool {
        if (id < 0 || id >= 248040) return false;
        const std::string& s = get_tok_str(id);
        size_t pos = (!s.empty() && s[0] == ' ') ? 1 : 0;
        return pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]));
    };

    auto is_entity_or_num = [&](int32_t id) -> bool {
        if (id < 0 || id >= 248040) return false;
        const std::string& s = get_tok_str(id);
        if (s.empty()) return false;
        size_t pos = (s[0] == ' ') ? 1 : 0;
        if (s.size() - pos < 2) return false;
        unsigned char c0 = static_cast<unsigned char>(s[pos]);
        if (std::isdigit(c0)) return true;
        if (!std::isupper(c0)) return false;
        std::string w = s.substr(pos);
        static const char* const kCapStop[] = {
            "The", "A", "An", "In", "On", "At", "To", "For", "Of", "With", "By", "From", "As",
            "Is", "Are", "Was", "Were", "Be", "This", "That", "These", "Those", "It", "Its",
            "They", "We", "You", "And", "Or", "But", "If", "When", "While", "Where", "Why",
            "How", "What", "Which", "Who", "Here", "There", "Each", "Both", "All", "Some",
            "No", "Not", "Only", "Also", "However", "Therefore", "Thus", "Summarize",
            "Explain", "Describe", "List", "Compare", "Key", "Summary", "Overview", "Note",
            "Contains", "Equipped", "Uses", "Based", "Running", "Routes", "Total", "Step"
        };
        for (const char* sw : kCapStop) {
            if (w == sw) return false;
        }
        return true;
    };

    // Returns true if cand_id is the first digit of a bracketed/bolded value (e.g. "(0-7)" vs "(8-15)", "**96" vs "**32")
    auto is_new_number_head = [&](int32_t /*prev2_id*/, int32_t prev_id, int32_t cand_id) -> bool {
        if (!is_digit_tok(cand_id) || prev_id < 0) return false;
        const std::string& ps = get_tok_str(prev_id);
        if (ps.empty()) return false;
        char pb = ps.back();
        return (pb == '(' || pb == '*' || pb == '[');
    };

    auto is_content_token = [&](int32_t id) -> bool {
        if (is_entity_or_num(id)) return true;
        if (id < 0 || id >= 248040) return false;
        const std::string& s = get_tok_str(id);
        size_t pos = (!s.empty() && s[0] == ' ') ? 1 : 0;
        if (s.size() - pos < 4) return false;
        if (!std::isalpha(static_cast<unsigned char>(s[pos]))) return false;
        std::string w = s.substr(pos);
        static const char* const kLowStop[] = {
            "that", "this", "with", "from", "have", "will", "what", "when", "where", "which",
            "while", "their", "there", "these", "those", "about", "into", "over", "after",
            "before", "under", "between", "through", "during", "without", "within", "because",
            "should", "would", "could", "other", "such", "more", "most", "some", "many",
            "much", "very", "also", "only", "even", "just", "than", "then", "them", "they",
            "were", "been", "being", "does", "make", "made", "used", "using", "uses", "each", "both"
        };
        for (const char* sw : kLowStop) {
            if (w == sw) return false;
        }
        return true;
    };

    auto draft_ngram = [&](int32_t cur_tok, size_t prompt_len, std::vector<int32_t>& draft_out) {
        draft_out.clear();
        if (!spec_ngram || prompt_len < 8 || history.size() < 4) return;
        const size_t n = history.size();
        const int32_t w0 = history[n - 3];
        const int32_t w1 = history[n - 2];
        const int32_t w2 = history[n - 1];
        const int32_t w3 = cur_tok;

        // Multi-level consensus suffix matcher: L = 4, 3, 2, 1
        for (int L = 4; L >= 1; --L) {
            int total1 = 0, total2 = 0, total3 = 0, total4 = 0;
            int agree1 = 0, agree2 = 0, agree3 = 0, agree4 = 0;
            int32_t c1 = -1, c2 = -1, c3 = -1, c4 = -1;
            const size_t min_i = static_cast<size_t>(L - 1);
            for (size_t i = n - 2;; --i) {
                bool match = (history[i] == w3);
                if (match && L >= 2) match = (history[i - 1] == w2);
                if (match && L >= 3) match = (history[i - 2] == w1);
                if (match && L >= 4) match = (history[i - 3] == w0);
                if (match && (i + 1) < n) {
                    int32_t cand1 = history[i + 1];
                    if (cand1 >= 0 && cand1 < 248040 && !is_rejected(w2, w3, cand1) &&
                        !(L == 1 && is_rejected(-1, w3, cand1))) {
                        total1++;
                        int32_t cand2 = ((i + 2) < n && history[i + 2] < 248040) ? history[i + 2] : -1;
                        int32_t cand3 = (cand2 >= 0 && (i + 3) < n && history[i + 3] < 248040) ? history[i + 3] : -1;
                        int32_t cand4 = (cand3 >= 0 && (i + 4) < n && history[i + 4] < 248040) ? history[i + 4] : -1;
                        if (c1 < 0) {
                            c1 = cand1;
                            c2 = cand2;
                            c3 = cand3;
                            c4 = cand4;
                        }
                        if (cand1 == c1) {
                            agree1++;
                            if (cand2 >= 0) {
                                if (c2 < 0) c2 = cand2;
                                total2++;
                                if (cand2 == c2) {
                                    agree2++;
                                    if (cand3 >= 0) {
                                        if (c3 < 0) c3 = cand3;
                                        total3++;
                                        if (cand3 == c3) {
                                            agree3++;
                                            if (cand4 >= 0) {
                                                if (c4 < 0) c4 = cand4;
                                                total4++;
                                                if (cand4 == c4) agree4++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (i == min_i) break;
            }
            if (total1 == 0 || c1 < 0) continue;

            // A bare space token (" ") is only safe to draft if the token following it in history is also unanimous
            auto space_ok = [&](int32_t ck, int32_t ck_next, int ag_next, int tot_next, int32_t prev_t) -> bool {
                if (get_tok_str(ck) != " ") return true;
                return (ck_next >= 0 && tot_next > 0 && ag_next == tot_next && !is_new_number_head(prev_t, ck, ck_next));
            };

            if (L == 4) {
                if (is_new_number_head(w2, w3, c1) && agree1 < 2) continue;
                draft_out.push_back(c1);
                if (c2 >= 0 && !is_rejected(w3, c1, c2) && !is_new_number_head(w3, c1, c2)) {
                    draft_out.push_back(c2);
                    if (c3 >= 0 && !is_rejected(c1, c2, c3) && !is_new_number_head(c1, c2, c3)) {
                        draft_out.push_back(c3);
                    }
                }
                if (draft_out.size() >= 2) return;
                draft_out.clear();
                continue;
            }
            if (L == 3) {
                if (agree1 == total1 && !is_clause_break(c1, c2) && !is_new_number_head(w2, w3, c1) &&
                    space_ok(c1, c2, agree2, total2, w3)) {
                    draft_out.push_back(c1);
                    if (c2 >= 0 && total2 > 0 && agree2 == total2 && !is_rejected(w3, c1, c2) &&
                        !is_clause_break(c2, c3) && !is_new_number_head(w3, c1, c2) &&
                        space_ok(c2, c3, agree3, total3, c1)) {
                        draft_out.push_back(c2);
                        if (c3 >= 0 && total3 > 0 && agree3 == total3 && !is_rejected(c1, c2, c3) &&
                            !is_clause_break(c3, c4) && !is_new_number_head(c1, c2, c3) &&
                            space_ok(c3, c4, agree4, total4, c2)) {
                            draft_out.push_back(c3);
                        }
                    }
                    if (draft_out.size() >= 2) return;
                    draft_out.clear();
                }
            }
            if (L == 2) {
                bool ok2 = (agree1 == total1) && !is_clause_break(c1, c2) && !is_new_number_head(w2, w3, c1) &&
                           space_ok(c1, c2, agree2, total2, w3) &&
                           (is_subword_cont(w3, c1, c2) || is_subword_cont(w2, w3, c1) || is_content_token(w3) ||
                            (is_content_token(w2) && is_content_token(c1)));
                if (ok2) {
                    draft_out.push_back(c1);
                    if (c2 >= 0 && total2 > 0 && agree2 == total2 && !is_rejected(w3, c1, c2) &&
                        !is_clause_break(c2, c3) && !is_new_number_head(w3, c1, c2) &&
                        space_ok(c2, c3, agree3, total3, c1) &&
                        (is_subword_cont(c1, c2, c3) || is_content_token(c2) || total1 >= 2)) {
                        draft_out.push_back(c2);
                        if (c3 >= 0 && total3 > 0 && agree3 == total3 && !is_rejected(c1, c2, c3) &&
                            !is_clause_break(c3, c4) && !is_new_number_head(c1, c2, c3) &&
                            space_ok(c3, c4, agree3, total4, c2) &&
                            (is_subword_cont(c2, c3, c4) || is_entity_or_num(c3) || total1 >= 2)) {
                            draft_out.push_back(c3);
                        }
                    }
                    if (draft_out.size() >= 2) return;
                    draft_out.clear();
                }
            }
            if (L == 1) {
                const std::string& w3_s = get_tok_str(w3);
                size_t w3_pos = (!w3_s.empty() && w3_s[0] == ' ') ? 1 : 0;
                bool w3_substantial = (w3_s.size() - w3_pos >= 2) || is_entity_or_num(w3);
                bool sub1 = w3_substantial && is_subword_cont(w3, c1, c2);
                bool ent1 = (total1 <= 6) && is_entity_or_num(w3) &&
                            (is_entity_or_num(c1) || (get_tok_str(c1) == " " && space_ok(c1, c2, agree2, total2, w3)));
                if (agree1 == total1 && !is_clause_break(c1, c2) && !is_new_number_head(w2, w3, c1) &&
                    space_ok(c1, c2, agree2, total2, w3) && (sub1 || ent1)) {
                    draft_out.push_back(c1);
                    if (c2 >= 0 && total2 > 0 && agree2 == total2 && !is_rejected(w3, c1, c2) &&
                        !is_clause_break(c2, c3) && !is_new_number_head(w3, c1, c2) &&
                        space_ok(c2, c3, agree3, total3, c1) &&
                        (is_subword_cont(c1, c2, c3) || is_entity_or_num(c2) ||
                         (get_tok_str(c1) == " " && is_digit_tok(c2)))) {
                        draft_out.push_back(c2);
                        if (c3 >= 0 && total3 > 0 && agree3 == total3 && !is_rejected(c1, c2, c3) &&
                            !is_clause_break(c3, c4) && !is_new_number_head(c1, c2, c3) &&
                            space_ok(c3, c4, agree4, total4, c2) &&
                            (is_subword_cont(c2, c3, c4) || (is_digit_tok(c2) && is_digit_tok(c3)))) {
                            draft_out.push_back(c3);
                        }
                    }
                    if (draft_out.size() >= 2) return;
                    draft_out.clear();
                }
            }
        }
    };

    if (serve_mode) {
        std::printf("INFO kv_resident=%d expert_slots=%d vram_reserve_mib=%d batch_slots=0 engine=flydsl-nvfp4\n",
                    engine.vram_cap(), engine.total_vram_experts(), vram_reserve_mib);
        std::printf("READY %d stop\n", engine.max_seq());
        std::fflush(stdout);

        std::mutex q_mtx;
        std::condition_variable q_cv;
        std::deque<std::string> cmd_q;
        std::atomic<bool> stop_req{false};
        std::atomic<bool> stdin_eof{false};

        std::thread reader_thr([&]() {
            std::string line;
            while (std::getline(std::cin, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
                    line.pop_back();
                }
                if (line == "STOP") {
                    stop_req.store(true, std::memory_order_relaxed);
                    continue;
                }
                if (line.empty()) continue;
                {
                    std::lock_guard<std::mutex> lk(q_mtx);
                    cmd_q.push_back(std::move(line));
                }
                q_cv.notify_one();
            }
            stdin_eof.store(true, std::memory_order_relaxed);
            q_cv.notify_all();
        });
        reader_thr.detach();

        while (true) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(q_mtx);
                q_cv.wait(lk, [&]() { return !cmd_q.empty() || stdin_eof.load(std::memory_order_relaxed); });
                if (cmd_q.empty()) break;
                line = std::move(cmd_q.front());
                cmd_q.pop_front();
            }
            if (line == "QUIT") break;
            if (line.rfind("VRAM", 0) == 0) {
                size_t free_v = 0, total_v = 0;
                (void) hipMemGetInfo(&free_v, &total_v);
                std::printf("VRAM reserve_mib=%d expert_slots=%d vram_free_mib=%zu\n",
                            vram_reserve_mib, engine.total_vram_experts(),
                            static_cast<size_t>(free_v / (1024ULL * 1024ULL)));
                std::fflush(stdout);
                continue;
            }
            if (line.rfind("GEN ", 0) == 0) {
                stop_req.store(false, std::memory_order_relaxed);
                rejected_set.clear();

                // Format: GEN <max_new> [key=val ...] <id0,id1,...>
                std::istringstream iss(line);
                std::string cmd;
                int req_max_new = max_new_tokens;
                iss >> cmd >> req_max_new;
                std::vector<std::string> fields;
                std::string tok_field;
                while (iss >> tok_field) {
                    fields.push_back(std::move(tok_field));
                }
                if (fields.empty()) {
                    std::printf("DONE 0 0 0 0 stop 0 0 0\n");
                    std::fflush(stdout);
                    continue;
                }
                SamplingParams req_sp = cli_sp;
                for (size_t fi = 0; fi + 1 < fields.size(); ++fi) {
                    const std::string& kv = fields[fi];
                    auto eq = kv.find('=');
                    if (eq == std::string::npos) continue;
                    std::string k = kv.substr(0, eq);
                    std::string v = kv.substr(eq + 1);
                    if (k == "temp" || k == "temperature") {
                        req_sp.temperature = static_cast<float>(std::atof(v.c_str()));
                    } else if (k == "top_p") {
                        req_sp.top_p = static_cast<float>(std::atof(v.c_str()));
                    } else if (k == "top_k") {
                        req_sp.top_k = std::atoi(v.c_str());
                    } else if (k == "presence_penalty") {
                        req_sp.presence_penalty = static_cast<float>(std::atof(v.c_str()));
                    } else if (k == "frequency_penalty") {
                        req_sp.frequency_penalty = static_cast<float>(std::atof(v.c_str()));
                    } else if (k == "repeat_penalty") {
                        req_sp.repeat_penalty = static_cast<float>(std::atof(v.c_str()));
                    } else if (k == "seed") {
                        req_sp.seed = static_cast<uint64_t>(std::strtoull(v.c_str(), nullptr, 10));
                    }
                }
                std::mt19937_64 req_rng(req_sp.seed);

                const std::string& ids_csv = fields.back();
                std::vector<int32_t> req_ids;
                req_ids.reserve(1024);
                {
                    size_t pos = 0;
                    while (pos < ids_csv.size()) {
                        size_t comma = ids_csv.find(',', pos);
                        if (comma == std::string::npos) comma = ids_csv.size();
                        if (comma > pos) {
                            req_ids.push_back(static_cast<int32_t>(std::strtol(ids_csv.c_str() + pos, nullptr, 10)));
                        }
                        pos = comma + 1;
                    }
                }
                const size_t N = req_ids.size();
                if (N == 0) {
                    std::printf("DONE 0 0 0 0 stop 0 0 0\n");
                    std::fflush(stdout);
                    continue;
                }

                // Check prefix match against current `history` or previous turn's `turn_ckpt_ids()`
                size_t reused = 0;
                if (!history.empty() && N > history.size()) {
                    bool match = true;
                    for (size_t k = 0; k < history.size(); ++k) {
                        if (req_ids[k] != history[k]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        reused = history.size();
                    }
                }
                if (reused == 0) {
                    const auto& ckpt_ids = engine.turn_ckpt_ids();
                    if (!ckpt_ids.empty() && N > ckpt_ids.size()) {
                        bool match = true;
                        for (size_t k = 0; k < ckpt_ids.size(); ++k) {
                            if (req_ids[k] != ckpt_ids[k]) {
                                match = false;
                                break;
                            }
                        }
                        if (match) {
                            engine.restore_turn_ckpt(history, next_tok);
                            reused = history.size();
                        }
                    }
                }
                if (reused == 0) {
                    engine.reset_session();
                    history.clear();
                    next_tok = -1;
                } else {
                    std::printf("RESUME %zu\n", reused);
                    std::fflush(stdout);
                }

                auto t_pf0 = std::chrono::steady_clock::now();
                bool cancelled = false;
                for (size_t i = reused; i < N;) {
                    if (stop_req.load(std::memory_order_relaxed)) {
                        cancelled = true;
                        break;
                    }
                    size_t chunk = std::min<size_t>(256, N - i);
                    std::vector<int32_t> prev0s(chunk);
                    std::vector<int32_t> prev1s(chunk);
                    for (size_t t = 0; t < chunk; ++t) {
                        prev0s[t] = (history.size() >= 2) ? history[history.size() - 2] : strata::kernels::TOKEN_NULL;
                        prev1s[t] = (history.size() >= 1) ? history[history.size() - 1] : strata::kernels::TOKEN_NULL;
                        history.push_back(req_ids[i + t]);
                    }
                    bool is_last = (i + chunk == N);
                    next_tok = engine.step_prefill_batch(&req_ids[i], prev0s.data(), prev1s.data(),
                                                         static_cast<int32_t>(chunk), static_cast<int32_t>(i),
                                                         is_last, nullptr, nullptr,
                                                         &req_sp, &history, &req_rng);
                    i += chunk;
                    double pf_ms_so_far = std::chrono::duration<double, std::milli>(
                                              std::chrono::steady_clock::now() - t_pf0)
                                              .count();
                    double pf_tok_s = (pf_ms_so_far > 0.0) ? ((i - reused) * 1000.0 / pf_ms_so_far) : 0.0;
                    std::printf("PP %zu %zu %.0f %.1f\n", i, N, pf_ms_so_far, pf_tok_s);
                    std::fflush(stdout);
                }
                double prefill_ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - t_pf0)
                                        .count();

                if (!cancelled) {
                    engine.save_turn_ckpt(history, next_tok);
                }

                auto t_dec0 = std::chrono::steady_clock::now();
                int generated = 0;
                int spec_accepted_total = 0;
                int spec_drafted_total = 0;
                const char* finish_reason = cancelled ? "cancel" : "length";

                for (int step = 0; !cancelled && step < req_max_new;) {
                    if (stop_req.load(std::memory_order_relaxed)) {
                        finish_reason = "cancel";
                        break;
                    }
                    if (next_tok < 0) {
                        finish_reason = "stop";
                        break;
                    }
                    if (next_tok == 248044 || next_tok == 248046) {
                        std::printf("T %d\n", next_tok);
                        std::fflush(stdout);
                        generated++;
                        finish_reason = "stop";
                        break;
                    }
                    std::printf("T %d\n", next_tok);
                    std::fflush(stdout);
                    generated++;
                    step++;
                    if (step >= req_max_new) {
                        finish_reason = "length";
                        break;
                    }
                    if (stop_req.load(std::memory_order_relaxed)) {
                        finish_reason = "cancel";
                        break;
                    }

                    std::vector<int32_t> draft;
                    if (req_sp.is_greedy()) {
                        draft_ngram(next_tok, N, draft);
                    }
                    int max_rem = req_max_new - step - 1;
                    if (static_cast<int>(draft.size()) > max_rem) {
                        draft.resize(std::max(0, max_rem));
                    }

                    if (!draft.empty()) {
                        const int32_t K = static_cast<int32_t>(draft.size());
                        const int32_t M = 1 + K;
                        spec_drafted_total += K;
                        engine.save_recurrent_ckpt();

                        int32_t pos0 = static_cast<int32_t>(history.size());
                        int32_t b_toks[4], b_p0[4], b_p1[4], preds[4]{};
                        for (int32_t j = 0; j < M; ++j) {
                            int32_t tk = (j == 0) ? next_tok : draft[j - 1];
                            b_toks[j] = tk;
                            b_p0[j] = (history.size() >= 2) ? history[history.size() - 2] : strata::kernels::TOKEN_NULL;
                            b_p1[j] = (history.size() >= 1) ? history[history.size() - 1] : strata::kernels::TOKEN_NULL;
                            history.push_back(tk);
                        }
                        (void) engine.step_prefill_batch(b_toks, b_p0, b_p1, M, pos0, true, nullptr, preds);

                        int32_t accepted = 0;
                        while (accepted < K && preds[accepted] == draft[accepted]) {
                            accepted++;
                        }
                        spec_accepted_total += accepted;
                        const int32_t n_adv = 1 + accepted;
                        engine.finish_verify_step(M, n_adv);

                        if (accepted < K) {
                            int32_t prev_t = b_toks[accepted];
                            int32_t prev2_t = b_p1[accepted];
                            int32_t bad_cand = draft[accepted];
                            rejected_set[make_rej_key(prev2_t, prev_t, bad_cand)] = 1;
                            rejected_set[make_rej_key(-1, prev_t, bad_cand)] = 1;
                            history.resize(static_cast<size_t>(pos0 + n_adv));
                        }
                        bool hit_eos = false;
                        for (int32_t j = 0; j < accepted; ++j) {
                            if (draft[j] == 248044 || draft[j] == 248046) {
                                finish_reason = "stop";
                                hit_eos = true;
                                break;
                            }
                            std::printf("T %d\n", draft[j]);
                            std::fflush(stdout);
                            generated++;
                            step++;
                        }
                        if (hit_eos) break;
                        next_tok = preds[accepted];
                    } else {
                        int32_t prev0 = (history.size() >= 2) ? history[history.size() - 2] : strata::kernels::TOKEN_NULL;
                        int32_t prev1 = (history.size() >= 1) ? history[history.size() - 1] : strata::kernels::TOKEN_NULL;
                        int32_t pos = static_cast<int32_t>(history.size());
                        history.push_back(next_tok);
                        next_tok = engine.step_token(history.back(), prev0, prev1, pos, true, nullptr,
                                                     &req_sp, &history, &req_rng);
                    }
                }
                double decode_ms = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - t_dec0)
                                       .count();
                std::printf("DONE %d %zu %.0f %.0f %s %d %d %zu\n",
                            generated, N, prefill_ms, decode_ms, finish_reason,
                            spec_accepted_total, spec_drafted_total, reused);
                std::fflush(stdout);
                std::fprintf(stderr,
                             "[flydsl-nvfp4-serve] prompt=%zu (reused=%zu, new=%zu) in %.1f ms (%.1f tok/s) | "
                             "gen=%d in %.1f ms (%.2f tok/s, spec=%d/%d) | finish=%s\n",
                             N, reused, N - reused, prefill_ms,
                             (N > reused && prefill_ms > 0.0) ? ((N - reused) * 1000.0 / prefill_ms) : 0.0,
                             generated, decode_ms,
                             (generated > 0 && decode_ms > 0.0) ? (generated * 1000.0 / decode_ms) : 0.0,
                             spec_accepted_total, spec_drafted_total, finish_reason);
                std::fflush(stderr);
            }
        }
        return 0;
    }

    std::mt19937_64 cli_rng(cli_sp.seed);
    std::vector<float> dumped_logits;
    auto t_prefill_start = std::chrono::steady_clock::now();

    const size_t total_prompt = prompt_ids.size();
    for (size_t i = 0; i < total_prompt;) {
        size_t chunk = std::min<size_t>(256, total_prompt - i);
        std::vector<int32_t> prev0s(chunk);
        std::vector<int32_t> prev1s(chunk);
        for (size_t t = 0; t < chunk; ++t) {
            prev0s[t] = (history.size() >= 2) ? history[history.size() - 2] : strata::kernels::TOKEN_NULL;
            prev1s[t] = (history.size() >= 1) ? history[history.size() - 1] : strata::kernels::TOKEN_NULL;
            history.push_back(prompt_ids[i + t]);
        }
        bool is_last = (i + chunk == total_prompt);
        next_tok = engine.step_prefill_batch(&prompt_ids[i], prev0s.data(), prev1s.data(),
                                             static_cast<int32_t>(chunk), static_cast<int32_t>(i), is_last,
                                             (is_last && !dump_logits_path.empty()) ? &dumped_logits : nullptr,
                                             nullptr, &cli_sp, &history, &cli_rng);
        i += chunk;
    }
    double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prefill_start).count();

    if (!dump_logits_path.empty() && !dumped_logits.empty()) {
        std::ofstream lf(dump_logits_path, std::ios::binary);
        lf.write(reinterpret_cast<const char*>(dumped_logits.data()),
                 static_cast<std::streamsize>(dumped_logits.size() * sizeof(float)));
        std::printf("[flydsl-nvfp4] Dumped %zu float32 logits to %s\n", dumped_logits.size(), dump_logits_path.c_str());
    }

    std::string display_prompt;
    if (tokens_csv.empty()) {
        display_prompt = prompt;
    } else {
        size_t start_idx = prompt_ids.size() > 64 ? prompt_ids.size() - 64 : 0;
        if (start_idx > 0) display_prompt = "... ";
        for (size_t k = start_idx; k < prompt_ids.size(); ++k) {
            display_prompt += tokenizer.decode_token(prompt_ids[k]);
        }
    }
    if (display_prompt.size() > 240) {
        display_prompt = "... " + display_prompt.substr(display_prompt.size() - 240);
    }
    std::printf("\n--- Output ---\n%s", display_prompt.c_str());
    auto t_decode_start = std::chrono::steady_clock::now();
    int generated = 0;
    int spec_accepted_total = 0;
    int spec_drafted_total = 0;

    for (int step = 0; step < max_new_tokens;) {
        if (next_tok < 0) break;
        std::string piece = tokenizer.decode_token(next_tok);
        std::printf("%s", piece.c_str());
        std::fflush(stdout);
        generated++;
        step++;
        if (next_tok == 248044 || next_tok == 248046 || step >= max_new_tokens) break;

        std::vector<int32_t> draft;
        if (cli_sp.is_greedy()) {
            draft_ngram(next_tok, total_prompt, draft);
        }
        int max_rem = max_new_tokens - step - 1;
        if (static_cast<int>(draft.size()) > max_rem) {
            draft.resize(std::max(0, max_rem));
        }

        if (!draft.empty()) {
            const int32_t K = static_cast<int32_t>(draft.size());
            const int32_t M = 1 + K;
            spec_drafted_total += K;
            engine.save_recurrent_ckpt();

            int32_t pos0 = static_cast<int32_t>(history.size());
            int32_t b_toks[4], b_p0[4], b_p1[4], preds[4]{};
            for (int32_t j = 0; j < M; ++j) {
                int32_t tk = (j == 0) ? next_tok : draft[j - 1];
                b_toks[j] = tk;
                b_p0[j] = (history.size() >= 2) ? history[history.size() - 2] : strata::kernels::TOKEN_NULL;
                b_p1[j] = (history.size() >= 1) ? history[history.size() - 1] : strata::kernels::TOKEN_NULL;
                history.push_back(tk);
            }
            (void) engine.step_prefill_batch(b_toks, b_p0, b_p1, M, pos0, true, nullptr, preds);

            int32_t accepted = 0;
            while (accepted < K && preds[accepted] == draft[accepted]) {
                accepted++;
            }
            spec_accepted_total += accepted;
            const int32_t n_adv = 1 + accepted;
            engine.finish_verify_step(M, n_adv);

            if (accepted < K) {
                // Record rejected transition so it is never re-drafted in this context
                int32_t prev_t = b_toks[accepted];
                int32_t prev2_t = b_p1[accepted];
                int32_t bad_cand = draft[accepted];
                rejected_set[make_rej_key(prev2_t, prev_t, bad_cand)] = 1;
                rejected_set[make_rej_key(-1, prev_t, bad_cand)] = 1;
                history.resize(static_cast<size_t>(pos0 + n_adv));
            }
            for (int32_t j = 0; j < accepted; ++j) {
                std::string dp = tokenizer.decode_token(draft[j]);
                std::printf("%s", dp.c_str());
                std::fflush(stdout);
                generated++;
                step++;
            }
            next_tok = preds[accepted];
        } else {
            int32_t prev0 = (history.size() >= 2) ? history[history.size() - 2] : strata::kernels::TOKEN_NULL;
            int32_t prev1 = (history.size() >= 1) ? history[history.size() - 1] : strata::kernels::TOKEN_NULL;
            int32_t pos = static_cast<int32_t>(history.size());
            history.push_back(next_tok);
            next_tok = engine.step_token(history.back(), prev0, prev1, pos, true, nullptr,
                                         &cli_sp, &history, &cli_rng);
        }
    }
    double decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_decode_start).count();
    int decode_steps = std::max(1, generated - 1);
    std::printf("\n--------------\n");
    if (spec_drafted_total > 0) {
        std::printf("[flydsl-nvfp4] spec-ngram: accepted %d / %d drafted tokens (%.1f%%)\n",
                    spec_accepted_total, spec_drafted_total, 100.0 * spec_accepted_total / spec_drafted_total);
    }
    std::printf("[flydsl-nvfp4] prefill: %zu tokens in %.3f s (%.2f tok/s) | decode: %d steps in %.3f s (%.2f tok/s, %.2f ms/tok)\n",
                prompt_ids.size(), prefill_s, prompt_ids.size() / prefill_s,
                decode_steps, decode_s, decode_steps / std::max(1e-6, decode_s),
                1000.0 * decode_s / decode_steps);
    return 0;
}

}  // namespace strata::flydsl_nvfp4
