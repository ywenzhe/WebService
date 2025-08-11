#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <iomanip>

// 依赖库头文件
#include "snappy.h"
#include "cryptopp/aes.h"
#include "cryptopp/modes.h"
#include "cryptopp/filters.h"
#include "cryptopp/osrng.h" 

// 自定义 Zipfian 生成器
#include "zipf.hpp" // 请确保您将最新的zipf.hpp保存为这个名字

// --- 模拟原始代码的常量定义 ---
namespace config {
    // 哈希表配置
    constexpr uint32_t kKeyLen = 12;
    constexpr uint32_t kValueLen = 4;
    constexpr uint64_t kNumKVPairs = 1ul << 27; // 128M

    // 数组配置
    constexpr uint64_t kNumArrayEntries = 2ul << 20; // 2M
    constexpr uint32_t kArrayEntrySize = 8192;      // 8KB

    // 运行时配置
    constexpr uint32_t kNumMutatorThreads = 40;
    constexpr double kZipfParamS = 0.85;
    constexpr uint32_t kNumKeysPerRequest = 32;
    constexpr uint64_t kNumReqs = kNumKVPairs / static_cast<uint64_t>(kNumKeysPerRequest);
    constexpr uint32_t kLog10NumKeysPerRequest = 2; // log10(32) approx 1.5, use 2 for suffix
    constexpr uint32_t kReqLen = kKeyLen - kLog10NumKeysPerRequest;

    // 基准测试的运行时间评估参数
    constexpr uint32_t kBenchmarkDurationSeconds = 100; // 基准测试的总运行秒数 (可配置)
    constexpr uint32_t kStatsPrintIntervalSeconds = 1;  // 每隔多少秒输出一次实时统计
    constexpr uint32_t kDefaultNumSetupThreads = 8; // 用于数据准备阶段的默认线程数
}

// --- 数据结构定义 ---
struct Key {
    char data[config::kKeyLen];

    bool operator==(const Key& other) const {
        return std::memcmp(data, other.data, config::kKeyLen) == 0;
    }
};

// Key 的哈希函数，以便用于 std::unordered_map
template<>
struct std::hash<Key> {
    std::size_t operator()(const Key& k) const {
        size_t result = 0xcbf29ce484222325ULL; // FNV offset basis
        for (std::size_t i = 0; i < config::kKeyLen; ++i) {
            result ^= static_cast<size_t>(k.data[i]);
            result *= 1099511628211ULL; // FNV prime
        }
        return result;
    }
};

union Value {
    uint32_t num;
    char data[config::kValueLen];
};

struct ArrayEntry {
    uint8_t data[config::kArrayEntrySize];
};

// 全局变量和数据结构
// !! 警告: 这会占用大量内存 (约 26GB) !!
std::unordered_map<Key, Value> hashtable;
std::vector<ArrayEntry> data_array;
std::vector<Key> base_request_keys;
// 现在 zipf_request_indices 存储的就是 0-indexed 的结果
std::vector<uint64_t> zipf_request_indices;

// 性能统计
std::atomic<uint64_t> total_requests_done{0};
std::atomic<bool> should_stop{false};

// -- 辅助函数 --
// 将数字转为固定长度的字符串后缀
void append_uint32_to_char_array(uint32_t n, uint32_t suffix_len, char* array) {
    std::string s = std::to_string(n);
    std::string padded_s = std::string(suffix_len - std::min((size_t)suffix_len, s.length()), '0') + s;
    std::memcpy(array, padded_s.c_str(), suffix_len);
}

// 用于并行生成 base_request_keys 的工作函数
void generate_base_keys_worker(uint64_t start_idx, uint64_t end_idx, uint32_t worker_seed) {
    std::mt19937 generator(worker_seed);
    std::uniform_int_distribution<int> distribution('a', 'z');
    for (uint64_t i = start_idx; i < end_idx; ++i) {
        for (uint32_t j = 0; j < config::kReqLen; ++j) {
            base_request_keys[static_cast<std::size_t>(i)].data[j] = static_cast<char>(distribution(generator));
        }
    }
}

// 用于并行填充 data_array 的工作函数
void populate_array_worker(uint64_t start_idx, uint64_t end_idx, uint32_t worker_seed) {
    std::mt19937 generator(worker_seed);
    std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
    for (uint64_t i = start_idx; i < end_idx; ++i) {
        // 为每个 8KB 块生成一个随机起始字节
        uint8_t seed_byte = byte_dist(generator);
        // 使用简单的循环填充整个 8KB 块，确保每个块的内容是变化的，且速度快
        for (uint32_t j = 0; j < config::kArrayEntrySize; ++j) {
            data_array[static_cast<std::size_t>(i)].data[j] = (seed_byte + j) % 256; // 确保不溢出 255，并产生变化
        }
    }
}


// 准备阶段：填充数据
void prepare_data(uint32_t num_setup_threads) {
    std::cout << "Preparing data... This will take a while and use ~26GB of RAM." << std::endl;

    std::vector<std::thread> setup_threads;
    std::random_device rd; // For initial seeds

    // 1. 准备哈希表的基础键 (并行)
    std::cout << "  [1/4] Generating " << config::kNumReqs << " base keys (in parallel)..." << std::endl;
    base_request_keys.resize(static_cast<std::size_t>(config::kNumReqs)); 
    uint64_t chunk_size_keys = config::kNumReqs / num_setup_threads;
    for (uint32_t t = 0; t < num_setup_threads; ++t) {
        uint64_t start_idx = t * chunk_size_keys;
        uint64_t end_idx = (t == num_setup_threads - 1) ? config::kNumReqs : (t + 1) * chunk_size_keys;
        setup_threads.emplace_back(generate_base_keys_worker, start_idx, end_idx, rd() + t);
    }
    for (auto& t : setup_threads) { t.join(); }
    setup_threads.clear();


    // 2. 填充哈希表 (单线程 - std::unordered_map 的并发写入需要额外复杂的同步或专门的并发容器)
    std::cout << "  [2/4] Populating hash table with " << config::kNumKVPairs << " key-value pairs (single-threaded for stability)..." << std::endl;
    hashtable.reserve(config::kNumKVPairs);
    // 使用一个通用的随机数生成器用于Zipfian，为了可复现性。哈希表填充也用它。
    std::mt19937 general_generator(12345); 

    for (uint64_t i = 0; i < config::kNumReqs; ++i) {
        if (i > 0 && i % (config::kNumReqs / 10) == 0) {
            std::cout << "    ... Hash table population " << (i * 100.0 / config::kNumReqs) << "% done" << std::endl; 
        }
        Key current_key_template = base_request_keys[static_cast<std::size_t>(i)]; // 从已填充的 base_request_keys 获取
        for (uint32_t j = 0; j < config::kNumKeysPerRequest; ++j) {
            Key final_key = current_key_template;
            append_uint32_to_char_array(j, config::kLog10NumKeysPerRequest, final_key.data + config::kReqLen);
            
            Value v;
            v.num = (j == 0) ? (uint32_t)i : 0; 
            hashtable.emplace(final_key, v);
        }
    }

    // 3. 准备大数组 (并行)
    std::cout << "  [3/4] Allocating and populating 16GB array (in parallel)..." << std::endl;
    data_array.resize(static_cast<std::size_t>(config::kNumArrayEntries)); 
    uint64_t chunk_size_array = config::kNumArrayEntries / num_setup_threads;
    for (uint32_t t = 0; t < num_setup_threads; ++t) {
        uint64_t start_idx = t * chunk_size_array;
        uint64_t end_idx = (t == num_setup_threads - 1) ? config::kNumArrayEntries : (t + 1) * chunk_size_array;
        setup_threads.emplace_back(populate_array_worker, start_idx, end_idx, rd() + t + num_setup_threads); // 区分种子
    }
    for (auto& t : setup_threads) { t.join(); }
    setup_threads.clear();

    // 4. 准备 Zipfian 分布的请求序列 (Zipfian预计算可能较慢，但比填充Array快得多，如果它也慢，需要并行化该特定部分的Loop)
    std::cout << "  [4/4] Generating Zipfian request distribution (this step will take time for 1.34亿 entries!)..." << std::endl;
    // 使用新的 zipf_table_distribution
    benchmark::zipf_table_distribution<uint64_t, double> zipf(config::kNumReqs, config::kZipfParamS);
    zipf_request_indices.resize(static_cast<std::size_t>(config::kNumReqs));
    for (uint64_t i = 0; i < config::kNumReqs; ++i) {
        // Now directly return 0-indexed results, no need for -1
        zipf_request_indices[static_cast<std::size_t>(i)] = zipf(general_generator); 
        if (i > 0 && i % (config::kNumReqs / 10) == 0) {
             std::cout << "    ... Zipfian generation " << (i * 100.0 / config::kNumReqs) << "% done" << std::endl;
        }
    }
    
    std::cout << "Data preparation complete." << std::endl;
}

// 修正：benchmark_worker 现在直接接收 CryptoPP 对象的 unique_ptr，并接管它们的生命周期
void benchmark_worker(std::unique_ptr<CryptoPP::AES::Encryption> aes_encryption_ptr,
                      std::unique_ptr<CryptoPP::CBC_Mode_ExternalCipher::Encryption> cbc_encryption_ptr) {
    
    // 现在可以直接通过解引用 unique_ptr 来使用加密器对象，它们会随着线程的结束而销毁
    // 确保 cbc_encryption_ptr 确实是指向一个有效的对象
    if (!cbc_encryption_ptr) {
        std::cerr << "Error: cbc_encryption_ptr is null in worker thread!" << std::endl;
        return;
    }

    std::string compressed, ciphertext;
    std::atomic<uint64_t>& global_req_idx = total_requests_done;

    while (!should_stop.load(std::memory_order_relaxed)) {
        uint64_t req_k_idx = global_req_idx.fetch_add(1, std::memory_order_relaxed);
        size_t base_key_idx = static_cast<size_t>(zipf_request_indices[req_k_idx % config::kNumReqs]); 
        
        Key current_key_template = base_request_keys[base_key_idx];
        uint32_t array_index_base = 0;
        
        for (uint32_t i = 0; i < config::kNumKeysPerRequest; ++i) {
            Key final_key = current_key_template;
            append_uint32_to_char_array(i, config::kLog10NumKeysPerRequest, final_key.data + config::kReqLen);
            
            auto it = hashtable.find(final_key);
            if (it != hashtable.end()) {
                array_index_base += it->second.num;
            }
        }
        
        uint64_t final_array_index = array_index_base % config::kNumArrayEntries;
        const ArrayEntry& entry = data_array[static_cast<std::size_t>(final_array_index)]; 
        
        // 加密 (Crypto++) - 使用线程局部持有的 cbc_encryption_ptr
        ciphertext.clear();
        CryptoPP::StreamTransformationFilter stfEncryptor(
            *cbc_encryption_ptr, new CryptoPP::StringSink(ciphertext)
        );
        stfEncryptor.Put(entry.data, config::kArrayEntrySize);
        stfEncryptor.MessageEnd();
        
        // 压缩 (Snappy)
        compressed.clear();
        snappy::Compress(ciphertext.c_str(), ciphertext.size(), &compressed);

        (void)compressed;
    }
}


int main() {
    std::cout << "WARNING: This benchmark requires ~26 GB of available RAM or swap space." << std::endl;
    std::cout << "Press Enter to continue or Ctrl+C to abort..." << std::endl;
    std::cin.get();
    
    uint32_t num_setup_threads = config::kDefaultNumSetupThreads;

    uint32_t hardware_threads = std::thread::hardware_concurrency();
    if (hardware_threads > 0 && num_setup_threads > hardware_threads) {
        std::cerr << "Warning: Default kNumSetupThreads (" << num_setup_threads 
                  << ") exceeds hardware concurrency (" << hardware_threads << "). Adjusting to " 
                  << hardware_threads << "." << std::endl;
        num_setup_threads = hardware_threads; 
    }

    prepare_data(num_setup_threads);
    
    std::cout << "\nStarting benchmark with " << config::kNumMutatorThreads << " threads. Target duration: " 
              << config::kBenchmarkDurationSeconds << " seconds." << std::endl;

    // key 和 iv 可以在 main 函数的局部范围声明，因为它们只是用于初始化加密器
    byte key[CryptoPP::AES::DEFAULT_KEYLENGTH];
    byte iv[CryptoPP::AES::BLOCKSIZE];
    memset(key, 0x01, sizeof(key)); 
    memset(iv, 0x01, sizeof(iv));

    std::vector<std::thread> threads;
    // 修正：不再需要 encryptors vector 来管理生命周期，因为所有权会转移给线程
    // std::vector<std::unique_ptr<CryptoPP::CBC_Mode_ExternalCipher::Encryption>> encryptors;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (std::size_t i = 0; i < config::kNumMutatorThreads; ++i) {
        // 为每个线程创建并初始化其专属的加密器对象
        auto aesEncryption = std::make_unique<CryptoPP::AES::Encryption>(key, sizeof(key));
        auto cbcEncryption = std::make_unique<CryptoPP::CBC_Mode_ExternalCipher::Encryption>(*aesEncryption, iv);

        // 修正：将两个 unique_ptr 的所有权通过 std::move 转移到新线程的参数中
        threads.emplace_back(benchmark_worker, std::move(aesEncryption), std::move(cbcEncryption));
    }
    
    // 监控并报告性能
    uint64_t last_req_count = 0;
    auto last_time = start_time;
    auto next_stats_print_time = start_time + std::chrono::seconds(config::kStatsPrintIntervalSeconds);
    
    while (std::chrono::high_resolution_clock::now() - start_time <
           std::chrono::seconds(config::kBenchmarkDurationSeconds)) {
        
        std::this_thread::sleep_until(next_stats_print_time);

        auto current_time = std::chrono::high_resolution_clock::now();
        uint64_t current_req_count = total_requests_done.load(std::memory_order_relaxed);
        
        double duration_sec = std::chrono::duration<double>(current_time - last_time).count();
        uint64_t reqs_this_interval = current_req_count - last_req_count;
        
        double throughput = duration_sec > 0 ? static_cast<double>(reqs_this_interval) / duration_sec : 0.0;
        
        std::cout << "Time: " << std::setw(3) << std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count() << "s | "
                  << "Throughput: " << std::fixed << std::setprecision(2) << std::setw(10) << throughput << " req/s | "
                  << "Total Requests: " << current_req_count << std::endl;
                  
        last_req_count = current_req_count;
        last_time = current_time;
        next_stats_print_time += std::chrono::seconds(config::kStatsPrintIntervalSeconds);
    }
    
    should_stop.store(true, std::memory_order_relaxed);
    std::cout << "\nStopping threads..." << std::endl;

    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_actual_duration = std::chrono::duration<double>(end_time - start_time).count();
    uint64_t final_req_count = total_requests_done.load();
    
    std::cout << "\nBenchmark finished." << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Target runtime: " << config::kBenchmarkDurationSeconds << " seconds" << std::endl;
    std::cout << "Actual runtime: " << std::fixed << std::setprecision(2) << total_actual_duration << " seconds" << std::endl;
    std::cout << "Total requests processed: " << final_req_count << std::endl;
    double avg_throughput = total_actual_duration > 0 ? static_cast<double>(final_req_count) / total_actual_duration : 0.0;
    std::cout << "Average throughput: " << avg_throughput << " req/s" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    
    return 0;
}

