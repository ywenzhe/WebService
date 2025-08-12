#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <atomic>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <memory>

// 第三方库依赖
#include "snappy.h"
#include "zipf.hpp" // 使用你提供的 zipf.hpp

// crypto++
#include "cryptopp/aes.h"
#include "cryptopp/modes.h"
#include "cryptopp/filters.h"
#include "cryptopp/osrng.h" // 用于生成随机密钥/IV

// Intel TBB Concurrent Hash Map - 高性能线程安全哈希表
#include <tbb/concurrent_unordered_map.h>

// 宏定义以保持代码一致性，尽管在本地版本中作用不大
#define ACCESS_ONCE(x) (*(volatile decltype(x) *)&(x))
#define BUG_ON(condition) if (condition) { std::cerr << "BUG_ON failed at " << __FILE__ << ":" << __LINE__ << std::endl; abort(); }

namespace far_memory {

    class LocalWebServiceTest {
        private:
        // --- Benchmark Parameters ---
        // Hashtable.
        static constexpr uint32_t kKeyLen = 12;
        static constexpr uint32_t kValueLen = 4;
        static constexpr uint64_t kNumKVPairs = 1ULL << 27; // 128M key-value pairs

        // Array.
        static constexpr uint32_t kNumArrayEntries = 2ULL << 20; // 2 M entries.
        static constexpr uint32_t kArrayEntrySize = 8192;   // 8 KB

        // Runtime.
        static constexpr uint32_t kNumMutatorThreads = 40;
        static constexpr double kZipfParamS = 0.85;
        static constexpr uint32_t kNumKeysPerRequest = 32;
        static constexpr uint64_t kNumReqs = (1ULL << 27) / kNumKeysPerRequest;
        static constexpr uint64_t kTotalTargetRequests = 2e7; // 2千万次请求
        static constexpr uint32_t kLog10NumKeysPerRequest = 2; // log10(32) approx 1.5, use 2 for suffix
        static constexpr uint32_t kReqLen = kKeyLen - kLog10NumKeysPerRequest;
        static constexpr uint64_t kReqSeqLen = kNumReqs;

        // --- 数据结构定义 ---
        struct Req {
            char data[kReqLen];
        };

        struct Key {
            char data[kKeyLen];
        };

        union Value {
            uint32_t num;
            char data[kValueLen];
        };

        struct ArrayEntry {
            uint8_t data[kArrayEntrySize];
        };

        // 为 TBB Hash Map 定义 Hash 和 Equal 函数
        struct KeyHash {
            std::size_t operator()(const Key& k) const {
                // 使用简单的字符串哈希函数
                std::size_t h = 0;
                for (size_t i = 0; i < kKeyLen; ++i) {
                    h = h * 31 + k.data[i];
                }
                return h;
            }
        };

        struct KeyEqual {
            bool operator()(const Key& a, const Key& b) const {
                return std::memcmp(a.data, b.data, kKeyLen) == 0;
            }
        };

        // 使用 TBB 的并发哈希表和标准 vector
        using HashTable = tbb::concurrent_unordered_map<Key, Value, KeyHash, KeyEqual>;
        using DataArray = std::vector<ArrayEntry>;

        // --- 成员变量 ---
        std::unique_ptr<HashTable> hopscotch_map_;
        std::unique_ptr<DataArray> data_array_;

        std::vector<Req> all_gen_reqs_;
        std::vector<uint32_t> all_zipf_req_indices_;

        std::atomic<uint64_t> total_completed_reqs_{ 0 };
        std::atomic<bool> preparing_done_{ false };


        unsigned char crypto_key_[CryptoPP::AES::DEFAULT_KEYLENGTH];
        unsigned char crypto_iv_[CryptoPP::AES::BLOCKSIZE];

        // --- 辅助函数 ---
        inline void append_uint32_to_char_array(uint32_t n, uint32_t suffix_len, char* array) {
            std::string s = std::to_string(n);
            std::string padded_s = std::string(suffix_len - std::min((uint32_t)s.length(), suffix_len), '0') + s;
            memcpy(array, padded_s.c_str(), suffix_len);
        }

        inline void random_string(char* data, uint32_t len, std::mt19937& generator) {
            std::uniform_int_distribution<int> distribution('a', 'z');
            for (uint32_t i = 0; i < len; i++) {
                data[i] = static_cast<char>(distribution(generator));
            }
        }

        void consume_array_entry(const ArrayEntry& entry, CryptoPP::CBC_Mode_ExternalCipher::Encryption& cbcEncryption) {
            std::string ciphertext, compressed;

            // 加密
            CryptoPP::StreamTransformationFilter stfEncryptor(
                cbcEncryption, new CryptoPP::StringSink(ciphertext));
            stfEncryptor.Put(entry.data, kArrayEntrySize);
            stfEncryptor.MessageEnd();

            // 压缩
            snappy::Compress(ciphertext.c_str(), ciphertext.size(), &compressed);

            auto compressed_len = compressed.length();
            // 确保编译器不会优化掉这个操作
            ACCESS_ONCE(compressed_len);
        }

        public:
        LocalWebServiceTest() {
            // 初始化加密密钥和 IV
            CryptoPP::OS_GenerateRandomBlock(false, crypto_key_, sizeof(crypto_key_));
            CryptoPP::OS_GenerateRandomBlock(false, crypto_iv_, sizeof(crypto_iv_));
        }

        void prepare() {
            std::cout << "Preparing dataset (" << kNumKVPairs << " K/V pairs, "
                << kNumArrayEntries << " array entries)..." << std::endl;

            // 1. 初始化数据结构
            std::cout << "Allocating memory for HashMap and Array (approx. 26GB)..." << std::endl;
            hopscotch_map_ = std::make_unique<HashTable>();
            // 预分配 buckets，避免测试中 rehash
            hopscotch_map_->rehash(kNumKVPairs * 1.2);

            data_array_ = std::make_unique<DataArray>(kNumArrayEntries);
            // 初始化数组，可以填充随机数据
            std::mt19937_64 array_gen(12345); // 固定种子以保证可复现
            for (auto& entry : *data_array_) {
                for (size_t i = 0; i < kArrayEntrySize; i += sizeof(uint64_t)) {
                    *(uint64_t*)(entry.data + i) = array_gen();
                }
            }
            std::cout << "Memory allocated." << std::endl;

            all_gen_reqs_.resize(kNumReqs);

            // 2. 多线程填充哈希表
            std::vector<std::thread> threads;
            std::cout << "Populating HashMap with " << kNumMutatorThreads << " threads..." << std::endl;
            for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
                threads.emplace_back([this, tid]() {
                    std::mt19937 generator(tid);
                    uint64_t reqs_per_thread = kNumReqs / kNumMutatorThreads;
                    uint64_t start_idx = tid * reqs_per_thread;
                    uint64_t end_idx = (tid == kNumMutatorThreads - 1) ? kNumReqs : start_idx + reqs_per_thread;

                    for (uint64_t i = start_idx; i < end_idx; ++i) {
                        Req req;
                        random_string(req.data, kReqLen, generator);

                        for (uint32_t j = 0; j < kNumKeysPerRequest; j++) {
                            Key key;
                            memcpy(key.data, req.data, kReqLen);
                            append_uint32_to_char_array(j, kLog10NumKeysPerRequest, key.data + kReqLen);

                            Value value;
                            value.num = (j == 0) ? i : 0;
                            hopscotch_map_->insert({ key, value });
                        }
                        all_gen_reqs_[i] = req;
                    }
                    });
            }
            for (auto& t : threads) t.join();

            BUG_ON(hopscotch_map_->size() != kNumKVPairs);
            std::cout << "HashMap populated successfully. Total size: " << hopscotch_map_->size() << std::endl;

            // 3. 生成 Zipfian 分布的请求序列
            std::cout << "Generating Zipfian request sequence..." << std::endl;
            all_zipf_req_indices_.resize(kReqSeqLen);
            std::mt19937 zipf_gen(54321);
            zipf_table_distribution<> zipf(kNumReqs, kZipfParamS);
            for (uint64_t i = 0; i < kReqSeqLen; i++) {
                all_zipf_req_indices_[i] = zipf(zipf_gen);
            }

            preparing_done_ = true;
            std::cout << "Preparation complete." << std::endl;
        }

        void bench() {
            if (!preparing_done_) {
                std::cerr << "Error: `prepare()` must be called before `bench()`." << std::endl;
                return;
            }

            std::cout << "\nStarting benchmark with " << kNumMutatorThreads << " threads..." << std::endl;
            std::cout << "Total target requests: " << kTotalTargetRequests << std::endl;

            std::vector<std::thread> threads;
            auto start_time = std::chrono::high_resolution_clock::now();

            std::atomic<uint64_t> request_idx_counter{ 0 };

            for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
                threads.emplace_back([this, tid, &request_idx_counter]() {

                    // 每个线程创建自己的加密器实例，避免线程竞争
                    CryptoPP::AES::Encryption aesEncryption(crypto_key_, CryptoPP::AES::DEFAULT_KEYLENGTH);
                    CryptoPP::CBC_Mode_ExternalCipher::Encryption cbcEncryption(aesEncryption, crypto_iv_);

                    while (true) {
                        uint64_t current_req_count = total_completed_reqs_.load(std::memory_order_relaxed);
                        if (current_req_count >= kTotalTargetRequests) {
                            break;
                        }

                        // 原子地获取下一个请求的索引
                        uint64_t req_seq_idx = request_idx_counter.fetch_add(1, std::memory_order_relaxed) % kReqSeqLen;
                        uint32_t req_idx = all_zipf_req_indices_[req_seq_idx];
                        const Req& req = all_gen_reqs_[req_idx];

                        uint32_t array_index = 0;
                        Key key;
                        memcpy(key.data, req.data, kReqLen);

                        // 1. 查找 32 个 key
                        for (uint32_t i = 0; i < kNumKeysPerRequest; i++) {
                            append_uint32_to_char_array(i, kLog10NumKeysPerRequest, key.data + kReqLen);

                            auto it = hopscotch_map_->find(key);
                            if (it != hopscotch_map_->end()) {
                                array_index += it->second.num;
                            }
                        }

                        // 2. 访问数组、加密和压缩
                        array_index %= kNumArrayEntries;
                        const auto& array_entry = (*data_array_)[array_index];
                        consume_array_entry(array_entry, cbcEncryption);

                        total_completed_reqs_.fetch_add(1, std::memory_order_relaxed);
                    }
                    });
            }

            // 进度报告线程
            std::thread reporter([this, &start_time]() {
                while (true) {
                    uint64_t completed = total_completed_reqs_.load();
                    if (completed >= kTotalTargetRequests) break;

                    auto now = std::chrono::high_resolution_clock::now();
                    double elapsed_sec = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
                    double progress_percent = (double)completed * 100.0 / kTotalTargetRequests;
                    double tps = (elapsed_sec > 0) ? completed / elapsed_sec : 0;

                    std::cout << "[Progress] " << std::fixed << std::setprecision(2) << progress_percent << "% | "
                        << "Reqs: " << completed << "/" << kTotalTargetRequests
                        << " | TPS: " << (long)tps << "\r" << std::flush;

                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
                });

            for (auto& t : threads) t.join();
            if (reporter.joinable()) reporter.join();

            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> runtime_seconds = end_time - start_time;

            std::cout << "\n---------------------------------------------------" << std::endl;
            std::cout << "Benchmark Finished!" << std::endl;
            uint64_t final_reqs = total_completed_reqs_.load();
            std::cout << "Total Requests Processed: " << final_reqs << std::endl;
            std::cout << "Total Runtime: " << std::fixed << std::setprecision(4) << runtime_seconds.count() << " seconds" << std::endl;
            std::cout << "Throughput: " << std::fixed << std::setprecision(2) << final_reqs / runtime_seconds.count() << " reqs/sec" << std::endl;
            std::cout << "---------------------------------------------------\n" << std::endl;
        }

        void run() {
            prepare();
            bench();
        }
    };

} // namespace far_memory

int main() {
    far_memory::LocalWebServiceTest test;
    test.run();
    return 0;
}

