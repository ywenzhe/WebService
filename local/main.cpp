#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <shared_mutex>
#include <thread>
#include <vector>
#include <sched.h>

// crypto++
#include <cryptopp/aes.h>
#include <cryptopp/filters.h>
#include <cryptopp/modes.h>

// snappy
#include <snappy.h>

#include "zipf.hpp"
#include "hopscotch_hashtable.hpp"

#define ACCESS_ONCE(x) (*(volatile decltype(x) *)&(x))

using namespace far_memory;

class LocalBenchmark {
private:
    // Constants
    constexpr static uint32_t kKeyLen = 12;
    constexpr static uint32_t kValueLen = 4;
    constexpr static uint32_t kNumKVPairs = 1 << 27; // 128M pairs
    
    // Array
    constexpr static uint32_t kNumArrayEntries = 2 << 20; // 2M entries
    constexpr static uint32_t kArrayEntrySize = 8192;     // 8KB
    
    // Runtime
    constexpr static uint32_t kNumMutatorThreads = 40;
    constexpr static uint32_t kNumCPUs = 20;
    constexpr static double kZipfParamS = 0.85;
    constexpr static uint32_t kNumKeysPerRequest = 32;
    constexpr static uint32_t kNumReqs = kNumKVPairs / kNumKeysPerRequest;
    constexpr static uint32_t kLog10NumKeysPerRequest = 2; // log10(32) = 1.5, rounded up
    constexpr static uint32_t kReqLen = kKeyLen - kLog10NumKeysPerRequest;
    constexpr static uint32_t kReqSeqLen = kNumReqs;
    
    // Output
    constexpr static uint32_t kPrintPerIters = 8192;
    constexpr static uint32_t kMaxPrintIntervalUs = 1000 * 1000; // 1 second
    constexpr static uint32_t kPrintTimes = 30; // 减少运行时间

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

    struct alignas(64) Cnt {
        uint64_t c;
    };

    // 使用更复杂的Hopscotch哈希表

    // 成员变量
    std::unique_ptr<std::mt19937> generators[kNumCPUs];
    std::vector<Req> all_gen_reqs;
    std::vector<std::vector<uint32_t>> all_zipf_req_indices;
    std::vector<ArrayEntry> array_entries;
    std::unique_ptr<HopscotchHashTable> hashtable;

    Cnt req_cnts[kNumMutatorThreads] = {}; // 初始化为0
    std::atomic_flag flag;
    uint64_t print_times = 0;
    uint64_t prev_sum_reqs = 0;
    uint64_t prev_us = 0;
    uint64_t running_us = 0;
    std::vector<double> mops_records;
    std::atomic<bool> should_stop{ false };

    unsigned char key[CryptoPP::AES::DEFAULT_KEYLENGTH];
    unsigned char iv[CryptoPP::AES::BLOCKSIZE];
    std::unique_ptr<CryptoPP::CBC_Mode_ExternalCipher::Encryption> cbcEncryption;
    std::unique_ptr<CryptoPP::AES::Encryption> aesEncryption;

    // 辅助函数
    inline void append_uint32_to_char_array(uint32_t n, uint32_t suffix_len, char* array) {
        uint32_t len = 0;
        while (n) {
            auto digit = n % 10;
            array[len++] = digit + '0';
            n = n / 10;
        }
        while (len < suffix_len) {
            array[len++] = '0';
        }
        std::reverse(array, array + suffix_len);
    }

    inline void random_string(char* data, uint32_t len) {
        uint32_t core_num = sched_getcpu();  // 获取当前CPU核心号
        auto& generator = *generators[core_num];
        std::uniform_int_distribution<int> distribution('a', 'z' + 1);
        for (uint32_t i = 0; i < len; i++) {
            data[i] = char(distribution(generator));
        }
    }

    inline void random_req(char* data, uint32_t tid) {
        auto tid_len = 2;  // log10(40) ≈ 1.6，取2
        random_string(data, kReqLen - tid_len);
        append_uint32_to_char_array(tid, tid_len, data + kReqLen - tid_len);
    }

    uint64_t microtime() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

    void prepare() {
        // 初始化随机数生成器
        std::random_device rd;
        for (uint32_t i = 0; i < kNumCPUs; i++) {
            generators[i].reset(new std::mt19937(rd()));
        }

        // 初始化加密组件
        memset(key, 0x00, CryptoPP::AES::DEFAULT_KEYLENGTH);
        memset(iv, 0x00, CryptoPP::AES::BLOCKSIZE);
        aesEncryption.reset(
            new CryptoPP::AES::Encryption(key, CryptoPP::AES::DEFAULT_KEYLENGTH));
        cbcEncryption.reset(
            new CryptoPP::CBC_Mode_ExternalCipher::Encryption(*aesEncryption, iv));

        // 初始化请求数据
        all_gen_reqs.resize(kNumReqs);
        hashtable = std::make_unique<HopscotchHashTable>(28);  // 2^28 = 256M entries

        // 生成初始数据
        std::vector<std::thread> threads;
        for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
            threads.emplace_back([&, tid]() {
                auto num_reqs_per_thread = kNumReqs / kNumMutatorThreads;
                auto req_offset = tid * num_reqs_per_thread;
                auto* thread_gen_reqs = &all_gen_reqs[req_offset];
                for (uint32_t i = 0; i < num_reqs_per_thread; i++) {
                    Req req;
                    random_req(req.data, tid);
                    Key key;
                    memcpy(key.data, req.data, kReqLen);
                    for (uint32_t j = 0; j < kNumKeysPerRequest; j++) {
                        append_uint32_to_char_array(j, kLog10NumKeysPerRequest,
                            key.data + kReqLen);
                        Value value;
                        value.num = (j ? 0 : req_offset + i);
                        hashtable->put(kKeyLen, (const uint8_t*)key.data, kValueLen, (const uint8_t*)value.data);
                    }
                    thread_gen_reqs[i] = req;
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }

        // 初始化Zipf分布的请求索引
        all_zipf_req_indices.resize(kNumCPUs);
        for (auto& indices : all_zipf_req_indices) {
            indices.resize(kReqSeqLen);
        }

        zipf_table_distribution<> zipf(kNumReqs, kZipfParamS);
        uint32_t core_num = sched_getcpu();  // 获取当前CPU核心号
        auto& generator = *generators[core_num];
        constexpr uint32_t kPerCoreWinInterval = kReqSeqLen / kNumCPUs;
        for (uint32_t i = 0; i < kReqSeqLen; i++) {
            auto rand_idx = zipf(generator);
            for (uint32_t j = 0; j < kNumCPUs; j++) {
                all_zipf_req_indices[j][(i + (j * kPerCoreWinInterval)) % kReqSeqLen] = rand_idx;
            }
        }

        // 初始化数组大小，但不填充数据
        // 对于性能基准测试，数组内容并不重要
        array_entries.resize(kNumArrayEntries);
    }

    void consume_array_entry(const ArrayEntry& entry) {
        std::string ciphertext;
        CryptoPP::StreamTransformationFilter stfEncryptor(
            *cbcEncryption, new CryptoPP::StringSink(ciphertext));
        stfEncryptor.Put((const unsigned char*)&entry.data, sizeof(entry));
        stfEncryptor.MessageEnd();
        std::string compressed;
        snappy::Compress(ciphertext.c_str(), ciphertext.size(), &compressed);
        volatile auto compressed_len = compressed.size();
        (void)compressed_len; // 避免未使用变量警告
    }

    void print_perf() {
        // 使用原子操作检查是否有其他线程在打印
        static std::atomic<bool> is_printing{false};
        if (is_printing.load(std::memory_order_acquire)) {
            return;
        }

        // 尝试获取打印权限
        bool expected = false;
        if (!is_printing.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            return;
        }

        // 获取性能数据
        auto us = microtime();
        uint64_t sum_reqs = 0;
        for (uint32_t i = 0; i < kNumMutatorThreads; i++) {
            sum_reqs += ACCESS_ONCE(req_cnts[i].c);
        }

        if (us - prev_us > kMaxPrintIntervalUs) {
            auto mops = ((double)(sum_reqs - prev_sum_reqs) / (us - prev_us)) * 1.098;
            mops_records.push_back(mops);
            us = microtime();
            running_us += (us - prev_us);

            // 输出当前性能数据
            std::cout << "\rCurrent MOPS: " << mops 
                     << " | Requests: " << sum_reqs 
                     << " | Time: " << running_us / 1e6 << "s" 
                     << std::flush;

            if (print_times++ >= kPrintTimes) {
                constexpr double kRatioChosenRecords = 0.1;
                uint32_t num_chosen_records = mops_records.size() * kRatioChosenRecords;
                mops_records.erase(mops_records.begin(),
                    mops_records.end() - num_chosen_records);

                std::cout << "\n\nFinal Results:" << std::endl;
                std::cout << "running time = "
                    << running_us / 1e6 << " seconds"
                    << std::endl;

                std::cout << "mops = "
                    << accumulate(mops_records.begin(), mops_records.end(), 0.0) / mops_records.size()
                    << std::endl;

                should_stop.store(true, std::memory_order_release);
            }
            prev_us = us;
            prev_sum_reqs = sum_reqs;
        }

        // 释放打印权限
        is_printing.store(false, std::memory_order_release);
    }

    void bench() {
        std::vector<std::thread> threads;
        prev_us = microtime();
        std::vector<uint64_t> per_core_req_idx(kNumCPUs, 0);

        for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
            threads.emplace_back([&, tid]() {
                uint32_t cnt = 0;
                uint32_t core_num = sched_getcpu();  // 获取实际的CPU核心号

                while (!should_stop.load(std::memory_order_relaxed)) {
                    if (cnt++ % kPrintPerIters == 0) {
                        print_perf();
                    }

                    auto req_idx = all_zipf_req_indices[core_num][per_core_req_idx[core_num]];
                    if (++per_core_req_idx[core_num] == kReqSeqLen) {
                        per_core_req_idx[core_num] = 0;
                    }

                    auto& req = all_gen_reqs[req_idx];
                    Key key;
                    memcpy(key.data, req.data, kReqLen);
                    uint32_t array_index = 0;

                    // 查询哈希表
                    for (uint32_t i = 0; i < kNumKeysPerRequest; i++) {
                        append_uint32_to_char_array(i, kLog10NumKeysPerRequest,
                            key.data + kReqLen);
                        Value value;
                        uint16_t value_len;
                        if (hashtable->get(kKeyLen, (const uint8_t*)key.data, &value_len, (uint8_t*)value.data)) {
                            array_index += value.num;
                        }
                    }

                    // 访问数组
                    array_index %= kNumArrayEntries;
                    const auto& array_entry = array_entries[array_index];
                    consume_array_entry(array_entry);

                    req_cnts[tid].c++;
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

public:
    void run() {
        std::cout << "Preparing..." << std::endl;
        prepare();
        std::cout << "Benchmarking..." << std::endl;
        bench();
    }
};

int main() {
    LocalBenchmark benchmark;
    benchmark.run();
    return 0;
}