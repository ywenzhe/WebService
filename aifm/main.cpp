extern "C" {
#include <runtime/runtime.h>
}
#include "thread.h"

#include "array.hpp"
#include "deref_scope.hpp"
#include "device.hpp"
#include "helpers.hpp"
#include "manager.hpp"
#include "snappy.h"
#include "stats.hpp"
#include "zipf.hpp"

// crypto++
#include "cryptopp/aes.h"
#include "cryptopp/filters.h"
#include "cryptopp/modes.h"

#include <algorithm>
#include <array>
#include <atomic> // Used for std::atomic
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

using namespace far_memory;

#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))

namespace far_memory {
    class FarMemTest {
        private:
        // FarMemManager.
        constexpr static uint64_t kCacheSize = 563 * Region::kSize; // 1MB
        constexpr static uint64_t kFarMemSize = (17ULL << 30); // 17 GB
        constexpr static uint32_t kNumGCThreads = 12;
        constexpr static uint32_t kNumConnections = 300;

        // Hashtable.
        constexpr static uint32_t kKeyLen = 12;
        constexpr static uint32_t kValueLen = 4;
        constexpr static uint32_t kLocalHashTableNumEntriesShift = 25;
        constexpr static uint32_t kRemoteHashTableNumEntriesShift = 28;
        constexpr static uint64_t kRemoteHashTableSlabSize = (4ULL << 30) * 1.05; // 4GB
        constexpr static uint32_t kNumKVPairs = 1 << 27;

        // Array.
        constexpr static uint32_t kNumArrayEntries = 2 << 20; // 2 M entries.
        constexpr static uint32_t kArrayEntrySize = 8192;     // 8 K

        // Runtime.
        constexpr static uint32_t kNumMutatorThreads = 40;
        constexpr static double kZipfParamS = 0.85;
        constexpr static uint32_t kNumKeysPerRequest = 32;
        constexpr static uint32_t kNumReqs = (1 << 27) / kNumKeysPerRequest;
        constexpr static uint32_t kNumBenchmarkLoops = 3;
        constexpr static uint32_t kTotalTargetRequests = kNumReqs * kNumBenchmarkLoops;
        constexpr static uint32_t kLog10NumKeysPerRequest =
            helpers::static_log(10, kNumKeysPerRequest);
        constexpr static uint32_t kReqLen = kKeyLen - kLog10NumKeysPerRequest;
        constexpr static uint32_t kReqSeqLen = kNumReqs;

        // Output.
        constexpr static uint32_t kPrintPerIters = 8192;
        constexpr static uint32_t kMaxPrintIntervalUs = 1000 * 1000; // 1 second(s).

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

        using AppArray = Array<ArrayEntry, kNumArrayEntries>;

        std::unique_ptr<std::mt19937> generators[helpers::kNumCPUs];
        alignas(helpers::kHugepageSize) Req all_gen_reqs[kNumReqs];
        uint32_t all_zipf_req_indices[helpers::kNumCPUs][kReqSeqLen];

        Cnt req_cnts[kNumMutatorThreads];
        Cnt local_array_miss_cnts[kNumMutatorThreads];
        Cnt local_hashtable_miss_cnts[kNumMutatorThreads];
        Cnt per_core_req_idx[helpers::kNumCPUs];

        std::atomic<uint64_t> total_completed_reqs{0}; // Global atomic counter for completed requests
        std::atomic_flag print_lock_flag = ATOMIC_FLAG_INIT; // For protected print_progress
        uint64_t last_print_us = 0; // Last time progress was actually printed

        unsigned char key[CryptoPP::AES::DEFAULT_KEYLENGTH];
        unsigned char iv[CryptoPP::AES::BLOCKSIZE];
        std::unique_ptr<CryptoPP::CBC_Mode_ExternalCipher::Encryption> cbcEncryption;
        std::unique_ptr<CryptoPP::AES::Encryption> aesEncryption;


        inline void append_uint32_to_char_array(uint32_t n, uint32_t suffix_len,
            char* array) {
            uint32_t len = 0;
            if (n == 0) { // Handle case n=0
                array[len++] = '0';
            } else {
                while (n) {
                    auto digit = n % 10;
                    array[len++] = digit + '0';
                    n = n / 10;
                }
            }
            while (len < suffix_len) {
                array[len++] = '0';
            }
            std::reverse(array, array + suffix_len);
        }

        inline void random_string(char* data, uint32_t len) {
            BUG_ON(len <= 0);
            preempt_disable();
            auto guard = helpers::finally([&]() { preempt_enable(); });
            auto& generator = *generators[get_core_num()];
            std::uniform_int_distribution<int> distribution('a', 'z' + 1);
            for (uint32_t i = 0; i < len; i++) {
                data[i] = char(distribution(generator));
            }
        }

        inline void random_req(char* data, uint32_t tid) {
            auto tid_len = helpers::static_log(10, kNumMutatorThreads);
            random_string(data, kReqLen - tid_len);
            append_uint32_to_char_array(tid, tid_len, data + kReqLen - tid_len);
        }

        inline uint32_t random_uint32() {
            preempt_disable();
            auto guard = helpers::finally([&]() { preempt_enable(); });
            auto& generator = *generators[get_core_num()];
            std::uniform_int_distribution<uint32_t> distribution(
                0, std::numeric_limits<uint32_t>::max());
            return distribution(generator);
        }

        void prepare(GenericConcurrentHopscotch* hopscotch) {
            for (uint32_t i = 0; i < helpers::kNumCPUs; i++) {
                std::random_device rd;
                generators[i].reset(new std::mt19937(rd()));
            }
            memset(key, 0x00, CryptoPP::AES::DEFAULT_KEYLENGTH);
            memset(iv, 0x00, CryptoPP::AES::BLOCKSIZE);
            aesEncryption.reset(
                new CryptoPP::AES::Encryption(key, CryptoPP::AES::DEFAULT_KEYLENGTH));
            cbcEncryption.reset(
                new CryptoPP::CBC_Mode_ExternalCipher::Encryption(*aesEncryption, iv));
            std::vector<rt::Thread> threads;
            for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
                threads.emplace_back(rt::Thread([&, tid]() {
                    // Each thread prepares a portion of kNumReqs
                    auto num_reqs_per_thread = kNumReqs / kNumMutatorThreads;
                    auto req_offset = tid * num_reqs_per_thread;
                    auto* thread_gen_reqs = &all_gen_reqs[req_offset];
                    
                    // Loop through kNumReqs to generate and put unique KVs
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
                            DerefScope scope;
                            hopscotch->put(scope, kKeyLen, (const uint8_t*)key.data, kValueLen,
                                (uint8_t*)value.data);
                        }
                        thread_gen_reqs[i] = req;
                    }
                    }));
            }
            for (auto& thread : threads) {
                thread.Join();
            }
            preempt_disable();
            zipf_table_distribution<> zipf(kNumReqs, kZipfParamS);
            auto& generator = generators[get_core_num()];
            constexpr uint32_t kPerCoreWinInterval = kReqSeqLen / helpers::kNumCPUs;
            for (uint32_t i = 0; i < kReqSeqLen; i++) {
                auto rand_idx = zipf(*generator);
                for (uint32_t j = 0; j < helpers::kNumCPUs; j++) {
                    all_zipf_req_indices[j][(i + (j * kPerCoreWinInterval)) % kReqSeqLen] =
                        rand_idx;
                }
            }
            preempt_enable();
        }

        void prepare(AppArray* array) {
            // We may put something into array for initialization.
            // But for the performance benchmark, we just do nothing here.
        }

        void consume_array_entry(const ArrayEntry& entry) {
            std::string ciphertext;
            CryptoPP::StreamTransformationFilter stfEncryptor(
                *cbcEncryption, new CryptoPP::StringSink(ciphertext));
            stfEncryptor.Put((const unsigned char*)&entry.data, sizeof(entry));
            stfEncryptor.MessageEnd();
            std::string compressed;
            snappy::Compress(ciphertext.c_str(), ciphertext.size(), &compressed);
            auto compressed_len = compressed.size();
            ACCESS_ONCE(compressed_len);
        }

        void print_progress_status() {
            // Only one thread should enter this section to print
            if (!print_lock_flag.test_and_set()) {
                preempt_disable();
                auto us = microtime();
                // Throttle prints by time interval
                if (us - last_print_us > kMaxPrintIntervalUs) {
                    uint64_t current_completed = total_completed_reqs.load(std::memory_order_relaxed);
                    double progress_percent = (double)current_completed * 100.0 / kTotalTargetRequests;
                    
                    std::cout << "[Progress] Completed " << current_completed << " / " << kTotalTargetRequests 
                              << " requests (" << std::fixed << std::setprecision(2) << progress_percent << " %)." << std::endl;
                    last_print_us = us;
                }
                preempt_enable();
                print_lock_flag.clear();
            }
        }

        void bench(GenericConcurrentHopscotch* hopscotch, AppArray* array) {
            std::vector<rt::Thread> threads;
            
            // --- START TIMER ---
            uint64_t start_us = microtime();
            last_print_us = start_us; // Initialize for progress printing

            for (uint32_t tid = 0; tid < kNumMutatorThreads; tid++) {
                threads.emplace_back(rt::Thread([&, tid]() {
                    uint32_t cnt = 0;
                    // --- Loop until total_completed_reqs reaches kTotalTargetRequests ---
                    while (total_completed_reqs.load(std::memory_order_relaxed) < kTotalTargetRequests) {
                        // Check for progress printing periodically
                        if (unlikely(cnt++ % kPrintPerIters == 0)) {
                            preempt_disable();
                            print_progress_status(); // Call new progress function
                            preempt_enable();
                        }
                        
                        preempt_disable();
                        auto core_num = get_core_num();
                        auto req_idx =
                            all_zipf_req_indices[core_num][per_core_req_idx[core_num].c];
                        if (unlikely(++per_core_req_idx[core_num].c == kReqSeqLen)) {
                            per_core_req_idx[core_num].c = 0;
                        }
                        preempt_enable();

                        auto& req = all_gen_reqs[req_idx];
                        Key key;
                        memcpy(key.data, req.data, kReqLen);
                        uint32_t array_index = 0;
                        {
                            DerefScope scope;
                            for (uint32_t i = 0; i < kNumKeysPerRequest; i++) {
                                append_uint32_to_char_array(i, kLog10NumKeysPerRequest,
                                    key.data + kReqLen);
                                Value value;
                                uint16_t value_len;
                                bool forwarded = false;
                                hopscotch->_get(kKeyLen, (const uint8_t*)key.data,
                                    &value_len, (uint8_t*)value.data, &forwarded);
                                ACCESS_ONCE(local_hashtable_miss_cnts[tid].c) += forwarded;
                                array_index += value.num;
                            }
                        }
                        {
                            array_index %= kNumArrayEntries;
                            DerefScope scope;
                            ACCESS_ONCE(local_array_miss_cnts[tid].c) +=
                                !array->ptrs_[array_index].meta().is_present();
                            const auto& array_entry =
                                array->at</* NT = */ true>(scope, array_index);
                            preempt_disable();
                            consume_array_entry(array_entry);
                            preempt_enable();
                        }
                        preempt_disable();
                        core_num = get_core_num();
                        preempt_enable();
                        ACCESS_ONCE(req_cnts[tid].c)++;
                        // --- Increment the global completed requests counter ---
                        total_completed_reqs.fetch_add(1, std::memory_order_relaxed);
                    }
                    }));
            }
            for (auto& thread : threads) {
                thread.Join();
            }
            
            // --- END TIMER AND CALCULATE RUNTIME ---
            uint64_t end_us = microtime();
            double runtime_seconds = (double)(end_us - start_us) / 1e6;
            
            std::cout << "\n---------------------------------------------------" << std::endl;
            std::cout << "Benchmark Finished!" << std::endl;
            std::cout << "Total Requests Processed: " << kTotalTargetRequests << std::endl;
            std::cout << "Total Runtime: " << std::fixed << std::setprecision(4) << runtime_seconds << " seconds" << std::endl;
            
            // Print total miss counts if desired
            uint64_t total_ht_misses = 0;
            uint64_t total_array_misses = 0;
            for(uint32_t tid = 0; tid < kNumMutatorThreads; ++tid) {
                total_ht_misses += local_hashtable_miss_cnts[tid].c;
                total_array_misses += local_array_miss_cnts[tid].c;
            }
            std::cout << "Total Hashtable Misses: " << total_ht_misses << std::endl;
            std::cout << "Total Array Misses: " << total_array_misses << std::endl;
            std::cout << "---------------------------------------------------\n" << std::endl;
        }

        public:
        void do_work(FarMemManager* manager) {
            auto hopscotch = std::unique_ptr<GenericConcurrentHopscotch>(
                manager->allocate_concurrent_hopscotch_heap(
                    kLocalHashTableNumEntriesShift, kRemoteHashTableNumEntriesShift,
                    kRemoteHashTableSlabSize));
            std::cout << "Prepare..." << std::endl;
            prepare(hopscotch.get());
            auto array_ptr = std::unique_ptr<AppArray>(
                manager->allocate_array_heap<ArrayEntry, kNumArrayEntries>());
            array_ptr->disable_prefetch();
            prepare(array_ptr.get());
            std::cout << "Benchmarking " << kTotalTargetRequests << " requests..." << std::endl;
            bench(hopscotch.get(), array_ptr.get());
            hopscotch.reset();
            array_ptr.reset();
        }

        void run(netaddr raddr) {
            BUG_ON(madvise(all_gen_reqs, sizeof(Req) * kNumReqs, MADV_HUGEPAGE) != 0); // Adjust size here
            std::unique_ptr<FarMemManager> manager =
                std::unique_ptr<FarMemManager>(FarMemManagerFactory::build(
                    kCacheSize, kNumGCThreads,
                    new TCPDevice(raddr, kNumConnections, kFarMemSize)));
            do_work(manager.get());
            manager.reset();
        }
    };
} // namespace far_memory

int argc;
FarMemTest test;
void _main(void* arg) {
    char** argv = (char**)arg;
    std::string ip_addr_port(argv[1]);
    auto raddr = helpers::str_to_netaddr(ip_addr_port);
    test.run(raddr);
}

int main(int _argc, char* argv[]) {
    int ret;

    if (_argc < 3) {
        std::cerr << "usage: [cfg_file] [ip_addr:port]" << std::endl;
        return -EINVAL;
    }

    char conf_path[strlen(argv[1]) + 1];
    strcpy(conf_path, argv[1]);
    for (int i = 2; i < _argc; i++) {
        argv[i - 1] = argv[i];
    }
    argc = _argc - 1;

    ret = runtime_init(conf_path, _main, argv);
    if (ret) {
        std::cerr << "failed to start runtime" << std::endl;
        return ret;
    }

    return 0;
}

