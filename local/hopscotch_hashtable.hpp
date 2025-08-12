#pragma once

#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <thread>

#define ACCESS_ONCE(x) (*(volatile decltype(x) *)&(x))

namespace far_memory {

// 模拟原代码的复杂哈希表实现
class HopscotchHashTable {
private:
    struct BucketEntry {
        std::atomic<uint32_t> bitmap{0};
        std::atomic<uint64_t> timestamp{0};
        mutable std::shared_mutex spin;
        
        struct KeyValuePair {
            char key[12];
            char value[4];
            std::atomic<bool> valid{false};
            
            KeyValuePair() {
                memset(key, 0, sizeof(key));
                memset(value, 0, sizeof(value));
                valid.store(false, std::memory_order_relaxed);
            }
        };
        
        std::unique_ptr<KeyValuePair> kvp;
        
        BucketEntry() {
            bitmap.store(0, std::memory_order_relaxed);
            timestamp.store(0, std::memory_order_relaxed);
            kvp = std::make_unique<KeyValuePair>();
        }
    };

    static constexpr uint32_t kNeighborhood = 32;
    static constexpr uint32_t kMaxRetries = 2;
    
    const uint32_t kHashMask_;
    const uint32_t kNumEntries_;
    std::vector<BucketEntry> buckets_;

    // 哈希函数 - 模拟MurmurHash3
    uint32_t hash_32(const void* key, uint32_t len) const {
        const uint8_t* data = static_cast<const uint8_t*>(key);
        uint32_t h1 = 0x12345678;
        const uint32_t c1 = 0xcc9e2d51;
        const uint32_t c2 = 0x1b873593;
        
        for (uint32_t i = 0; i < len; i++) {
            uint32_t k1 = data[i];
            k1 *= c1;
            k1 = (k1 << 15) | (k1 >> 17);
            k1 *= c2;
            
            h1 ^= k1;
            h1 = (h1 << 13) | (h1 >> 19);
            h1 = h1 * 5 + 0xe6546b64;
        }
        
        h1 ^= len;
        h1 ^= h1 >> 16;
        h1 *= 0x85ebca6b;
        h1 ^= h1 >> 13;
        h1 *= 0xc2b2ae35;
        h1 ^= h1 >> 16;
        
        return h1;
    }

    // 位扫描函数
    uint32_t bsf_32(uint32_t mask) const {
        if (mask == 0) return 32;
        return __builtin_ctz(mask);
    }

    // 模拟线程让出
    void thread_yield() const {
        std::this_thread::yield();
    }

    // 模拟内存屏障
    void load_acquire(const std::atomic<uint32_t>* ptr, uint32_t& val) const {
        val = ptr->load(std::memory_order_acquire);
    }

    void load_acquire(const std::atomic<uint64_t>* ptr, uint64_t& val) const {
        val = ptr->load(std::memory_order_acquire);
    }

public:
    HopscotchHashTable(uint32_t num_entries_shift) 
        : kHashMask_((1 << num_entries_shift) - 1),
          kNumEntries_((1 << num_entries_shift) + kNeighborhood),
          buckets_(kNumEntries_) {
    }

    bool get(uint8_t key_len, const uint8_t* key, uint16_t* val_len, uint8_t* val) {
        uint32_t hash = hash_32(static_cast<const void*>(key), key_len);
        uint32_t bucket_idx = hash & kHashMask_;
        auto* bucket = &buckets_[bucket_idx];
        uint64_t timestamp;
        uint32_t retry_counter = 0;

        auto get_once = [&](bool lock) -> bool {
            std::unique_lock<std::shared_mutex> spin_guard;
            if (lock) {
                spin_guard = std::unique_lock<std::shared_mutex>(bucket->spin);
            }

            load_acquire(&bucket->timestamp, timestamp);
            uint32_t bitmap;
            load_acquire(&bucket->bitmap, bitmap);

            while (bitmap) {
                auto offset = bsf_32(bitmap);
                if (bucket_idx + offset >= kNumEntries_) {
                    bitmap ^= (1 << offset);
                    continue;
                }

                auto& entry = buckets_[bucket_idx + offset];
                if (entry.kvp && entry.kvp->valid.load(std::memory_order_acquire)) {
                    if (memcmp(entry.kvp->key, key, key_len) == 0) {
                        *val_len = 4; // 固定值长度
                        memcpy(val, entry.kvp->value, 4);
                        return true;
                    }
                }
                bitmap ^= (1 << offset);
            }
            return false;
        };

        // Fast path - 无锁读取
        uint64_t current_timestamp;
        do {
            if (get_once(false)) {
                return true;
            }
            current_timestamp = bucket->timestamp.load(std::memory_order_relaxed);
        } while (timestamp != current_timestamp && retry_counter++ < kMaxRetries);

        // Slow path - 带锁读取
        uint64_t final_timestamp = bucket->timestamp.load(std::memory_order_relaxed);
        if (timestamp != final_timestamp) {
            if (get_once(true)) {
                return true;
            }
        }

        return false; // 未找到
    }

    void put(uint8_t key_len, const uint8_t* key, uint16_t val_len, const uint8_t* val) {
        uint32_t hash = hash_32(static_cast<const void*>(key), key_len);
        uint32_t bucket_idx = hash & kHashMask_;
        
        auto* bucket = &buckets_[bucket_idx];
        auto orig_bucket_idx = bucket_idx;

        std::unique_lock<std::shared_mutex> bucket_lock_guard(bucket->spin);

        uint32_t bitmap;
        load_acquire(&bucket->bitmap, bitmap);
        
        // 检查是否已存在
        while (bitmap) {
            auto offset = bsf_32(bitmap);
            if (bucket_idx + offset >= kNumEntries_) {
                bitmap ^= (1 << offset);
                continue;
            }

            auto& entry = buckets_[bucket_idx + offset];
            if (entry.kvp && entry.kvp->valid.load(std::memory_order_acquire)) {
                if (memcmp(entry.kvp->key, key, key_len) == 0) {
                    // 更新现有条目
                    memcpy(entry.kvp->value, val, val_len);
                    return;
                }
            }
            bitmap ^= (1 << offset);
        }

        // 寻找空槽位
        while (bucket_idx < kNumEntries_) {
            auto& entry = buckets_[bucket_idx];
            if (!entry.kvp->valid.load(std::memory_order_acquire)) {
                break;
            }
            bucket_idx++;
        }

        if (bucket_idx == kNumEntries_) {
            // 表满了，简化处理：直接返回
            return;
        }

        uint32_t distance_to_orig_bucket;
        // Hopscotch算法：移动元素使其在邻域内
        while ((distance_to_orig_bucket = bucket_idx - orig_bucket_idx) >= kNeighborhood) {
            uint32_t distance;
            bool moved = false;
            
            for (distance = kNeighborhood - 1; distance > 0; distance--) {
                auto idx = bucket_idx - distance;
                if (idx < orig_bucket_idx) continue;
                
                auto& anchor_entry = buckets_[idx];
                std::unique_lock<std::shared_mutex> lock_guard(anchor_entry.spin);
                
                uint32_t anchor_bitmap;
                load_acquire(&anchor_entry.bitmap, anchor_bitmap);
                if (!anchor_bitmap) continue;

                auto offset = bsf_32(anchor_bitmap);
                if (idx + offset >= bucket_idx) continue;

                // 移动元素
                auto& from_entry = buckets_[idx + offset];
                auto& to_entry = buckets_[bucket_idx];
                
                if (from_entry.kvp->valid.load(std::memory_order_acquire)) {
                    // 复制数据
                    memcpy(to_entry.kvp->key, from_entry.kvp->key, 12);
                    memcpy(to_entry.kvp->value, from_entry.kvp->value, 4);
                    to_entry.kvp->valid.store(true, std::memory_order_release);
                    
                    // 更新bitmap
                    anchor_entry.bitmap.fetch_or(1 << distance, std::memory_order_acq_rel);
                    anchor_entry.timestamp.fetch_add(1, std::memory_order_acq_rel);
                    anchor_entry.bitmap.fetch_and(~(1 << offset), std::memory_order_acq_rel);
                    
                    // 清除原位置
                    from_entry.kvp->valid.store(false, std::memory_order_release);
                    
                    bucket_idx = idx + offset;
                    moved = true;
                    break;
                }
            }

            if (!moved) {
                // 无法移动，表满
                return;
            }
        }

        // 插入新元素
        auto& final_entry = buckets_[bucket_idx];
        memcpy(final_entry.kvp->key, key, key_len);
        memcpy(final_entry.kvp->value, val, val_len);
        final_entry.kvp->valid.store(true, std::memory_order_release);

        // 更新bitmap
        bucket->bitmap.fetch_or(1 << distance_to_orig_bucket, std::memory_order_acq_rel);
    }
};

} // namespace far_memory
