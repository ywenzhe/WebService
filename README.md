# In-Memory WebService Access Benchmark

该项目旨在模拟并评估一个假想的 Web 服务对大型数据集进行并发访问和处理时的性能。它模拟了从数据存储中查找键、获取数据、加密和压缩数据再返回给客户端的典型工作流程。

下面主要介绍本地运行程序，另外 AIFM 相关内容可见：
- [AIFM 环境搭建](https://jianmucloud.feishu.cn/docx/EalGd85t6oFF8Mx5KQUcfwYOnDe?from=from_copylink)
- [AIFM 下 WebService 运行结果及分析](https://jianmucloud.feishu.cn/docx/Vct3dhBx0oPzDkxoFdbcu803nZe?from=from_copylink)
- [AIFM API](https://jianmucloud.feishu.cn/docx/PmDUdaqAiosqHUxihzvc9IrqnOc?from=from_copylink)

## 项目概述

本项目模拟了一个 Web 服务后端的数据处理流程，主要包含以下步骤：
1.  **数据初始化**: 在内存中构建一个大型的键值对哈希表和一个8KB大小对象的数据数组。
2.  **请求模拟**: 多个并发线程（客户端）模拟请求，每个请求包含对哈希表的多次查找和一个数据数组的获取操作。
3.  **数据处理**: 获取到的数据会经过加密（使用 Crypto++）和压缩（使用 Snappy）处理。
4.  **性能测量**: 统计在给定时间内能处理的总请求数，并计算吞吐量（请求数/秒）。

请求模式遵循 Zipfian 分布，以模拟真实世界中访问热点不均的数据访问模式。

## 核心功能模型

每个客户端（线程）在一次模拟请求中执行以下操作：
1.  **查找键**: 向内存中的哈希表发送 **32** 个查找请求以获取键值对。
2.  **获取数据**: 从一个内存数组中取出 **一个 8KB 大小** 的数据元素。
3.  **数据加密**: 使用 **Crypto++** 库对获取到的 8KB 元素进行 AES 加密。
4.  **数据压缩**: 使用 **Snappy** 库对加密后的数据进行压缩。
5.  **返回数据**: 模拟将处理后的数据返回给客户端（通过 `ACCESS_ONCE` 确保操作不会被编译器优化掉）。

整个基准测试模拟 **2000万次** 请求。

## 数据集详情

本项目使用一个总计约 **26 GB** 的内存数据集进行评估，具体构成如下：

*   **远程可访问哈希表 (Remoteable Hashtable)**：
    *   包含 **1.28 亿 (128M)** 个键值对（`kNumKVPairs = 1ULL << 27`）。
    *   总数据大小约为 **10 GB**：
        *   **6 GB** 用于索引数据（哈希表本身开销）。
        *   **4 GB** 用于值数据。
*   **远程可访问数组 (Remoteable Array)**：
    *   包含 **200 万 (2M)** 个 8KB 大小的对象（`kNumArrayEntries = 2ULL << 20`）。
    *   总数据大小约为 **16 GB**（2M \* 8KB = 16GB）。

哈希表使用 Intel TBB (Threading Building Blocks) 中的 `concurrent_unordered_map` 实现，以支持高效的并发访问。

## 依赖

本项目依赖以下库：

-   **Intel TBB (Threading Building Blocks)**: 提供高性能的并发哈希表 (`tbb::concurrent_unordered_map`)。
-   **Crypto++**: 提供加密功能 (AES).
-   **Snappy**: 提供快速数据压缩功能.
-   **Zipfian 分布实现**: `zipf.hpp` 和 `zipf.ipp` 文件已包含在项目中。

安装依赖
```bash
sudo apt-get update
sudo apt-get install -y libtbb-dev libcryptopp-dev libsnappy-dev
```

## 构建与运行

提供的 `run.sh` 脚本可以自动化编译和运行过程。

### 构建步骤:

```bash
#!/bin/bash

# --- 编译 ---
echo ">>> Compiling main.cpp..."

g++ -std=c++17 -O3 main.cpp -o main -I. \
    -ltbb -lcryptopp -lsnappy -pthread

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo ">>> Compilation failed!"
    exit 1
fi

echo ">>> Compilation successful."
```

### 运行步骤:

**⚠️ 内存警告 ⚠️**
本程序在运行时将消耗大约 **26 GB** 的内存。请确保您的系统有足够的物理内存，否则可能导致系统不稳定或程序崩溃。

直接运行 `./run.sh` 即可。

## 基准测试参数

以下是代码中定义的一些关键参数，它们决定了基准测试的规模和行为：

*   **数据集大小**:
    *   `kNumKVPairs`: 134,217,728 (128M) - 哈希表中键值对的数量。
    *   `kNumArrayEntries`: 2,097,152 (2M) - 数组中元素的数量。
    *   `kArrayEntrySize`: 8192 Bytes (8KB) - 数组中每个元素的大小。
*   **工作负载**:
    *   `kNumMutatorThreads`: 40 - 用于填充数据和执行请求的并发线程数。
    *   `kZipfParamS`: 0.85 - Zipfian 分布的 skewness 参数，值越大，数据访问的集中度越高。
    *   `kNumKeysPerRequest`: 32 - 每个客户端请求会查找的键的数量。
    *   `kTotalTargetRequests`: 20,000,000 (2千万) - 基准测试中要完成的总请求数。

## 输出示例

程序运行时会实时打印进度，并在结束时输出最终的性能汇总：

```
Preparing dataset (134217728 K/V pairs, 2097152 array entries)...
Allocating memory for HashMap and Array (approx. 26GB)...
Memory allocated.
Populating HashMap with 40 threads...
HashMap populated successfully. Total size: 134217728
Generating Zipfian request sequence...
Preparation complete.

Starting benchmark with 40 threads...
Total target requests: 20000000
[Progress] 99.42% | Reqs: 19884780/20000000 | TPS: 601762
---------------------------------------------------
Benchmark Finished!
Total Requests Processed: 20000039
Total Runtime: 34.0445 seconds
Throughput: 587468.26 reqs/sec
---------------------------------------------------
```
*(上述输出数据为示例，实际运行结果可能因硬件和系统环境而异)*

## 代码结构简述

*   **`main.cpp`**: 包含 `LocalWebServiceTest` 类，是项目的核心。
    *   **参数定义**: 核心的基准测试参数均使用 `static constexpr` 定义在类内部。
    *   **数据结构**: 使用 `tbb::concurrent_unordered_map` 作为哈希表，`std::vector<ArrayEntry>` 作为数组。
    *   **`prepare()` 方法**: 负责初始化数据集，包括分配内存、填充哈希表和生成 Zipfian 请求序列。此步骤是多线程的。
    *   **`bench()` 方法**: 执行基准测试的核心逻辑。多个线程并发地执行查找、读取、加密和压缩操作。包含进度报告机制。
    *   **`consume_array_entry()`**: 封装了加密和压缩逻辑。
*   **`run.sh`**: 简单的 shell 脚本，用于编译 `main.cpp` 并运行生成的可执行文件。
*   **`zipf.hpp`, `zipf.ipp`**: 提供了 Zipfian 分布的C++实现，供项目内部使用。