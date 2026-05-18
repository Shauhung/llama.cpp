# KV Cache 实现流程 - llama.cpp

本文档详细说明了 llama.cpp 中 KV Cache（Key-Value 缓存）的工作原理和实现流程。

## 📚 目录

1. [概述](#概述)
2. [KV Cache 的作用](#kv-cache-的作用)
3. [初始化流程](#初始化流程)
4. [核心代码位置](#核心代码位置)
5. [内存计算](#内存计算)
6. [推理时的使用](#推理时的使用)
7. [调试方法](#调试方法)

---

## 概述

### 什么是 KV Cache？

KV Cache 是在 Transformer 模型推理时用来存储历史 tokens 的 Key（K）和 Value（V）向量的缓存。

**没有 KV Cache 的情况**：
```
推理第 1 token: 计算 K₁、V₁
推理第 2 token: 重新计算 K₁、V₁ ❌ 浪费算力！计算新的 K₂、V₂
推理第 3 token: 重新计算 K₁、V₁、K₂、V₂ ❌ 更加浪费！计算新的 K₃、V₃
```
性能：O(n²) - 非常慢

**有 KV Cache 的情况**：
```
推理第 1 token: 计算 K₁、V₁ → 保存到 KV Cache
推理第 2 token: 直接使用 K₁、V₁ ✅ 计算新的 K₂、V₂ → 保存到 KV Cache
推理第 3 token: 直接使用 K₁、V₁、K₂、V₂ ✅ 计算新的 K₃、V₃ → 保存到 KV Cache
```
性能：O(n) - 非常快 ⚡

---

## KV Cache 的作用

### 核心优势

| 方面 | 说明 |
|------|------|
| **性能** | 避免重复计算，速度提升 10-50 倍 |
| **内存** | 用相对小的额外内存换取巨大的性能提升 |
| **可预测性** | 提前分配，推理过程中不需要动态内存分配 |
| **并发** | 支持多个序列（sequence）独立使用各自的 KV Cache |

### 为什么要提前分配？

即使模型刚加载，还没开始推理，llama.cpp 也会立即分配好 KV Cache 内存：

1. **效率高**：避免推理中途的内存分配延迟
2. **性能一致**：不会因为内存碎片导致性能波动
3. **失败检查**：提前检查机器是否有足够 RAM
4. **算法设计**：KV Cache 是环形缓冲区，需要提前决定大小

---

## 初始化流程

### 完整的加载链

```
┌─────────────────────────────────────────┐
│ 1. 应用程序启动                          │
│    (例如: ./llama-server -m model.gguf) │
└──────────────┬──────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│ 2. llama_model_load()                   │
│    加载模型权重（第 1 部分）             │
└──────────────┬──────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│ 3. llama_context_new()                  │
│    创建推理上下文                        │
│    位置: src/llama-context.cpp          │
└──────────────┬──────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│ 4. new llama_kv_cache()                 │
│    开始 KV Cache 初始化                  │
│    位置: src/llama-kv-cache.cpp:71      │
└──────────────┬──────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│ 5. for (uint32_t il = 0; il < layers)  │
│    {                                     │
│      遍历模型的每一层（32 层）           │
│    }                                     │
└──────────────┬──────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│ 6. ggml_new_tensor_3d()                 │
│    为每一层创建 K、V 张量                │
│    位置: src/llama-kv-cache.cpp:207-208 │
│                                          │
│    K: [n_embd, ctx_size, n_stream]      │
│    V: [n_embd, ctx_size, n_stream]      │
└──────────────┬──────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│ 7. ggml_backend_alloc_ctx_tensors...()  │
│    分配实际物理内存 ⭐                    │
│    位置: src/llama-kv-cache.cpp:241     │
│                                          │
│    可能分配位置：                        │
│    • CPU RAM                            │
│    • GPU VRAM (CUDA/Metal/ROCm)        │
└──────────────┬──────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│ 8. LLAMA_LOG_INFO()                     │
│    输出: "KV buffer size = X MiB"       │
│    位置: src/llama-kv-cache.cpp:247     │
└──────────────┬──────────────────────────┘
               ↓
┌──────────────────────────────────────────┐
│ ✅ KV Cache 初始化完成                    │
│    可以开始推理                          │
└──────────────────────────────────────────┘
```

---

## 核心代码位置

### 关键文件

| 文件 | 作用 |
|------|------|
| `src/llama-kv-cache.h` | KV Cache 类定义 |
| `src/llama-kv-cache.cpp` | KV Cache 实现（**核心**） |
| `src/llama-kv-cells.h` | KV 单元管理 |
| `src/llama-memory.h` | 内存接口 |
| `include/llama.h` | 公开 API |

### 关键代码片段

#### 1. 构造函数（第 71 行）

```cpp
llama_kv_cache::llama_kv_cache(
    const llama_model & model,
    ggml_type   type_k,        // 数据类型 (fp16)
    ggml_type   type_v,        // 数据类型 (fp16)
    bool   v_trans,            // V 是否转置
    bool   offload,            // 是否 offload 到 GPU
    bool   unified,            // 是否统一流
    uint32_t   kv_size,        // ← Context Size (2048)
    uint32_t   n_seq_max,      // 最大序列数
    uint32_t   n_pad,          // 填充大小
    uint32_t   n_swa,          // SWA 大小
    llama_swa_type   swa_type, // SWA 类型
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse) { ... }
```

#### 2. 创建张量（第 207-208 行）

```cpp
// 为每一层创建 K 和 V 张量
ggml_tensor * k = has_k ? ggml_new_tensor_3d(
    ctx,              // ggml 上下文
    type_k,           // 数据类型 (fp16 = 2 字节)
    n_embd_k_gqa,     // 维度 (4096)
    kv_size,          // Context 大小 (2048)
    n_stream          // 流数量 (1)
) : nullptr;

ggml_tensor * v = has_v ? ggml_new_tensor_3d(
    ctx,
    type_v,
    n_embd_v_gqa,     // 维度 (4096)
    kv_size,          // Context 大小 (2048)
    n_stream
) : nullptr;

// 结果：
// K tensor shape: [4096, 2048, 1]
// V tensor shape: [4096, 2048, 1]
```

#### 3. 分配内存（第 241 行） ⭐

```cpp
// 这一行真正分配了物理内存！
ggml_backend_buffer_t buf = 
    ggml_backend_alloc_ctx_tensors_from_buft(
        ctx.get(),    // ggml 上下文
        buft          // buffer type (CPU/GPU)
    );

if (!buf) {
    throw std::runtime_error(
        "failed to allocate buffer for kv cache");
}
```

buft 的类型决定了分配位置：
- `ggml_backend_cpu_buffer_type()` → CPU RAM
- `ggml_backend_dev_buffer_type(dev)` → GPU VRAM
- `ggml_backend_cuda_buffer_type()` → CUDA 内存
- `ggml_backend_metal_buffer_type()` → Metal 内存

#### 4. 记录信息（第 247 行）

```cpp
LLAMA_LOG_INFO(
    "%s: %10s KV buffer size = %8.2f MiB\n",
    __func__,
    ggml_backend_buffer_name(buf),  // "CUDA" / "CPU" / etc
    ggml_backend_buffer_get_size(buf)/1024.0/1024.0
);

// 输出示例：
// llama_kv_cache: CUDA KV buffer size = 1024.00 MiB
```

---

## 内存计算

### Llama3-8B 的 KV Cache 计算

```
对于 Context Size = 2048:

KV Cache 大小 = 2 × num_layers × ctx_size × hidden_dim × bytes_per_param
              = 2 × 32 × 2048 × 4096 × 2
              = 1,073,741,824 bytes
              = 1.00 GB

详细分解：
• 2：K 和 V 两个缓存
• 32：Llama3-8B 的层数
• 2048：Context Size（保存 2048 个 tokens 的 K、V）
• 4096：隐藏层维度
• 2：fp16 格式（16 位浮点数 = 2 字节）
```

### Context Size 的影响

| Context | KV Cache | 倍数 |
|---------|----------|------|
| 256 | 0.13 GB | 1x |
| 512 | 0.25 GB | 2x |
| 1024 | 0.50 GB | 4x |
| **2048** | **1.00 GB** | **8x** |
| 4096 | 2.00 GB | 16x |
| 8192 | 4.00 GB | 32x |

**结论**：Context Size 和 KV Cache 成**正比**

### 总内存占用

```
总 RAM = 模型权重 + KV Cache + 系统开销
       = 4.58 GB + 1.00 GB + 0.45 GB (≈8%)
       = 6.03 GB
```

---

## 推理时的使用

### 推理步骤

```
推理步骤：
1. 用户输入 token（例如 "你好"）
2. 模型计算 attention
3. 生成新的 K、V 向量
4. ✅ 写入 KV Cache 的槽位
5. 生成输出 token
6. 返回给用户
7. 重复步骤 2-6，直到生成结束

每一步都会从 KV Cache 中读取历史的 K、V
这样模型就能"记住"整个对话历史
```

### 关键数据结构

#### slot_info 结构体

位置：`src/llama-kv-cache.h`

```cpp
struct slot_info {
    uint32_t s0;                           // 起始流编号
    uint32_t s1;                           // 结束流编号
    std::vector<llama_seq_id> strm;        // 序列 ID 列表
    std::vector<std::vector<uint32_t>> idxs; // 槽位索引
    
    // 示例：
    // strm = [1, 2, 3]
    // idxs = [[100, 101, ...], [200, 201, ...], ...]
    // 表示序列 1 的 tokens 存在 100, 101, ... 位置
};
```

这个结构体负责管理：
- 每个序列在 KV Cache 中的位置
- 支持多序列并发处理
- 回收和重用槽位

---

## 调试方法

### 1. 查看日志输出

运行模型加载时，会输出 KV Cache 分配信息：

```bash
./build/bin/llama-server -m model.gguf -c 2048

# 输出示例：
# llama_kv_cache: CUDA KV buffer size = 1024.00 MiB
# llama_kv_cache: size = 1024.00 MiB (2048 cells, 32 layers)
#                K (f16): 512.00 MiB, V (f16): 512.00 MiB
```

### 2. 启用详细日志

```bash
export LLAMA_LOG_DEBUG=1
./build/bin/llama-server -m model.gguf -c 2048

# 会输出更多的层级信息和初始化细节
```

### 3. 设置断点调试

在 `src/llama-kv-cache.cpp` 的关键位置设置断点：

| 断点 | 位置 | 用途 |
|------|------|------|
| 第 71 行 | 构造函数 | 看 KV Cache 初始化开始 |
| 第 207 行 | 张量创建 | 看张量大小是否正确 |
| 第 241 行 | 内存分配 | **最关键** - 看内存真正分配的地方 |
| 第 247 行 | 日志输出 | 看最终的大小输出 |

### 4. 检查内存占用

**在 macOS 上**：
```bash
# 启动 server
./build/bin/llama-server -m model.gguf -c 2048 &

# 查看进程内存
top -p $(pgrep llama-server)
```

**在 Linux 上**：
```bash
# 查看内存映射
cat /proc/$(pgrep llama-server)/status | grep Vm

# 或使用 ps
ps aux | grep llama-server
```

### 5. 实验不同 Context Size

```bash
# Context 512
./build/bin/llama-server -m model.gguf -c 512

# Context 2048（默认）
./build/bin/llama-server -m model.gguf -c 2048

# Context 4096（需要更多内存）
./build/bin/llama-server -m model.gguf -c 4096

# 比较日志输出中的 "KV buffer size" 数字
```

---

## 性能建议

### 选择合适的 Context Size

| 用途 | 推荐 Context | 内存 (Q4) |
|------|-------------|---------|
| 短问答 | 512 | 5.2 GB |
| 一般对话 | 2048 | 6.0 GB |
| 长文档理解 | 4096 | 7.1 GB |
| 非常长的输入 | 8192 | 9.3 GB |

### GPU vs CPU

**使用 GPU 加速** (`-ngl 99`)：
```
内存占用 ≈ 模型大小 + KV Cache 大小 + 少量 CPU RAM
```

**纯 CPU**：
```
内存占用 ≈ 模型大小 + KV Cache 大小 + 系统开销
```

---

## 相关函数

### 内存计算

在 `src/llama-kv-cache.cpp` 中：

```cpp
size_t size_k_bytes() const;  // K 的总字节数
size_t size_v_bytes() const;  // V 的总字节数
size_t size() const;          // 总 KV Cache 大小
```

### 槽位管理

```cpp
slot_info find_slot(
    const llama_ubatch & ubatch,
    bool cont) const;           // 查找可用槽位

void apply_ubatch(
    const slot_info & sinfo,
    const llama_ubatch & ubatch); // 应用批处理
```

---

## 总结

| 阶段 | 位置 | 关键动作 |
|------|------|---------|
| **初始化** | llama-kv-cache.cpp:71 | 构造函数开始 |
| **张量创建** | llama-kv-cache.cpp:207 | 为每层创建 K、V 张量 |
| **内存分配** | llama-kv-cache.cpp:241 | ⭐ 真正分配物理内存 |
| **记录日志** | llama-kv-cache.cpp:247 | 输出分配大小 |
| **推理时使用** | 推理循环中 | 读写 KV Cache 槽位 |

KV Cache 是 llama.cpp 高效推理的核心机制，通过提前分配和合理管理，实现了从 O(n²) 到 O(n) 的性能飞跃。

---

## 参考资源

- [llama.h 头文件](../include/llama.h) - API 定义
- [llama-context.cpp](../src/llama-context.cpp) - 上下文创建
- [llama-model.cpp](../src/llama-model.cpp) - 模型加载
- [CONTRIBUTING.md](../CONTRIBUTING.md) - 贡献指南
