# KV Cache 與模型記憶體估算筆記

這份筆記整理本地 benchmark / memory estimation 時常用到的 KV cache 概念、
RAM 估算公式，以及 llama.cpp 中相關實作位置。原本以 Python `print()` 輸出的
解釋內容，整理成 Markdown 後比較適合作為閱讀筆記。

## 1. KV Cache 是什麼

Transformer 推理時，每一步 attention 都會用到：

- `Q`：目前位置的 query
- `K`：歷史 token 的 key
- `V`：歷史 token 的 value

如果不保存歷史 token 的 `K` / `V`，每產生一個新 token 都要重算整段歷史，
成本很高。KV cache 的用途就是把已經算過的歷史 `K` / `V` 存起來，後續 token
直接重用。

簡化流程：

```text
prompt tokens
  -> llama_decode()
  -> 產生每層 attention 的 K/V
  -> 寫入 KV cache

next generated token
  -> llama_decode()
  -> 只計算新 token 的 K/V
  -> attention 直接讀取歷史 KV cache
```

所以 KV cache 是推理加速的核心資料結構之一。

## 2. Context Size 為什麼影響 RAM

`ctx_size` 代表一條 sequence 最多可以保留多少 token 的上下文。因為 KV cache
要為每個可用位置保存 `K` 和 `V`，所以 context 越大，KV cache 越大。

關係近似線性：

```text
context 翻倍 -> KV cache 約翻倍
context 四倍 -> KV cache 約四倍
```

## 3. KV Cache 估算公式

一般估算式：

```text
KV cache bytes = 2 * num_layers * ctx_size * hidden_dim * bytes_per_value
```

其中：

| 參數 | 意義 |
|---|---|
| `2` | K 與 V 兩份 cache |
| `num_layers` | Transformer layer 數 |
| `ctx_size` | context token 數 |
| `hidden_dim` | hidden size / embedding dimension |
| `bytes_per_value` | cache dtype 的 byte 數，例如 fp16 是 2 |

以 Llama3-8B 常見設定估算：

```text
num_layers      = 32
hidden_dim      = 4096
bytes_per_value = 2  // fp16
```

`ctx_size = 2048` 時：

```text
2 * 32 * 2048 * 4096 * 2
= 1,073,741,824 bytes
= 1.00 GiB
```

注意這是簡化估算。實際 llama.cpp 會依模型架構、GQA、KV dtype、offload、
batching 與 memory layout 有差異。

## 4. Llama3-8B KV Cache 對照表

使用上面的簡化公式：

| Context | KV Cache | 相對 ctx=256 |
|---:|---:|---:|
| 256 | 0.125 GiB | 1x |
| 512 | 0.25 GiB | 2x |
| 1024 | 0.50 GiB | 4x |
| 2048 | 1.00 GiB | 8x |
| 4096 | 2.00 GiB | 16x |
| 8192 | 4.00 GiB | 32x |

## 5. RAM 估算方式

粗略總記憶體：

```text
total RAM ~= model weights + KV cache + runtime overhead
```

其中：

- `model weights`：GGUF 檔案大小可作為粗略估計。
- `KV cache`：由 context size 與模型維度決定。
- `runtime overhead`：中間 tensor、allocator、metadata、alignment 等額外開銷。

常用粗估：

```text
runtime overhead ~= (model weights + KV cache) * 0.08
```

這個 8% 只是方便估算，不是 llama.cpp 的固定保證。

## 6. Llama3-8B 範例

假設：

```text
model weights = 4.58 GiB  // 例如 Q4_K_M GGUF
ctx_size      = 2048
KV cache      = 1.00 GiB
overhead      ~= 8%
```

計算：

```text
base     = 4.58 + 1.00 = 5.58 GiB
overhead = 5.58 * 0.08 = 0.45 GiB
total    = 5.58 + 0.45 = 6.03 GiB
```

所以可以粗估需要約 `6.03 GiB` RAM。

不同 context 的估算：

| Context | Model weights | KV Cache | Overhead 8% | Total |
|---:|---:|---:|---:|---:|
| 512 | 4.58 GiB | 0.25 GiB | 0.39 GiB | 5.22 GiB |
| 1024 | 4.58 GiB | 0.50 GiB | 0.41 GiB | 5.49 GiB |
| 2048 | 4.58 GiB | 1.00 GiB | 0.45 GiB | 6.03 GiB |
| 4096 | 4.58 GiB | 2.00 GiB | 0.53 GiB | 7.11 GiB |

## 7. 量化大小與 context 的組合估算

假設 Llama3-8B 不同量化權重大小如下：

| Quant | Model weights |
|---|---:|
| Q4_K_M | 4.58 GiB |
| Q5_K_M | 5.34 GiB |
| Q6_K | 6.14 GiB |
| Q8_0 | 7.95 GiB |

套用同樣 KV cache 與 8% overhead 估算：

| Quant | ctx=512 | ctx=1024 | ctx=2048 | ctx=4096 | ctx=8192 |
|---|---:|---:|---:|---:|---:|
| Q4_K_M | 5.22 GiB | 5.49 GiB | 6.03 GiB | 7.11 GiB | 9.27 GiB |
| Q5_K_M | 6.04 GiB | 6.31 GiB | 6.85 GiB | 7.93 GiB | 10.09 GiB |
| Q6_K | 6.90 GiB | 7.17 GiB | 7.71 GiB | 8.79 GiB | 10.95 GiB |
| Q8_0 | 8.86 GiB | 9.13 GiB | 9.67 GiB | 10.75 GiB | 12.91 GiB |

## 8. llama.cpp 中的實作位置

KV cache 相關核心檔案：

| 檔案 | 用途 |
|---|---|
| `src/llama-kv-cache.h` | KV cache class 定義與介面 |
| `src/llama-kv-cache.cpp` | KV cache 建立、張量配置與記憶體分配 |
| `src/llama-kv-cache-iswa.h` | ISWA 版本定義 |
| `src/llama-kv-cache-iswa.cpp` | ISWA 版本實作 |
| `src/llama-kv-cells.h` | KV cell 管理 |
| `src/llama-memory.h` | memory 管理介面 |
| `src/llama-batch.h` | batch 結構 |

初始化概念流程：

```text
llama_context_new()
  -> 建立 llama_context
  -> 建立 llama_kv_cache
  -> 為每一層建立 K / V tensor
  -> ggml_backend_alloc_ctx_tensors_from_buft()
  -> 實際配置 CPU RAM 或 GPU VRAM
```

推理時：

```text
input token
  -> llama_decode()
  -> 計算 attention K/V
  -> 寫入對應 sequence / position 的 KV cache
  -> 下一步 decode 重用歷史 K/V
```

## 9. Debug 提示

想追 KV cache 配置，可以看：

```text
src/llama-kv-cache.cpp
```

適合觀察的位置：

- 建立 K / V tensor 的地方。
- 呼叫 `ggml_backend_alloc_ctx_tensors_from_buft()` 的地方。
- 印出 KV buffer size 的 log。

啟動 server 時，也可以看載入模型的 log，通常會看到類似：

```text
KV buffer size = ... MiB
```

如果要更細的 log，可嘗試：

```bash
LLAMA_LOG_DEBUG=1 ./build/bin/llama-server -m model.gguf -c 2048
```

## 10. Context Size 建議

粗略建議：

| Context | 適合情境 |
|---:|---|
| 512 | 短 prompt、低記憶體需求 |
| 2048 | 一般對話與 benchmark 常用折衷 |
| 4096 | 長對話、較長文件 |
| 8192+ | 長文件或 agent 場景，需要更多 RAM/VRAM |

benchmark 時應固定 context size，否則結果會混入 KV cache 大小與記憶體壓力差異。
