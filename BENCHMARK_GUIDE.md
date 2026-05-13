# Comprehensive Benchmark Suite

運行完整的 llama.cpp 性能測試，包括純推理基準和實際延遲測試。

## 📋 腳本概述

### 1. `llama_bench_runner.py` - Pure Inference Benchmark
測試 llama-bench（批量推理）性能

**特點:**
- ✅ 快速基準測試
- ✅ 計算延遲分位數（p50, p95, p99）
- ✅ 支持多個測試配置
- ✅ 輸出 CSV 和 JSON

**輸出指標:**
```
- avg_ts: 平均吞吐量 (tokens/sec)
- total_latency_ms: 平均延遲
- p50_latency_ms: 中位延遲
- p95_latency_ms: 95 分位延遲
- p99_latency_ms: 99 分位延遲
```

### 2. `benchmark_server.py` - Real-World Latency Benchmark
測試 llama-server（HTTP 服務）的真實延遲

**特點:**
- ✅ 模擬真實 HTTP 請求
- ✅ 收集完整延遲數據
- ✅ 計算百分位數（p50, p95, p99）
- ✅ 監測資源使用和錯誤率

**輸出指標:**
```
- total_latency_avg_ms: 平均端到端延遲
- p95_latency_ms: 95 分位延遲（真實）
- p99_latency_ms: 99 分位延遲（真實）
- avg_tokens_per_sec: 實際吞吐量（含 HTTP 開銷）
- error_rate: 錯誤率 (%)
- avg_ram_gb: 平均 RAM 使用
- peak_ram_gb: 峰值 RAM 使用
```

### 3. `run_all_benchmarks.py` - Unified Runner
同時運行兩個基準測試，生成對比報告

**階段:**
1. Phase 1: llama-bench 測試
2. Phase 2: llama-server 測試  
3. Phase 3: 生成對比報告

## 🚀 快速開始

### 方式 A: 只運行 llama-bench
```bash
python3 llama_bench_runner.py
```

### 方式 B: 只運行 llama-server 測試
```bash
python3 benchmark_server.py
```

### 方式 C: 運行完整測試（推薦）
```bash
python3 run_all_benchmarks.py
```

## ⚙️ 配置

### llama_bench_runner.py

編輯文件中的 `TEST_CONFIGS`:
```python
TEST_CONFIGS = {
    "test_name": {
        "description": "Test Description",
        "args": ["-n", "128", "-p", "512", "-ngl", "99"]
    }
}
```

### benchmark_server.py

編輯 `MODELS` 列表:
```python
MODELS = [
    ("./models/llama_quant/Llama3-8.0B-Q4_K_M.gguf", 2048),
    ("./models/llama_quant/Llama3-8.0B-Q5_K_M.gguf", 2048),
]
```

修改請求配置:
```python
N_RUNS = 100                    # 每個模型運行次數
REQUEST_TIMEOUT = 120           # 請求超時時間 (秒)
PROMPT = "Your prompt here"     # 測試 prompt
```

## 📊 輸出文件

所有結果保存在 `benchmark_results/` 目錄:

```
benchmark_results/
├── llama_bench_20260512_223430.csv          # llama-bench 結果（CSV）
├── llama_bench_20260512_223430.json         # llama-bench 結果（JSON）
├── llama_server_benchmark_20260512_223430.csv   # llama-server 結果（CSV）
└── benchmark_comparison_20260512_223430.md  # 對比報告（Markdown）
```

## 📈 結果分析

### CSV 列說明

#### llama_bench 結果
| 列名 | 說明 |
|------|------|
| model_type | 模型名稱和量化類型 |
| avg_ts | 平均吞吐量（tokens/sec） |
| stddev_ts | 吞吐量標準差 |
| avg_ns | 平均延遲（納秒） |
| total_latency_ms | 平均延遲（毫秒） |
| p50_latency_ms | 中位延遲 |
| p95_latency_ms | 95 分位延遲 |
| p99_latency_ms | 99 分位延遲 |
| n_prompt | Prompt tokens 數 |
| n_gen | 生成 tokens 數 |

#### llama_server 結果
| 列名 | 說明 |
|------|------|
| model_name | 模型名稱 |
| total_latency_avg_ms | 平均端到端延遲 |
| p95_latency_ms | 95 分位延遲 |
| p99_latency_ms | 99 分位延遲 |
| avg_tokens_per_sec | 實際吞吐量 |
| error_rate | 錯誤率 (%) |
| avg_ram_gb | 平均 RAM 使用 |
| peak_ram_gb | 峰值 RAM 使用 |

## 📚 性能指標解釋

### 吞吐量 vs 延遲

- **llama-bench (avg_ts)**: 批量推理吞吐量，最大理論值
- **benchmark_server (avg_tokens_per_sec)**: 實際吞吐量，包含 HTTP 開銷

### 延遲分位數

- **p50**: 50% 的請求延遲在此以下（中位）
- **p95**: 95% 的請求延遲在此以下（良好）
- **p99**: 99% 的請求延遲在此以下（極端情況）

更低的 p95/p99 = 更穩定的性能

## 🎯 選擇量化類型的建議

基於通常的性能測試結果：

| 量化類型 | 生成速度 | 質量 | 使用場景 |
|---------|--------|------|--------|
| **Q4_K_M** | ⚡⚡⚡ 最快 | 👍 良好 | 實時應用、低延遲 |
| **Q5_K_M** | ⚡⚡ 中等 | 👍👍 更好 | 平衡方案 |
| **Q6_K** | ⚡ 較慢 | 👍👍👍 最好 | 離線分析 |
| **Q8_0** | 🐌 最慢 | 👍👍👍 最好 | 高精度需求 |

## 🔧 故障排除

### Q: Server 啟動失敗
```bash
# 檢查端口是否被占用
lsof -i :8080

# 強制終止舊進程
pkill -f llama-server

# 檢查模型文件路徑
ls -la ./models/llama_quant/
```

### Q: Timeout 錯誤
增加 `REQUEST_TIMEOUT`:
```python
REQUEST_TIMEOUT = 300  # 增加到 5 分鐘
```

### Q: 記憶體不足
減少 `N_RUNS` 或使用量化度更高的模型：
```python
N_RUNS = 50  # 改為 50 次運行
```

### Q: CSV 無法打開
確保使用支持大型 CSV 的工具：
```bash
# 用 pandas 查看
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('./benchmark_results/llama_bench_*.csv')
print(df.head())
EOF
```

## 💡 最佳實踐

1. **第一次運行** - 用 llama_bench_runner.py 快速測試
2. **驗證結果** - 用 benchmark_server.py 測試真實延遲
3. **對比分析** - 用 run_all_benchmarks.py 同時運行

4. **多次運行** - 運行多次獲得更可靠的統計數據
5. **監控資源** - 檢查 RAM 使用，確保系統穩定

## 📖 相關文檔

- [llama-bench 文檔](tools/llama-bench/README.md)
- [llama-server 文檔](tools/server/README.md)

## 🤝 貢獻

如果發現問題或有改進建議，歡迎提交 Issue 或 PR。

## 📝 License

遵循 llama.cpp 的許可證。
