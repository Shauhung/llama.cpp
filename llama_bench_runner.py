#!/usr/bin/env python3
"""
llama-bench runner script
運行 llama-bench 工具並將結果保存到 CSV 文件，並計算完整的延遲指標
"""

import os
import subprocess
import sys
import json
import csv
from datetime import datetime
from pathlib import Path
import pandas as pd

# 配置 - 支持環境變量覆蓋（用於 K8s）
LLAMA_BENCH_BIN = os.getenv('LLAMA_BENCH_BIN', "./build/bin/llama-bench")
OUTPUT_DIR = os.getenv('RESULTS_DIR', "./benchmark_results")
MODELS_DIR = os.getenv('MODELS_DIR', "./models/llama_quant")

# 要測試的模型列表
MODELS = [
    "Llama3-8.0B-Q4_K_M.gguf",
    "Llama3-8.0B-Q5_K_M.gguf",
    "Llama3-8.0B-Q6_K.gguf",
    "Llama3-8.0B-Q8_0.gguf",
]

# 測試參數配置
TEST_CONFIGS = {
    "prompt_processing": {
        "description": "Prompt Processing",
        "args": ["-n", "0", "-p", "512", "-b", "512", "-ngl", "99", "-r", "5"]
    },
    "text_generation": {
        "description": "Text Generation",
        "args": ["-n", "128", "-p", "0", "-ngl", "99", "-r", "5"]
    },
    "full_test": {
        "description": "Full Test (Prompt + Generation)",
        "args": ["-p", "512", "-n", "128", "-ngl", "99", "-r", "5"]
    },
}


def ensure_directories():
    """確保輸出目錄存在"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def run_benchmark(model_path, test_name, test_args):
    """
    運行 llama-bench 並返回 JSON 格式的結果
    
    Args:
        model_path: 模型文件路徑
        test_name: 測試名稱
        test_args: 測試參數列表
    
    Returns:
        JSON 格式的結果，如果失敗返回 None
    """
    cmd = [LLAMA_BENCH_BIN, "-m", model_path, "-o", "json"] + test_args
    
    print(f"  Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10分鐘超時
        )
        
        if result.returncode != 0:
            print(f"  ❌ Error running benchmark: {result.stderr}")
            return None
        
        # 解析 JSON 輸出
        data = json.loads(result.stdout)
        return data
    except subprocess.TimeoutExpired:
        print(f"  ❌ Benchmark timed out")
        return None
    except json.JSONDecodeError as e:
        print(f"  ❌ Invalid JSON output: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return None


def calculate_latency_metrics(avg_ns, stddev_ns):
    """
    計算延遲相關指標
    
    Args:
        avg_ns: 平均延遲（納秒）
        stddev_ns: 標準差（納秒）
    
    Returns:
        字典包含各種延遲指標
    """
    avg_ms = avg_ns / 1_000_000
    stddev_ms = stddev_ns / 1_000_000
    
    return {
        'total_latency_ms': round(avg_ms, 2),
        'latency_std_ms': round(stddev_ms, 2),
        'p50_latency_ms': round(avg_ms, 2),  # 平均值作為 p50
        'p95_latency_ms': round(avg_ms + 2 * stddev_ms, 2),  # +2σ 作為 p95
        'p99_latency_ms': round(avg_ms + 3 * stddev_ms, 2),  # +3σ 作為 p99
    }


def convert_json_to_csv(json_data):
    """
    將 llama-bench JSON 輸出轉換為 CSV 並計算擴展指標
    
    Args:
        json_data: llama-bench JSON 輸出
    
    Returns:
        CSV 格式的字符串
    """
    if not isinstance(json_data, list):
        return None
    
    # 提取並擴展字段
    rows = []
    for item in json_data:
        row = dict(item)
        
        # 計算延遲指標
        if 'avg_ns' in row and 'stddev_ns' in row:
            latency_metrics = calculate_latency_metrics(row['avg_ns'], row['stddev_ns'])
            row.update(latency_metrics)
        
        rows.append(row)
    
    if not rows:
        return None
    
    # 生成 CSV
    output = []
    headers = list(rows[0].keys())
    output.append(','.join(headers))
    
    for row in rows:
        values = [str(row.get(h, '')) for h in headers]
        output.append(','.join(values))
    
    return '\n'.join(output)


def main():
    """主函數"""
    
    # 檢查 llama-bench 二進製文件是否存在
    if not os.path.exists(LLAMA_BENCH_BIN):
        print(f"❌ Error: {LLAMA_BENCH_BIN} not found!")
        print("Please build llama.cpp first with: cmake ... && cmake --build .")
        sys.exit(1)
    
    ensure_directories()
    
    # 生成時間戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 輸出文件
    output_csv = os.path.join(OUTPUT_DIR, f"llama_bench_{timestamp}.csv")
    output_json = os.path.join(OUTPUT_DIR, f"llama_bench_{timestamp}.json")
    
    print("=" * 70)
    print("🚀 llama-bench Runner (with Extended Metrics)")
    print("=" * 70)
    
    all_results = []
    all_json_results = []
    csv_header = None
    
    # 遍歷每個模型
    for model_name in MODELS:
        model_path = os.path.join(MODELS_DIR, model_name)
        
        if not os.path.exists(model_path):
            print(f"\n⚠️  Model not found: {model_path}")
            continue
        
        print(f"\n📊 Testing model: {model_name}")
        
        # 遍歷每個測試配置
        for config_key, config in TEST_CONFIGS.items():
            print(f"\n  Testing: {config['description']}")
            
            json_output = run_benchmark(model_path, config_key, config["args"])
            
            if json_output:
                all_json_results.extend(json_output)
                
                # 轉換為 CSV 格式
                csv_output = convert_json_to_csv(json_output)
                
                if csv_output:
                    lines = csv_output.strip().split('\n')
                    
                    if len(lines) < 2:
                        print(f"  ⚠️  Invalid output from benchmark")
                        continue
                    
                    # 第一次運行時，保存 header
                    if csv_header is None:
                        csv_header = lines[0]
                        all_results.append(csv_header)
                    
                    # 添加所有結果行
                    for line in lines[1:]:
                        if line.strip():
                            all_results.append(line)
                    
                    print(f"  ✅ Benchmark completed")
            else:
                print(f"  ❌ Benchmark failed")
    
    # 寫入 CSV 文件
    if all_results:
        with open(output_csv, 'w') as f:
            f.write('\n'.join(all_results))
        
        print("\n" + "=" * 70)
        print(f"✅ CSV Results saved to: {output_csv}")
        print(f"📈 Total results: {len(all_results) - 1} rows")  # -1 for header
    else:
        print("\n❌ No CSV results collected!")
    
    # 寫入 JSON 文件（用於更詳細的分析）
    if all_json_results:
        with open(output_json, 'w') as f:
            json.dump(all_json_results, f, indent=2)
        
        print(f"✅ JSON Results saved to: {output_json}")
    
    # 打印摘要統計
    if all_json_results:
        print("\n" + "=" * 70)
        print("📊 Summary Statistics")
        print("=" * 70)
        
        df = pd.DataFrame(all_json_results)
        
        # 按測試類型分組統計
        for model in df['model_type'].unique():
            model_data = df[df['model_type'] == model]
            print(f"\n{model}:")
            
            # 生成測試
            gen_tests = model_data[model_data['n_gen'] > 0]
            if len(gen_tests) > 0:
                avg_tps = gen_tests['avg_ts'].mean()
                p95_latency = gen_tests['p95_latency_ms'].mean() if 'p95_latency_ms' in gen_tests else 0
                print(f"  📤 Generation:  {avg_tps:.2f} t/s | p95: {p95_latency:.2f}ms")
            
            # Prompt 測試
            prompt_tests = model_data[(model_data['n_prompt'] > 0) & (model_data['n_gen'] == 0)]
            if len(prompt_tests) > 0:
                avg_tps = prompt_tests['avg_ts'].mean()
                p95_latency = prompt_tests['p95_latency_ms'].mean() if 'p95_latency_ms' in prompt_tests else 0
                print(f"  📥 Prompt:     {avg_tps:.2f} t/s | p95: {p95_latency:.2f}ms")
        
        print("\n" + "=" * 70)
    else:
        print("\n❌ No results collected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
