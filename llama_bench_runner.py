#!/usr/bin/env python3
"""
llama-bench runner script
運行 llama-bench 工具並將結果保存到 CSV 文件
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

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
        "args": ["-n", "0", "-p", "512", "-b", "512", "-ngl", "99"]
    },
    "text_generation": {
        "description": "Text Generation",
        "args": ["-n", "128", "-p", "0", "-ngl", "99"]
    },
    "full_test": {
        "description": "Full Test (Prompt + Generation)",
        "args": ["-p", "512", "-n", "128", "-ngl", "99"]
    },
}


def ensure_directories():
    """確保輸出目錄存在"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def run_benchmark(model_path, test_name, test_args):
    """
    運行 llama-bench 並返回 CSV 輸出
    
    Args:
        model_path: 模型文件路徑
        test_name: 測試名稱
        test_args: 測試參數列表
    
    Returns:
        CSV 格式的結果字符串，如果失敗返回 None
    """
    cmd = [LLAMA_BENCH_BIN, "-m", model_path, "-o", "csv"] + test_args
    
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
        
        return result.stdout
    except subprocess.TimeoutExpired:
        print(f"  ❌ Benchmark timed out")
        return None
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return None


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
    output_csv = os.path.join(OUTPUT_DIR, f"llama_bench_{timestamp}.csv")
    
    print("=" * 70)
    print("🚀 llama-bench Runner")
    print("=" * 70)
    
    all_results = []
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
            
            csv_output = run_benchmark(model_path, config_key, config["args"])
            
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
        print(f"✅ Results saved to: {output_csv}")
        print(f"📈 Total results: {len(all_results) - 1} rows")  # -1 for header
        print("=" * 70)
    else:
        print("\n❌ No results collected!")
        sys.exit(1)


if __name__ == "__main__":
    main()
