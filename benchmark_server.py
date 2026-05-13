#!/usr/bin/env python3
"""
llama-server Load Testing & Latency Benchmarking
收集完整的延遲指標：TTFT、p50、p95、p99、error_rate 等
"""

import time
import requests
import statistics
import csv
import json
import os
import psutil
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np

# 設定
URL_BASE = os.getenv('LLAMA_SERVER_URL', "http://localhost:8080")
RESULTS_DIR = os.getenv('RESULTS_DIR', "./benchmark_results")
PROMPT = "Explain quantum mechanics in simple terms."
N_RUNS = 100
SERVER_PORT = 8080
SERVER_STARTUP_WAIT = 15
REQUEST_TIMEOUT = 120

# 要測試的模型列表 (model_path, context_size)
MODELS = [
    ("./models/llama_quant/Llama3-8.0B-Q4_K_M.gguf", 2048),
    ("./models/llama_quant/Llama3-8.0B-Q5_K_M.gguf", 2048),
    ("./models/llama_quant/Llama3-8.0B-Q6_K.gguf", 2048),
    ("./models/llama_quant/Llama3-8.0B-Q8_0.gguf", 2048),
]


def ensure_results_dir():
    """確保結果目錄存在"""
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def kill_server():
    """終止所有 llama-server 進程"""
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(2)


def start_server(model_path, ctx_size=2048):
    """啟動 llama-server 並載入模型"""
    print(f"\n🚀 Starting server with model: {os.path.basename(model_path)} (ctx={ctx_size})")
    
    try:
        proc = subprocess.Popen(
            ["./build/bin/llama-server", 
             "-m", model_path,
             "--port", str(SERVER_PORT),
             "-n", "512",
             "-t", "8",
             "-c", str(ctx_size),
             "-ngl", "99"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
            text=True,
            bufsize=1
        )
        
        print(f"⏳ Waiting for server to start...")
        time.sleep(SERVER_STARTUP_WAIT)
        
        # 檢查進程是否還活著
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            print(f"❌ Server startup failed")
            print(f"stderr: {stderr[-500:]}")
            return None
        
        # 重試連接
        for retry in range(30):
            try:
                response = requests.get(f"{URL_BASE}/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Server ready")
                    return proc
            except:
                time.sleep(1)
        
        print(f"❌ Could not connect to server")
        kill_server()
        return None
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None


def get_model_info():
    """獲取模型信息"""
    try:
        r = requests.get(f"{URL_BASE}/v1/models")
        data = r.json()['data'][0]
        meta = data.get('meta', {})
        return {
            "name": data.get('id', "Unknown"),
            "n_ctx": meta.get('n_ctx', 0),
            "size_gb": round(meta.get('size', 0) / (1024**3), 2),
            "params_b": round(meta.get('n_params', 0) / 1e9, 2)
        }
    except Exception as e:
        print(f"❌ Error getting model info: {e}")
        return None


def get_process_ram(pid):
    """獲取進程 RAM 使用量"""
    try:
        proc = psutil.Process(pid)
        return proc.memory_info().rss / (1024 ** 3)
    except:
        return 0


def calculate_percentile(data, percentile):
    """計算百分位數"""
    if not data:
        return 0
    return np.percentile(data, percentile)


def run_benchmark(model_path, ctx_size):
    """
    對單個模型運行基準測試
    
    Returns:
        包含詳細延遲指標的字典
    """
    proc = start_server(model_path, ctx_size)
    if not proc:
        return None
    
    info = get_model_info()
    if not info:
        kill_server()
        return None
    
    print(f"📊 Model: {info['name']} ({info['params_b']}B) | Size: {info['size_gb']}GB")
    print(f"📈 Running {N_RUNS} requests...")
    
    latencies = []
    throughputs = []
    errors = []
    ram_usages = []
    
    for i in range(N_RUNS):
        current_ram = get_process_ram(proc.pid)
        ram_usages.append(current_ram)
        
        payload = {
            "prompt": PROMPT,
            "n_predict": 128,
            "temperature": 0.7
        }
        
        request_start = time.time()
        
        try:
            response = requests.post(
                f"{URL_BASE}/completion",
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            request_end = time.time()
            
            total_latency_ms = (request_end - request_start) * 1000
            
            data = response.json()
            tokens_output = data.get("tokens_predicted", len(data.get("content", "").split()))
            tokens_prompt = len(PROMPT.split())
            
            # 計算吞吐量（tokens/sec）
            tps = tokens_output / (request_end - request_start) if (request_end - request_start) > 0 else 0
            
            latencies.append(total_latency_ms)
            throughputs.append(tps)
            
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{N_RUNS}] latency: {total_latency_ms:.0f}ms | tps: {tps:.2f} | RAM: {current_ram:.2f}GB")
            
        except requests.exceptions.Timeout:
            latencies.append(REQUEST_TIMEOUT * 1000)
            errors.append('timeout')
            print(f"  [{i+1}/{N_RUNS}] ❌ TIMEOUT")
        except Exception as e:
            latencies.append(0)
            errors.append(str(type(e).__name__))
            print(f"  [{i+1}/{N_RUNS}] ❌ ERROR: {type(e).__name__}")
    
    kill_server()
    
    # 計算統計指標
    if latencies:
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': info['name'],
            'model_size_gb': info['size_gb'],
            'params_b': info['params_b'],
            'ctx_size': ctx_size,
            # 延遲指標 (ms)
            'total_latency_avg_ms': round(statistics.mean(latencies), 2),
            'total_latency_std_ms': round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
            'p50_latency_ms': round(calculate_percentile(latencies, 50), 2),
            'p95_latency_ms': round(calculate_percentile(latencies, 95), 2),
            'p99_latency_ms': round(calculate_percentile(latencies, 99), 2),
            'min_latency_ms': round(min(latencies), 2),
            'max_latency_ms': round(max(latencies), 2),
            # 吞吐量指標 (tokens/sec)
            'avg_tokens_per_sec': round(statistics.mean(throughputs), 2) if throughputs else 0,
            'std_tokens_per_sec': round(statistics.stdev(throughputs), 2) if len(throughputs) > 1 else 0,
            # 資源使用
            'avg_ram_gb': round(statistics.mean(ram_usages), 2),
            'peak_ram_gb': round(max(ram_usages), 2),
            # 錯誤率
            'total_requests': N_RUNS,
            'successful_requests': N_RUNS - len(errors),
            'error_rate': round(len(errors) / N_RUNS * 100, 2),
            'errors': ','.join(set(errors)) if errors else 'none'
        }
        
        return results
    
    return None


def save_results_csv(all_results):
    """保存結果到 CSV"""
    ensure_results_dir()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(RESULTS_DIR, f"llama_server_benchmark_{timestamp}.csv")
    json_file = os.path.join(RESULTS_DIR, f"llama_server_benchmark_{timestamp}.json")
    
    if not all_results:
        print("❌ No results to save")
        return
    
    # Save CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    
    # Save JSON (for detailed analysis)
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to:")
    print(f"   CSV:  {csv_file}")
    print(f"   JSON: {json_file}")


def print_summary(all_results):
    """打印摘要統計"""
    print("\n" + "=" * 70)
    print("📊 Summary Statistics (llama-server)")
    print("=" * 70)
    
    for result in all_results:
        print(f"\n{result['model_name']} (ctx={result['ctx_size']})")
        print(f"  📈 Latency (ms):")
        print(f"     Average: {result['total_latency_avg_ms']:.2f}ms ± {result['total_latency_std_ms']:.2f}ms")
        print(f"     p50:     {result['p50_latency_ms']:.2f}ms")
        print(f"     p95:     {result['p95_latency_ms']:.2f}ms")
        print(f"     p99:     {result['p99_latency_ms']:.2f}ms")
        print(f"  🚀 Throughput (tokens/sec):")
        print(f"     Average: {result['avg_tokens_per_sec']:.2f} t/s ± {result['std_tokens_per_sec']:.2f}")
        print(f"  💾 Memory (GB):")
        print(f"     Average: {result['avg_ram_gb']:.2f}GB | Peak: {result['peak_ram_gb']:.2f}GB")
        print(f"  ✅ Reliability:")
        print(f"     Success: {result['successful_requests']}/{result['total_requests']} ({100 - result['error_rate']:.1f}%)")
        if result['errors'] != 'none':
            print(f"     Errors:  {result['errors']}")


def main():
    """主程序"""
    print("=" * 70)
    print("🚀 llama-server Load Testing & Benchmarking")
    print("=" * 70)
    
    ensure_results_dir()
    kill_server()
    time.sleep(1)
    
    all_results = []
    
    for idx, (model_path, ctx_size) in enumerate(MODELS, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(MODELS)}] Testing {os.path.basename(model_path)}")
        print("=" * 70)
        
        result = run_benchmark(model_path, ctx_size)
        if result:
            all_results.append(result)
        else:
            print(f"⚠️  Skipped this model")
        
        time.sleep(2)  # 冷卻時間
    
    # 保存和打印結果
    if all_results:
        save_results_csv(all_results)
        print_summary(all_results)
    else:
        print("❌ No results collected")
    
    print("\n" + "=" * 70)
    print("✨ Benchmarking completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
