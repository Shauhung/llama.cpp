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
from collections import defaultdict

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

# --- 0. 控制 server ---
def kill_server():
    """終止所有 llama-server 進程"""
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(2)

def start_server(model_path, ctx_size=2048):
    """啟動 llama-server 並載入模型"""
    print(f"\n🚀 啟動 server 並載入: {os.path.basename(model_path)} (ctx={ctx_size})")
    
    try:
        proc = subprocess.Popen(
            ["./build/bin/llama-server", 
             "-m", model_path,
             "--port", str(SERVER_PORT),
             "-n", "512",
             "-t", "8",
             "-c", str(ctx_size)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
            text=True,
            bufsize=1
        )
        
        print(f"⏳ 等待 server 啟動並檢查...")
        
        # 多次重試檢查 server 是否正常
        max_retries = 60  # 60 秒重試
        retry_count = 0
        initial_wait = SERVER_STARTUP_WAIT
        
        # 初始等待
        time.sleep(initial_wait)
        
        # 檢查進程是否還在運行
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            print(f"❌ server 啟動失敗，進程已終止")
            print(f"\n--- 錯誤信息 (stderr) ---")
            print(stderr[-1000:] if len(stderr) > 1000 else stderr)  # 顯示最後 1000 字
            print(f"--- 完整進程狀態 ---")
            print(f"進程已終止，返回碼: {proc.returncode}")
            return None
        
        # 重試連接
        while retry_count < max_retries:
            try:
                response = requests.get(f"{URL_BASE}/health", timeout=2)
                if response.status_code == 200:
                    print("✅ server 已就緒，連接成功")
                    return proc
            except requests.exceptions.ConnectionError:
                retry_count += 1
                if retry_count % 10 == 0:
                    print(f"   重試中... ({retry_count}/{max_retries}秒)")
                
                # 檢查進程是否還活著
                if proc.poll() is not None:
                    stdout, stderr = proc.communicate()
                    print(f"\n❌ server 進程在連接時已終止")
                    print(f"--- 錯誤信息 (stderr) ---")
                    print(stderr[-1000:] if len(stderr) > 1000 else stderr)
                    return None
                
                time.sleep(1)
            except Exception as e:
                retry_count += 1
                time.sleep(1)
        
        # 連接超時，檢查進程狀態
        if proc.poll() is None:
            print(f"❌ 無法連接到 server (進程在運行但無響應，可能卡在初始化)")
            # 嘗試讀取進程輸出
            try:
                import select
                if select.select([proc.stderr], [], [], 0)[0]:
                    err = proc.stderr.read()
                    if err:
                        print(f"--- 最近的進程輸出 ---")
                        print(err[-500:])
            except:
                pass
            kill_server()
        else:
            print(f"❌ server 進程已終止，返回碼: {proc.returncode}")
        
        return None
    except Exception as e:
        print(f"❌ 啟動失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- 1. 自動獲取模型詳細資訊 ---
def get_model_info():
    try:
        r = requests.get(f"{URL_BASE}/v1/models")
        data = r.json()['data'][0]  # 取得目前加載的模型資料
        meta = data.get('meta', {})
        return {
            "name": data.get('id', "Unknown"),
            "n_ctx": meta.get('n_ctx', 0),
            "size_gb": round(meta.get('size', 0) / (1024**3), 2),
            "params_b": round(meta.get('n_params', 0) / 1e9, 2)
        }
    except Exception as e:
        print(f"無法取得模型資訊，請確認 llama-server 是否已啟動: {e}")
        return None

# --- 2. 獲取 RAM 使用量 (GB) ---
def get_llama_ram(pid):
    try:
        proc = psutil.Process(pid)
        return proc.memory_info().rss / (1024 ** 3)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0

# 開始執行
print("=" * 70)
print("🤖 llama.cpp 多模型自動 Benchmark")
print("=" * 70)

# 清理現有 server
kill_server()
time.sleep(1)

for idx, (model_path, ctx_size) in enumerate(MODELS, 1):
    print(f"\n{'='*70}")
    print(f"📍 [{idx}/{len(MODELS)}] {os.path.basename(model_path)} (ctx={ctx_size})")
    print("="*70)
    
    # 啟動 server
    proc = start_server(model_path, ctx_size)
    if not proc:
        print("⚠️  跳過此模型")
        continue
    
    # 獲取模型資訊
    info = get_model_info()
    if not info:
        kill_server()
        continue

    print(f"🚀 開始測試模型: {info['name']} ({info['params_b']}B)")
    print(f"📏 Context Size: {info['n_ctx']} | Model Size: {info['size_gb']} GB")

    tok_speeds = []
    ram_usages = []

    for i in range(N_RUNS):
        # 紀錄 RAM
        current_ram = get_llama_ram(proc.pid)
        ram_usages.append(current_ram)

        payload = {
            "prompt": PROMPT,
            "n_predict": 128,
            "temperature": 0.7
        }

        start = time.time()
        r = requests.post(f"{URL_BASE}/completion", json=payload)
        end = time.time()

        data = r.json()
        # 優先使用 API 回傳的精確 token 數，若無則粗估
        tokens = data.get("tokens_predicted", len(data.get("content", "").split()))
        
        tok_s = tokens / (end - start)
        tok_speeds.append(tok_s)
        print(f"跑第 {i+1} 次: {tok_s:.2f} tok/s | 目前 RAM: {current_ram:.2f} GB")

    # 計算統計數據
    avg_speed = statistics.mean(tok_speeds)
    std_speed = statistics.stdev(tok_speeds) if len(tok_speeds) > 1 else 0
    avg_ram = statistics.mean(ram_usages)

    # --- 3. 寫入 CSV (Append 模式) ---
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            # 定義更完整的欄位
            writer.writerow([
                'Timestamp', 'Model_Name', 'Params_B', 'Model_Ctx', 'Config_Ctx',
                'File_Size_GB', 'Avg_Tok_s', 'Std_Dev', 'Avg_RAM_GB', 'Runs'
            ])
        
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            info['name'],
            info['params_b'],
            info['n_ctx'],
            ctx_size,  # 本次測試設定的 context 大小
            info['size_gb'],
            f"{avg_speed:.2f}",
            f"{std_speed:.2f}",
            f"{avg_ram:.2f}",
            N_RUNS
        ])

    print(f"\n✅ 模型測試完成！結果已附加至 {CSV_FILE}")
    print(f"平均速度: {avg_speed:.2f} tok/s | 平均 RAM: {avg_ram:.2f} GB")
    
    # 終止 server 準備下一個模型
    kill_server()
    time.sleep(1)

print("\n" + "=" * 70)
print("✨ 所有模型測試完成！")
print("📊 結果已保存到: benchmark_results.csv")
print("=" * 70)