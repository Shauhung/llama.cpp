#!/usr/bin/env python3
"""
Unified Benchmark Runner
同時運行 llama-bench（純推理）和 llama-server（實際延遲）測試
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

# 配置
RESULTS_DIR = os.getenv('RESULTS_DIR', "./benchmark_results")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_results_dir():
    """確保結果目錄存在"""
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


def print_header(title):
    """打印分隔符"""
    print("\n" + "=" * 70)
    print(f"🚀 {title}")
    print("=" * 70 + "\n")


def run_llama_bench():
    """運行 llama-bench 測試"""
    print_header("Phase 1: llama-bench (Pure Inference Benchmark)")
    
    try:
        result = subprocess.run(
            [sys.executable, "llama_bench_runner.py"],
            cwd=SCRIPT_DIR,
            timeout=3600  # 1 小時超時
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ llama-bench timed out")
        return False
    except Exception as e:
        print(f"❌ Error running llama-bench: {e}")
        return False


def run_server_benchmark():
    """運行 llama-server 負載測試"""
    print_header("Phase 2: llama-server (Real Latency Benchmark)")
    
    try:
        result = subprocess.run(
            [sys.executable, "benchmark_server.py"],
            cwd=SCRIPT_DIR,
            timeout=3600  # 1 小時超時
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ llama-server benchmark timed out")
        return False
    except Exception as e:
        print(f"❌ Error running llama-server benchmark: {e}")
        return False


def generate_comparison_report():
    """生成對比報告"""
    print_header("Phase 3: Generating Comparison Report")
    
    ensure_results_dir()
    
    try:
        # 找最新的 CSV 文件
        bench_files = sorted(Path(RESULTS_DIR).glob("llama_bench_*.csv"), 
                           key=lambda p: p.stat().st_mtime, reverse=True)
        server_files = sorted(Path(RESULTS_DIR).glob("llama_server_benchmark_*.csv"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not bench_files or not server_files:
            print("⚠️  Could not find benchmark results")
            return
        
        bench_df = pd.read_csv(bench_files[0])
        server_df = pd.read_csv(server_files[0])
        
        # 生成對比報告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(RESULTS_DIR, f"benchmark_comparison_{timestamp}.md")
        
        with open(report_file, 'w') as f:
            f.write("# Benchmark Comparison Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # llama-bench 摘要
            f.write("## Pure Inference Benchmarks (llama-bench)\n\n")
            f.write("### Metrics Explanation\n")
            f.write("- **avg_ts**: Average tokens per second (throughput)\n")
            f.write("- **stddev_ts**: Standard deviation of throughput\n")
            f.write("- **total_latency_ms**: Average latency in milliseconds\n")
            f.write("- **p95_latency_ms**: 95th percentile latency\n")
            f.write("- **p99_latency_ms**: 99th percentile latency\n\n")
            
            if 'model_type' in bench_df.columns:
                models_done = set()
                for _, row in bench_df.iterrows():
                    model = row['model_type']
                    if model in models_done:
                        continue
                    models_done.add(model)
                    
                    f.write(f"### {model}\n")
                    model_data = bench_df[bench_df['model_type'] == model]
                    
                    # Generation tests
                    gen = model_data[model_data['n_gen'] > 0]
                    if len(gen) > 0:
                        avg_tps = gen['avg_ts'].mean()
                        avg_latency = gen['total_latency_ms'].mean()
                        p95 = gen['p95_latency_ms'].mean() if 'p95_latency_ms' in gen.columns else 0
                        p99 = gen['p99_latency_ms'].mean() if 'p99_latency_ms' in gen.columns else 0
                        f.write(f"- **Generation** (n_gen=128):\n")
                        f.write(f"  - Throughput: {avg_tps:.2f} t/s\n")
                        f.write(f"  - Latency: {avg_latency:.2f}ms | p95: {p95:.2f}ms | p99: {p99:.2f}ms\n")
                    
                    # Prompt tests
                    prompt = model_data[(int(model_data['n_prompt']) > 0) & (model_data['n_gen'] == 0)]
                    if len(prompt) > 0:
                        avg_tps = prompt['avg_ts'].mean()
                        avg_latency = prompt['total_latency_ms'].mean()
                        f.write(f"- **Prompt Processing** (n_prompt=512):\n")
                        f.write(f"  - Throughput: {avg_tps:.2f} t/s\n")
                        f.write(f"  - Latency: {avg_latency:.2f}ms\n")
                    f.write("\n")
            
            # llama-server 摘要
            f.write("## Real-World Latency Benchmarks (llama-server)\n\n")
            f.write("### Metrics Explanation\n")
            f.write("- **total_latency_avg_ms**: Average end-to-end latency (includes HTTP overhead)\n")
            f.write("- **p95_latency_ms**: 95th percentile latency (actual delays, not estimated)\n")
            f.write("- **p99_latency_ms**: 99th percentile latency\n")
            f.write("- **avg_tokens_per_sec**: Actual throughput with HTTP and server overhead\n")
            f.write("- **error_rate**: % of failed requests (timeouts, errors, etc)\n\n")
            
            if 'model_name' in server_df.columns:
                for _, row in server_df.iterrows():
                    model_name = row['model_name']
                    f.write(f"### {model_name} (ctx={int(row['ctx_size'])})\n")
                    f.write(f"- **Latency**:\n")
                    f.write(f"  - Average: {row['total_latency_avg_ms']:.0f}ms ± {row['total_latency_std_ms']:.0f}ms\n")
                    f.write(f"  - p95: {row['p95_latency_ms']:.0f}ms | p99: {row['p99_latency_ms']:.0f}ms\n")
                    f.write(f"  - Range: {row['min_latency_ms']:.0f}ms ~ {row['max_latency_ms']:.0f}ms\n")
                    f.write(f"- **Throughput**: {row['avg_tokens_per_sec']:.2f} t/s ± {row['std_tokens_per_sec']:.2f}\n")
                    f.write(f"- **Memory**: {row['avg_ram_gb']:.2f}GB avg | {row['peak_ram_gb']:.2f}GB peak\n")
                    success_rate = 100 - row['error_rate']
                    f.write(f"- **Reliability**: {int(row['successful_requests'])}/{int(row['total_requests'])} requests successful ({success_rate:.1f}%)\n")
                    if pd.notna(row['errors']) and row['errors'] != 'none':
                        f.write(f"  - Error types: {row['errors']}\n")
                    f.write("\n")
            
            # 對比分析
            f.write("## Performance Analysis\n\n")
            f.write("### Throughput Comparison (llama-bench vs llama-server)\n")
            if not bench_df.empty and 'model_type' in bench_df.columns and not server_df.empty:
                for model_bench in bench_df['model_type'].unique():
                    bench_gen = bench_df[(bench_df['model_type'] == model_bench) & (bench_df['n_gen'] > 0)]
                    if len(bench_gen) > 0:
                        bench_tps = bench_gen['avg_ts'].mean()
                        
                        # 嘗試找匹配的 server 測試
                        for _, server_row in server_df.iterrows():
                            if model_bench.split()[0] in server_row['model_name']:
                                server_tps = server_row['avg_tokens_per_sec']
                                overhead = (bench_tps - server_tps) / bench_tps * 100
                                f.write(f"- **{model_bench}**:\n")
                                f.write(f"  - Pure inference: {bench_tps:.2f} t/s\n")
                                f.write(f"  - With HTTP: {server_tps:.2f} t/s\n")
                                f.write(f"  - Overhead: {overhead:.1f}%\n")
                                break
            
            # 建議
            f.write("\n## Recommendations\n\n")
            f.write("Based on the benchmark results:\n\n")
            
            # 找最快的模型（llama-bench）
            if not bench_df.empty and 'model_type' in bench_df.columns:
                fastest_gen = bench_df[bench_df['n_gen'] > 0].nlargest(1, 'avg_ts')
                if len(fastest_gen) > 0:
                    model = fastest_gen.iloc[0]['model_type']
                    speed = fastest_gen.iloc[0]['avg_ts']
                    f.write(f"- **Fastest for pure inference**: {model} ({speed:.2f} t/s)\n")
            
            # 找最穩定的模型（最小標準差）
            if not bench_df.empty and 'model_type' in bench_df.columns:
                most_stable = bench_df[bench_df['n_gen'] > 0].nsmallest(1, 'stddev_ts')
                if len(most_stable) > 0:
                    model = most_stable.iloc[0]['model_type']
                    stddev = most_stable.iloc[0]['stddev_ts']
                    f.write(f"- **Most stable**: {model} (stddev: {stddev:.3f})\n")
            
            # 實際延遲建議（llama-server）
            if not server_df.empty:
                best_p95 = server_df.nsmallest(1, 'p95_latency_ms')
                if len(best_p95) > 0:
                    model = best_p95.iloc[0]['model_name']
                    p95 = best_p95.iloc[0]['p95_latency_ms']
                    f.write(f"- **Best real-world p95 latency**: {model} ({p95:.0f}ms)\n")
            
            # 最高可靠性
            if not server_df.empty:
                most_reliable = server_df.nsmallest(1, 'error_rate')
                if len(most_reliable) > 0:
                    model = most_reliable.iloc[0]['model_name']
                    error_rate = most_reliable.iloc[0]['error_rate']
                    f.write(f"- **Most reliable**: {model} ({100 - error_rate:.1f}% success rate)\n")
            
            f.write("\n### Production Recommendation\n")
            f.write("Choose based on your use case:\n")
            f.write("- **Low latency required (p95 < 500ms)**: Look for models with lowest p95 in llama-server results\n")
            f.write("- **Throughput priority**: Use model with highest avg_ts from llama-bench\n")
            f.write("- **Stable performance**: Use model with lowest stddev_ts\n")
            f.write("- **Cost-effective**: Lower quantization (Q4_K_M) usually provides best value\n")
        
        print(f"✅ Comparison report saved to: {report_file}")
        return report_file
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主程序"""
    print("=" * 70)
    print("🔥 Unified Benchmark Runner")
    print("=" * 70)
    print("\nThis will run two complementary benchmarks:")
    print("1. llama-bench: Pure inference performance (batch mode)")
    print("2. llama-server: Real-world latency (HTTP server mode)")
    print("\n" + "=" * 70)
    
    ensure_results_dir()
    
    # Phase 1: llama-bench
    # bench_success = run_llama_bench()
    # time.sleep(2)
    
    # # Phase 2: llama-server
    # server_success = run_server_benchmark()
    # time.sleep(2)
    
    # Phase 3: Generate report
    report_file = generate_comparison_report()
    
    # 最終摘要
    print("\n" + "=" * 70)
    print("📊 Benchmark Summary")
    print("=" * 70)
    # print(f"✅ llama-bench:       {'SUCCESS' if bench_success else 'FAILED'}")
    # print(f"✅ llama-server:      {'SUCCESS' if server_success else 'FAILED'}")
    print(f"✅ Comparison Report: {report_file if report_file else 'FAILED'}")
    print("\n📁 Results directory: " + RESULTS_DIR)
    print("=" * 70)
    
    # return 0 if (bench_success and server_success) else 1


if __name__ == "__main__":
    sys.exit(main())
