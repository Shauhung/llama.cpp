#!/usr/bin/env python3
"""
計算模型加載到 RAM 的內存占用
包括：模型權重 + KV 緩存
"""

import os
import sys
from pathlib import Path

# Llama3 8B 參數
LLAMA3_8B_PARAMS = 8_000_000_000  # 8 billion parameters


def get_model_file_size(model_path):
    """獲取 GGUF 文件大小（GB）"""
    try:
        size_bytes = os.path.getsize(model_path)
        size_gb = size_bytes / (1024 ** 3)
        return size_gb
    except FileNotFoundError:
        return None


def estimate_kv_cache_size(ctx_size, hidden_dim=4096, num_layers=32, bytes_per_param=2):
    """
    計算 KV 緩存大小
    
    Args:
        ctx_size: Context 大小
        hidden_dim: 隱藏層維度（Llama3-8B = 4096）
        num_layers: 層數（Llama3-8B = 32）
        bytes_per_param: 每個參數的字節數（fp16 = 2 字節）
    
    Returns:
        KV 緩存大小（GB）
    """
    # KV 緩存 = 2 (K和V) * num_layers * ctx_size * hidden_dim * bytes_per_param
    kv_cache_bytes = 2 * num_layers * ctx_size * hidden_dim * bytes_per_param
    kv_cache_gb = kv_cache_bytes / (1024 ** 3)
    return kv_cache_gb


def analyze_model(model_path, ctx_size=2048):
    """
    分析模型的內存占用
    
    Args:
        model_path: 模型文件路徑
        ctx_size: Context 大小
    """
    print("\n" + "=" * 70)
    print("🧠 Model Memory Analysis")
    print("=" * 70 + "\n")
    
    model_file_size = get_model_file_size(model_path)
    
    if model_file_size is None:
        print(f"❌ Model file not found: {model_path}\n")
        return None
    
    model_name = os.path.basename(model_path)
    
    # 計算 KV 緩存
    kv_cache_size = estimate_kv_cache_size(ctx_size)
    
    # 總內存占用（模型權重 + KV 緩存）
    total_memory = model_file_size + kv_cache_size
    
    # 其他開銷（粗略估計，約 5-10%）
    overhead = total_memory * 0.08
    total_with_overhead = total_memory + overhead
    
    print(f"📦 Model: {model_name}")
    print(f"   Context size: {ctx_size}")
    print()
    print(f"Memory Breakdown:")
    print(f"  • Model weights:      {model_file_size:.2f} GB")
    print(f"  • KV cache (ctx={ctx_size}):  {kv_cache_size:.2f} GB")
    print(f"  • Other overhead:     {overhead:.2f} GB (estimate)")
    print(f"  " + "-" * 40)
    print(f"  💾 Total RAM needed:  {total_with_overhead:.2f} GB")
    print()
    
    # 根據量化方式提示推薦的 context 大小
    print(f"📊 Recommendations for {model_name}:")
    
    # 檢查模型大小來推測量化方式
    if model_file_size < 4:  # Q4
        vram_gpu = 4.0  # GPU VRAM
        vram_cpu = total_with_overhead
        print(f"   Quantization: Q4 (likely)")
        print(f"   • With GPU (ngl 99):  ~{vram_gpu:.1f} GB VRAM + ~1 GB CPU RAM")
        print(f"   • CPU only:           {vram_cpu:.2f} GB RAM")
    elif model_file_size < 5.5:  # Q5
        vram_gpu = 5.0
        vram_cpu = total_with_overhead
        print(f"   Quantization: Q5 (likely)")
        print(f"   • With GPU (ngl 99):  ~{vram_gpu:.1f} GB VRAM + ~1 GB CPU RAM")
        print(f"   • CPU only:           {vram_cpu:.2f} GB RAM")
    else:  # Q6/Q8
        vram_gpu = model_file_size
        vram_cpu = total_with_overhead
        print(f"   Quantization: Q6/Q8 (likely)")
        print(f"   • With GPU (ngl 99):  ~{vram_gpu:.1f} GB VRAM + ~1 GB CPU RAM")
        print(f"   • CPU only:           {vram_cpu:.2f} GB RAM")
    
    print()
    return {
        'model': model_name,
        'model_weights_gb': model_file_size,
        'kv_cache_gb': kv_cache_size,
        'overhead_gb': overhead,
        'total_ram_gb': total_with_overhead
    }


def batch_analyze_models(models_dir, ctx_size=2048):
    """分析目錄中所有的 GGUF 模型"""
    models_path = Path(models_dir)
    
    if not models_path.exists():
        print(f"❌ Directory not found: {models_dir}\n")
        return
    
    gguf_files = sorted(models_path.glob("*.gguf"))
    
    if not gguf_files:
        print(f"⚠️  No GGUF files found in {models_dir}\n")
        return
    
    print(f"\n📂 Found {len(gguf_files)} models in {models_dir}\n")
    
    results = []
    for model_file in gguf_files:
        result = analyze_model(str(model_file), ctx_size)
        if result:
            results.append(result)
    
    # 打印對比表
    if results:
        print("\n" + "=" * 70)
        print("📊 Comparison Table")
        print("=" * 70)
        print(f"{'Model':<35} {'Weights':<12} {'KV Cache':<12} {'Total RAM':<12}")
        print("-" * 70)
        for r in results:
            model_short = r['model'][:32]
            print(f"{model_short:<35} {r['model_weights_gb']:>10.2f}GB  {r['kv_cache_gb']:>10.2f}GB  {r['total_ram_gb']:>10.2f}GB")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 分析指定的模型或目錄
        path = sys.argv[1]
        ctx_size = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
        
        if os.path.isfile(path):
            analyze_model(path, ctx_size)
        elif os.path.isdir(path):
            batch_analyze_models(path, ctx_size)
        else:
            print(f"❌ Path not found: {path}\n")
    else:
        # 默認分析 models/llama_quant 目錄
        default_dir = "./models/llama_quant"
        
        if os.path.exists(default_dir):
            batch_analyze_models(default_dir)
        else:
            print(f"⚠️  Default models directory not found: {default_dir}")
            print(f"\nUsage:")
            print(f"  python calculate_model_memory.py <model_file> [ctx_size]")
            print(f"  python calculate_model_memory.py <models_dir> [ctx_size]")
            print(f"\nExample:")
            print(f"  python calculate_model_memory.py ./models/llama_quant/Llama3-8.0B-Q4_K_M.gguf 2048")
            print(f"  python calculate_model_memory.py ./models/llama_quant 4096\n")
