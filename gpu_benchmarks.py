"""
GPU Vector Search Benchmarks — RTX 4050 Laptop (6GB VRAM)

Questions we're answering:
1. At what DB size does GPU vector search become faster than CPU?
2. What's the throughput difference for batch embedding generation?
3. Can we fit the entire ZeroClaw DB in VRAM? (spoiler: yes, easily)
4. What's the latency difference for lever-runner's actual workload (1 query vs 1K DB)?
"""

import torch
import time
import numpy as np
import hashlib
import sqlite3
import os
import json
from dataclasses import dataclass, asdict

@dataclass
class BenchmarkResult:
    name: str
    db_size: int
    dim: int
    latency_us: float
    throughput_qps: float
    device: str
    batch_size: int
    notes: str = ""

def hash_embed(text: str, dim: int = 64) -> np.ndarray:
    """pincherOS hash embedder — deterministic, zero deps."""
    h = hashlib.blake2b(text.encode(), digest_size=dim).digest()
    v = np.array([b/255.0 for b in h], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)

def run_benchmarks():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    
    print(f"Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    if device == 'cuda':
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # === Benchmark 1: Vector search at different DB sizes ===
    print("=" * 60)
    print("BENCHMARK 1: Vector Search Latency (cosine similarity)")
    print("=" * 60)
    
    dims = [64, 128, 384]  # 64=hash embed, 128=small model, 384=sentence-transformer
    db_sizes = [100, 1000, 10000, 100000, 1000000]
    
    for dim in dims:
        for db_size in db_sizes:
            # Generate random DB
            db_vectors = np.random.randn(db_size, dim).astype(np.float32)
            db_vectors /= np.linalg.norm(db_vectors, axis=1, keepdims=True)
            
            # Query vector
            query = np.random.randn(1, dim).astype(np.float32)
            query /= np.linalg.norm(query)
            
            # CPU benchmark
            N = 100 if db_size < 100000 else 10
            start = time.perf_counter()
            for _ in range(N):
                sims = db_vectors @ query.T
            cpu_latency = (time.perf_counter() - start) / N * 1e6
            
            # GPU benchmark (if fits in VRAM)
            if device == 'cuda':
                try:
                    db_gpu = torch.from_numpy(db_vectors).cuda()
                    q_gpu = torch.from_numpy(query).cuda()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(N):
                        sims_gpu = db_gpu @ q_gpu.T
                    torch.cuda.synchronize()
                    gpu_latency = (time.perf_counter() - start) / N * 1e6
                    
                    speedup = cpu_latency / gpu_latency
                    print(f"  dim={dim} db={db_size:>7,} | CPU={cpu_latency:>8.0f}µs GPU={gpu_latency:>8.0f}µs speedup={speedup:.1f}x")
                    
                    results.append(BenchmarkResult("vector_search", db_size, dim, gpu_latency, N/(gpu_latency/1e6), "GPU", 1))
                except torch.cuda.OutOfMemoryError:
                    print(f"  dim={dim} db={db_size:>7,} | CPU={cpu_latency:>8.0f}µs GPU=OOM")
            
            results.append(BenchmarkResult("vector_search", db_size, dim, cpu_latency, N/(cpu_latency/1e6), "CPU", 1))
    
    # === Benchmark 2: Batch embedding throughput ===
    print()
    print("=" * 60)
    print("BENCHMARK 2: Batch Embedding Throughput")
    print("=" * 60)
    
    dim = 384
    batch_sizes = [1, 8, 32, 128, 512, 1024]
    
    for bs in batch_sizes:
        # Simulate embedding generation (matmul like a small transformer)
        x = torch.randn(bs, dim)
        w = torch.randn(dim, dim)
        
        # CPU
        N = 1000
        start = time.perf_counter()
        for _ in range(N):
            emb = x @ w
        cpu_per_item = (time.perf_counter() - start) / N / bs * 1e6
        
        # GPU
        if device == 'cuda':
            x_gpu = x.cuda()
            w_gpu = w.cuda()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(N):
                emb_gpu = x_gpu @ w_gpu
            torch.cuda.synchronize()
            gpu_per_item = (time.perf_counter() - start) / N / bs * 1e6
            
            throughput = bs / (gpu_per_item / 1e6)
            print(f"  batch={bs:>5} | CPU={cpu_per_item:>6.1f}µs/item GPU={gpu_per_item:>6.1f}µs/item throughput={throughput:>10,.0f} items/s")
            results.append(BenchmarkResult("batch_embed", bs, dim, gpu_per_item, throughput, "GPU", bs))
        else:
            throughput = bs / (cpu_per_item / 1e6)
            print(f"  batch={bs:>5} | CPU={cpu_per_item:>6.1f}µs/item throughput={throughput:>10,.0f} items/s")
        
        results.append(BenchmarkResult("batch_embed", bs, dim, cpu_per_item, throughput, "CPU", bs))
    
    # === Benchmark 3: ZeroClaw actual workload ===
    print()
    print("=" * 60)
    print("BENCHMARK 3: ZeroClaw Actual Workload (hash embed + vector search)")
    print("=" * 60)
    
    # Load actual ZeroClaw data if available
    db_path = "/tmp/zeroclaw-sandbox/zeroclaw-tictactoe/vectors.db"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        vectors = []
        for row in conn.execute("SELECT vector FROM vectors"):
            v = np.array([b/255.0 for b in row[0]], dtype=np.float32)
            v /= (np.linalg.norm(v) + 1e-10)
            vectors.append(v)
        conn.close()
        db_vectors = np.array(vectors)
        actual_size = len(vectors)
        print(f"  Loaded {actual_size} vectors from tic-tac-toe DB")
    else:
        db_vectors = np.random.randn(5000, 64).astype(np.float32)
        db_vectors /= np.linalg.norm(db_vectors, axis=1, keepdims=True)
        actual_size = 5000
        print(f"  Using synthetic data ({actual_size} vectors)")
    
    # Hash embed + search (pincherOS/zeroclaw style)
    N = 1000
    queries = ["XO.XO.X.." + str(i) for i in range(100)]
    
    start = time.perf_counter()
    for q in queries:
        vec = hash_embed(q)
        sims = db_vectors @ vec
        best = np.argmax(sims)
    cpu_latency = (time.perf_counter() - start) / len(queries) * 1e6
    
    # GPU version
    if device == 'cuda':
        db_gpu = torch.from_numpy(db_vectors).cuda()
        start = time.perf_counter()
        for q in queries:
            vec = hash_embed(q)
            q_gpu = torch.from_numpy(vec).cuda()
            sims = db_gpu @ q_gpu
            best = torch.argmax(sims).item()
        gpu_latency = (time.perf_counter() - start) / len(queries) * 1e6
        
        print(f"  CPU: {cpu_latency:.0f}µs per query")
        print(f"  GPU: {gpu_latency:.0f}µs per query (speedup: {cpu_latency/gpu_latency:.1f}x)")
        print(f"  Note: GPU includes CPU→GPU transfer per query (not realistic for batch)")
        results.append(BenchmarkResult("zeroclaw_workload", actual_size, 64, gpu_latency, 1/(gpu_latency/1e6), "GPU", 1, "includes cpu->gpu transfer"))
    
    results.append(BenchmarkResult("zeroclaw_workload", actual_size, 64, cpu_latency, 1/(cpu_latency/1e6), "CPU", 1))
    
    # === Benchmark 4: VRAM capacity analysis ===
    print()
    print("=" * 60)
    print("BENCHMARK 4: VRAM Capacity Analysis")
    print("=" * 60)
    
    if device == 'cuda':
        total_vram = torch.cuda.get_device_properties(0).total_memory
        for dim in [64, 128, 384, 768, 1536]:
            max_vectors = int(total_vram * 0.8 / (dim * 4))  # 80% of VRAM, float32
            max_fp16 = int(total_vram * 0.8 / (dim * 2))     # FP16
            print(f"  dim={dim:>4}: max {max_vectors:>12,} FP32 vectors ({max_vectors/1e6:.1f}M) or {max_fp16:>12,} FP16 ({max_fp16/1e6:.1f}M)")
    else:
        print("  No CUDA device available — skipping VRAM analysis")
    
    # Save results
    results_path = os.path.expanduser("~/repos/zeroclaw-arena/gpu-benchmarks.json")
    with open(results_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    run_benchmarks()
