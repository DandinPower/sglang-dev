# Forward Level Scheduler vs Block Level Scheduler for dLLM LowConfidence Decoding

This run compares the current `sglang` (forward level scheduler) implementation against `sglang-baseline` (block level scheduler) for dLLM `LowConfidence` decoding on `inclusionAI/LLaDA2.0-mini`. The goal is to understand how confidence threshold and concurrent request count affect forward counts, block completion latency, throughput, and average tokens accepted per forward.

## Results

| Experiment | bs | Throughput tok/s (per req) | Avg tokens/forward | Avg fwd/block | Avg block latency s |
|---|---:|---:|---:|---:|---:|
| sglang-0.7 | 1 | 378.584 | 2.653 | 11.353 | 0.084 |
| sglang-0.7 | 8 | 188.455 | 2.707 | 15.074 | 0.231 |
| sglang-0.7 | 16 | 92.945 | 2.202 | 17.077 | 0.476 |
| sglang-0.95 | 1 | 124.828 | 1.030 | 29.235 | 0.256 |
| sglang-0.95 | 8 | 81.049 | 1.490 | 23.015 | 0.435 |
| sglang-0.95 | 16 | 46.845 | 1.356 | 23.672 | 0.732 |
| sglang-baseline-0.7 | 1 | 502.952 | 2.653 | 11.353 | 0.063 |
| sglang-baseline-0.7 | 8 | 103.809 | 1.161 | 26.170 | 0.312 |
| sglang-baseline-0.7 | 16 | 59.548 | 1.030 | 29.347 | 0.536 |
| sglang-baseline-0.95 | 1 | 228.285 | 1.030 | 29.235 | 0.140 |
| sglang-baseline-0.95 | 8 | 82.627 | 0.940 | 32.311 | 0.387 |
| sglang-baseline-0.95 | 16 | 56.117 | 0.929 | 32.557 | 0.569 |
