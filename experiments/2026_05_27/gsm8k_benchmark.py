import asyncio
import json
import os
import re
import shlex
import subprocess
import threading
import time
import traceback
from datetime import datetime
from decimal import Decimal, InvalidOperation
from statistics import mean, median
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import modal


MODAL_APP_NAME = "SGLang-GSM8K"
MODAL_GPU_TYPE = "H100!:1"
MODAL_TIMEOUT = 3600
MODAL_SGLANG_DEV_IMAGE = (
    "lmsysorg/sglang@sha256:"
    "462b58d2363a51603c9c2c2c38201bf144bad799e0bc4722be97c1a530131274"
)
HUGGINGFACE_ACCESS_TOKEN = modal.Secret.from_name("huggingface-access-token")
REMOTE_MODEL_CACHE_DIR = "/hfcache"
REMOTE_DUMP_DIR = "/sgl-workspace/dump"

SGLANG_SERVER_HOST = "127.0.0.1"
SGLANG_SERVER_PORT = 30000
SGLANG_SERVER_BASE_URL = f"http://{SGLANG_SERVER_HOST}:{SGLANG_SERVER_PORT}"
SGLANG_STARTUP_TIMEOUT = 1800

SELECTED_STRATEGY = "sglang"
STRATEGY_LOCAL_DIRS = {
    "sglang": "/net/home/liaw/COSMOSLab-dLLM/sglang",
    "sglang-baseline": "/net/home/liaw/COSMOSLab-dLLM/sglang-baseline",
}
PATCH_REMOTE_DIR = "/sgl-workspace/sglang-baseline"
REMOTE_SGLANG_PYTHON_DIR = os.path.join(PATCH_REMOTE_DIR, "python")

if SELECTED_STRATEGY not in STRATEGY_LOCAL_DIRS:
    raise ValueError(
        f"SELECTED_STRATEGY must be one of {sorted(STRATEGY_LOCAL_DIRS)}. "
        f"Got {SELECTED_STRATEGY!r}."
    )

DLLM_ALGORITHM = "LowConfidence"
LOCAL_DLLM_ALGORITHM_CONFIG_PATH = "algorithm_config.yaml"
REMOTE_DLLM_ALGORITHM_CONFIG_PATH = "/sgl-workspace/algorithm_config.yaml"

MODEL_PATH = "inclusionAI/LLaDA2.0-mini"
CONCURRENCY_COUNTS = [1, 8, 16]
MAX_RUNNING_REQUESTS = 16
REQUEST_INTERVAL_SECONDS = 0.1
OBSERVABILITY_MAX_NEW_TOKENS = 1024
OBSERVABILITY_SEED = 42
IGNORE_EOS = False
FORWARD_COUNTS_KEY = "dllm_forward_counts_per_block"
BLOCK_LATENCY_KEY = "dllm_block_completion_latencies"

GSM8K_TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
GSM8K_TRAIN_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
GSM_FEW_SHOT_COUNT = 5
GSM_PROBLEM_COUNT = 200
GSM_EVAL_START_INDEX = 0
GSM_DATASET_TIMEOUT_SECONDS = 60

app = modal.App(MODAL_APP_NAME)
image = (
    modal.Image.from_registry(MODAL_SGLANG_DEV_IMAGE, force_build=False)
    .uv_pip_install("huggingface_hub", "hf_transfer", "aiohttp")
    .env(
        {
            "HF_HOME": REMOTE_MODEL_CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .add_local_dir(
        STRATEGY_LOCAL_DIRS[SELECTED_STRATEGY],
        PATCH_REMOTE_DIR,
        copy=False,
    )
    .add_local_file(
        LOCAL_DLLM_ALGORITHM_CONFIG_PATH,
        REMOTE_DLLM_ALGORITHM_CONFIG_PATH,
        copy=False,
    )
)
hf_volume = modal.Volume.from_name("hfcache", create_if_missing=True)
dump_volume = modal.Volume.from_name("dump", create_if_missing=True)


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _build_server_command(max_running_requests: int) -> list[str]:
    assert max_running_requests >= 1, "max_running_requests must be at least 1."
    return [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_PATH,
        "--trust-remote-code",
        "--tp-size",
        "1",
        "--mem-fraction-static",
        "0.9",
        "--max-running-requests",
        str(max_running_requests),
        "--attention-backend",
        "flashinfer",
        "--cuda-graph-bs",
        *[str(batch_size) for batch_size in range(1, max_running_requests + 1)],
        "--disable-radix-cache",
        "--dllm-algorithm",
        DLLM_ALGORITHM,
        "--dllm-algorithm-config",
        REMOTE_DLLM_ALGORITHM_CONFIG_PATH,
        "--host",
        SGLANG_SERVER_HOST,
        "--port",
        str(SGLANG_SERVER_PORT),
    ]


def _tee_process_output(stream, log_path: str) -> None:
    with open(log_path, "a", buffering=1) as log_file:
        for line in iter(stream.readline, ""):
            print(line, end="", flush=True)
            log_file.write(line)
    stream.close()


def _launch_server(
    log_path: str,
    max_running_requests: int,
) -> tuple[subprocess.Popen, threading.Thread, threading.Thread]:
    command = _build_server_command(max_running_requests)
    with open(log_path, "a", buffering=1) as log_file:
        log_file.write(f"server_command: {shlex.join(command)}\n")

    print(f"Launching SGLang server: {shlex.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=REMOTE_SGLANG_PYTHON_DIR,
        text=True,
        bufsize=1,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("Failed to capture server process output.")

    stdout_thread = threading.Thread(
        target=_tee_process_output, args=(process.stdout, log_path), daemon=True
    )
    stderr_thread = threading.Thread(
        target=_tee_process_output, args=(process.stderr, log_path), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    return process, stdout_thread, stderr_thread


def _wait_for_server_health(
    process: subprocess.Popen,
    base_url: str,
    timeout_seconds: float,
) -> None:
    start_time = time.perf_counter()
    health_url = f"{base_url}/health_generate"
    while time.perf_counter() - start_time < timeout_seconds:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"SGLang server exited with code {return_code}.")

        try:
            with urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    print("SGLang server is healthy.", flush=True)
                    return
        except (HTTPError, URLError, TimeoutError, OSError):
            pass

        time.sleep(10)

    raise TimeoutError(f"SGLang server did not become healthy within {timeout_seconds} seconds.")


def _write_json(path: str, data: dict[str, Any] | list[Any]) -> None:
    with open(path, "w") as fout:
        json.dump(data, fout, indent=2, ensure_ascii=False)
        fout.write("\n")


def _safe_divide(numerator: int | float | None, denominator: int | float | None):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _load_jsonl_from_url(url: str) -> list[dict[str, Any]]:
    request = Request(url, headers={"User-Agent": "sglang-gsm8k-benchmark/1.0"})
    with urlopen(request, timeout=GSM_DATASET_TIMEOUT_SECONDS) as response:
        text = response.read().decode("utf-8")
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not records:
        raise RuntimeError(f"No records loaded from {url}.")
    return records


def _extract_gsm_gold_answer(answer: str) -> str | None:
    marker = "####"
    if marker in answer:
        return answer.split(marker)[-1].strip()
    return _extract_last_number(answer)


def _extract_last_number(text: str | None) -> str | None:
    if not text:
        return None
    if "####" in text:
        text = text.split("####")[-1]
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def _to_decimal_or_none(value: str | None) -> Decimal | None:
    if value is None:
        return None
    value = value.strip().replace(",", "")
    if not value:
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def _answers_match(predicted: str | None, gold: str | None) -> bool:
    predicted_decimal = _to_decimal_or_none(predicted)
    gold_decimal = _to_decimal_or_none(gold)
    return predicted_decimal is not None and gold_decimal is not None and predicted_decimal == gold_decimal


def _format_few_shot_example(record: dict[str, Any]) -> str:
    return (
        f"Question: {record['question'].strip()}\n"
        f"Answer: {record['answer'].strip()}\n"
    )


def _build_gsm_prompt(question: str, few_shot_records: list[dict[str, Any]]) -> str:
    demonstrations = "\n\n".join(_format_few_shot_example(record) for record in few_shot_records)
    suffix = (
        f"Question: {question.strip()}\n"
        "Answer: Let's think step by step."
    )
    if demonstrations:
        return demonstrations + "\n\n" + suffix
    return suffix


def _prepare_gsm_eval_examples(
    few_shot_count: int,
    problem_count: int,
    eval_start_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if few_shot_count < 0:
        raise ValueError("few_shot_count must be non-negative.")
    if problem_count < 1:
        raise ValueError("problem_count must be at least 1.")
    if eval_start_index < 0:
        raise ValueError("eval_start_index must be non-negative.")

    train_records = _load_jsonl_from_url(GSM8K_TRAIN_URL)
    test_records = _load_jsonl_from_url(GSM8K_TEST_URL)
    if few_shot_count > len(train_records):
        raise ValueError(f"few_shot_count={few_shot_count} exceeds train set size {len(train_records)}.")
    if eval_start_index + problem_count > len(test_records):
        raise ValueError(
            f"eval_start_index + problem_count = {eval_start_index + problem_count} "
            f"exceeds test set size {len(test_records)}."
        )

    few_shot_records = train_records[:few_shot_count]
    selected_test_records = test_records[eval_start_index : eval_start_index + problem_count]
    examples: list[dict[str, Any]] = []
    for offset, record in enumerate(selected_test_records):
        eval_index = eval_start_index + offset
        gold_answer = _extract_gsm_gold_answer(record["answer"])
        prompt = _build_gsm_prompt(record["question"], few_shot_records)
        examples.append(
            {
                "eval_index": eval_index,
                "question": record["question"],
                "gold_solution": record["answer"],
                "gold_answer": gold_answer,
                "prompt": prompt,
            }
        )

    dataset_info = {
        "benchmark": "GSM8K",
        "test_url": GSM8K_TEST_URL,
        "train_url_for_few_shot": GSM8K_TRAIN_URL,
        "train_record_count": len(train_records),
        "test_record_count": len(test_records),
        "few_shot_count": few_shot_count,
        "problem_count": problem_count,
        "eval_start_index": eval_start_index,
        "few_shot_source": "train[:few_shot_count]",
        "eval_source": "test[eval_start_index:eval_start_index+problem_count]",
    }
    return examples, dataset_info


def _output_token_count(chunk: dict[str, Any]) -> int:
    output_ids = chunk.get("output_ids")
    if isinstance(output_ids, list):
        return len(output_ids)

    meta_info = chunk.get("meta_info")
    if isinstance(meta_info, dict):
        completion_tokens = meta_info.get("completion_tokens")
        if isinstance(completion_tokens, int):
            return completion_tokens

    return 0


def _chunk_record(
    chunk_index: int,
    elapsed_seconds: float,
    chunk: dict[str, Any],
    previous_output_tokens: int,
    previous_block_count: int,
) -> dict[str, Any]:
    meta_info = chunk.get("meta_info") or {}
    output_tokens = _output_token_count(chunk)
    block_latencies = _as_list(meta_info.get(BLOCK_LATENCY_KEY))
    forward_counts = _as_list(meta_info.get(FORWARD_COUNTS_KEY))

    return {
        "chunk_index": chunk_index,
        "elapsed_seconds": elapsed_seconds,
        "output_token_count": output_tokens,
        "new_output_tokens": max(0, output_tokens - previous_output_tokens),
        "block_metric_count": len(block_latencies),
        "new_block_latencies": block_latencies[previous_block_count:],
        "forward_count_metric_count": len(forward_counts),
        "text": chunk.get("text"),
        "meta_info": meta_info,
    }


async def _iter_sse_data(response):
    buffer = ""
    async for raw_chunk in response.content.iter_any():
        if not raw_chunk:
            continue
        buffer += raw_chunk.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line.startswith("data:"):
                yield line[len("data:") :].strip()

    tail = buffer.strip()
    if tail.startswith("data:"):
        yield tail[len("data:") :].strip()


async def _send_one_request(
    session,
    base_url: str,
    run_dir: str,
    experiment_start_time: float,
    request_interval_seconds: float,
    request_index: int,
    example: dict[str, Any],
) -> dict[str, Any]:
    prompt = example["prompt"]
    scheduled_delay = request_index * request_interval_seconds
    await asyncio.sleep(max(0.0, experiment_start_time + scheduled_delay - time.perf_counter()))
    payload = {
        "text": prompt,
        "sampling_params": {
            "sampling_seed": OBSERVABILITY_SEED,
            "temperature": 0.0,
            "max_new_tokens": OBSERVABILITY_MAX_NEW_TOKENS,
            "ignore_eos": IGNORE_EOS,
        },
        "stream": True,
    }
    result_path = os.path.join(run_dir, f"request_{request_index:05d}.json")
    request_start_time = time.perf_counter()
    request_start_realtime = datetime.now().isoformat(timespec="milliseconds")

    chunks: list[dict[str, Any]] = []
    final_meta_info: dict[str, Any] = {}
    first_new_token_elapsed = None
    block_arrival_elapsed_seconds: list[float] = []
    first_block_output_tokens = None
    previous_output_tokens = 0
    previous_block_count = 0
    final_text = ""

    try:
        async with session.post(f"{base_url}/generate", json=payload) as response:
            if response.status != 200:
                response_text = await response.text()
                raise RuntimeError(
                    f"/generate returned HTTP {response.status}: {response_text}"
                )

            async for data in _iter_sse_data(response):
                now = time.perf_counter()
                elapsed_seconds = now - request_start_time
                if not data:
                    continue
                if data == "[DONE]":
                    break

                chunk = json.loads(data)
                if "error" in chunk:
                    raise RuntimeError(f"SGLang streaming error: {chunk['error']}")

                meta_info = chunk.get("meta_info")
                if isinstance(meta_info, dict):
                    final_meta_info = meta_info
                else:
                    meta_info = {}

                if isinstance(chunk.get("text"), str):
                    final_text = chunk["text"]

                output_tokens = _output_token_count(chunk)
                if first_new_token_elapsed is None and output_tokens > previous_output_tokens:
                    first_new_token_elapsed = elapsed_seconds

                block_latencies = _as_list(meta_info.get(BLOCK_LATENCY_KEY))
                if len(block_latencies) > previous_block_count and output_tokens > 0:
                    for _ in range(len(block_latencies) - previous_block_count):
                        block_arrival_elapsed_seconds.append(elapsed_seconds)
                    if first_block_output_tokens is None:
                        first_block_output_tokens = output_tokens

                chunks.append(
                    _chunk_record(
                        chunk_index=len(chunks),
                        elapsed_seconds=elapsed_seconds,
                        chunk=chunk,
                        previous_output_tokens=previous_output_tokens,
                        previous_block_count=previous_block_count,
                    )
                )

                previous_output_tokens = max(previous_output_tokens, output_tokens)
                previous_block_count = max(previous_block_count, len(block_latencies))

        e2e_seconds = time.perf_counter() - request_start_time
        forward_counts = _as_list(final_meta_info.get(FORWARD_COUNTS_KEY))
        block_latencies = _as_list(final_meta_info.get(BLOCK_LATENCY_KEY))
        scheduler_ttfb_seconds = block_latencies[0] if block_latencies else None
        scheduler_tbb_seconds = block_latencies[1:]
        client_tbb_seconds = [
            curr - prev
            for prev, curr in zip(
                block_arrival_elapsed_seconds, block_arrival_elapsed_seconds[1:]
            )
        ]

        prompt_tokens = final_meta_info.get("prompt_tokens")
        completion_tokens = final_meta_info.get("completion_tokens")
        if not isinstance(prompt_tokens, int):
            prompt_tokens = None
        if not isinstance(completion_tokens, int):
            completion_tokens = previous_output_tokens
        total_tokens = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )

        first_block_elapsed = block_arrival_elapsed_seconds[0] if block_arrival_elapsed_seconds else None
        tokens_after_first_block = None
        generated_tokens_per_second_after_first_block = None
        if completion_tokens is not None and first_block_output_tokens is not None and first_block_elapsed is not None:
            tokens_after_first_block = max(0, completion_tokens - first_block_output_tokens)
            generated_tokens_per_second_after_first_block = _safe_divide(
                tokens_after_first_block,
                e2e_seconds - first_block_elapsed,
            )

        predicted_answer = _extract_last_number(final_text)
        correct = _answers_match(predicted_answer, example["gold_answer"])

        result = {
            "ok": True,
            "request_index": request_index,
            "eval_index": example["eval_index"],
            "prompt": prompt,
            "payload": payload,
            "scheduled_delay_seconds": scheduled_delay,
            "request_start_realtime": request_start_realtime,
            "benchmark": {
                "question": example["question"],
                "gold_solution": example["gold_solution"],
                "gold_answer": example["gold_answer"],
                "model_text": final_text,
                "predicted_answer": predicted_answer,
                "correct": correct,
            },
            "scheduler_metrics": {
                "forward_counts": forward_counts,
                "ttfb_seconds": scheduler_ttfb_seconds,
                "tbb_seconds": scheduler_tbb_seconds,
                BLOCK_LATENCY_KEY: block_latencies,
                FORWARD_COUNTS_KEY: forward_counts,
            },
            "client_metrics": {
                "ttfb_seconds": first_block_elapsed,
                "first_new_token_ttfb_seconds": first_new_token_elapsed,
                "block_ttfb_seconds": first_block_elapsed,
                "tbb_seconds": client_tbb_seconds,
                "block_arrival_elapsed_seconds": block_arrival_elapsed_seconds,
                "e2e_seconds": e2e_seconds,
            },
            "token_metrics": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "end_to_end_tokens_per_second": _safe_divide(total_tokens, e2e_seconds),
                "total_tokens_per_second": _safe_divide(total_tokens, e2e_seconds),
                "completion_tokens_per_second": _safe_divide(completion_tokens, e2e_seconds),
                "tokens_after_first_block": tokens_after_first_block,
                "generated_tokens_per_second_after_first_block": generated_tokens_per_second_after_first_block,
            },
            "final_meta_info": final_meta_info,
            "chunks": chunks,
        }
    except Exception as exc:
        e2e_seconds = time.perf_counter() - request_start_time
        result = {
            "ok": False,
            "request_index": request_index,
            "eval_index": example.get("eval_index"),
            "prompt": prompt,
            "payload": payload,
            "scheduled_delay_seconds": scheduled_delay,
            "request_start_realtime": request_start_realtime,
            "benchmark": {
                "question": example.get("question"),
                "gold_solution": example.get("gold_solution"),
                "gold_answer": example.get("gold_answer"),
                "model_text": final_text,
                "predicted_answer": _extract_last_number(final_text),
                "correct": False,
            },
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "client_metrics": {
                "ttfb_seconds": block_arrival_elapsed_seconds[0] if block_arrival_elapsed_seconds else None,
                "first_new_token_ttfb_seconds": first_new_token_elapsed,
                "block_arrival_elapsed_seconds": block_arrival_elapsed_seconds,
                "e2e_seconds": e2e_seconds,
            },
            "final_meta_info": final_meta_info,
            "chunks": chunks,
        }

    _write_json(result_path, result)
    return result


async def _run_async_experiment(
    base_url: str,
    run_dir: str,
    concurrency_count: int,
    eval_examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import aiohttp

    assert concurrency_count >= 1, "concurrency_count must be at least 1."
    assert eval_examples, "eval_examples must not be empty."

    request_interval_seconds = REQUEST_INTERVAL_SECONDS
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
    connector = aiohttp.TCPConnector(limit=concurrency_count)
    experiment_start_time = time.perf_counter()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            asyncio.create_task(
                _send_one_request(
                    session=session,
                    base_url=base_url,
                    run_dir=run_dir,
                    experiment_start_time=experiment_start_time,
                    request_interval_seconds=request_interval_seconds,
                    request_index=request_index,
                    example=example,
                )
            )
            for request_index, example in enumerate(eval_examples)
        ]
        return await asyncio.gather(*tasks)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def _mean_numeric(values: list[Any]) -> float | None:
    numeric = [value for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    return float(mean(numeric))


def _write_summary(
    run_dir: str,
    server_log_path: str,
    results: list[dict[str, Any]],
    result_folder: str,
    result_folder_path: str,
    concurrency_count: int,
    patch_target: str,
    patch_local_dir: str,
    patch_remote_dir: str,
    max_running_requests: int,
    server_command: list[str],
    dataset_info: dict[str, Any],
) -> dict[str, Any]:
    failed_requests = [result["request_index"] for result in results if not result["ok"]]
    scored_results = [result for result in results if result.get("ok")]
    correct_count = sum(1 for result in scored_results if result.get("benchmark", {}).get("correct"))
    scored_count = len(scored_results)
    e2e_seconds = [
        result.get("client_metrics", {}).get("e2e_seconds")
        for result in scored_results
        if isinstance(result.get("client_metrics", {}).get("e2e_seconds"), (int, float))
    ]
    ttfb_seconds = [
        result.get("client_metrics", {}).get("ttfb_seconds")
        for result in scored_results
        if isinstance(result.get("client_metrics", {}).get("ttfb_seconds"), (int, float))
    ]
    completion_tps = [
        result.get("token_metrics", {}).get("completion_tokens_per_second")
        for result in scored_results
        if isinstance(result.get("token_metrics", {}).get("completion_tokens_per_second"), (int, float))
    ]
    total_completion_tokens = sum(
        result.get("token_metrics", {}).get("completion_tokens", 0)
        for result in scored_results
        if isinstance(result.get("token_metrics", {}).get("completion_tokens"), int)
    )
    wall_time_seconds = max(e2e_seconds) if e2e_seconds else None

    summary = {
        "ok": not failed_requests,
        "run_dir": run_dir,
        "result_folder": result_folder,
        "result_folder_path": result_folder_path,
        "server_log_path": server_log_path,
        "base_url": SGLANG_SERVER_BASE_URL,
        "model_path": MODEL_PATH,
        "dllm_algorithm": DLLM_ALGORITHM,
        "concurrency_count": concurrency_count,
        "max_running_requests": max_running_requests,
        "request_interval_seconds": REQUEST_INTERVAL_SECONDS,
        "dataset": dataset_info,
        "score_metrics": {
            "correct_count": correct_count,
            "scored_count": scored_count,
            "problem_count": len(results),
            "accuracy": _safe_divide(correct_count, scored_count),
            "failed_request_count": len(failed_requests),
        },
        "latency_metrics": {
            "mean_e2e_seconds": _mean_numeric(e2e_seconds),
            "median_e2e_seconds": median(e2e_seconds) if e2e_seconds else None,
            "p95_e2e_seconds": _percentile(e2e_seconds, 95),
            "mean_ttfb_seconds": _mean_numeric(ttfb_seconds),
            "p95_ttfb_seconds": _percentile(ttfb_seconds, 95),
        },
        "throughput_metrics": {
            "mean_completion_tokens_per_second_per_request": _mean_numeric(completion_tps),
            "total_completion_tokens": total_completion_tokens,
            "estimated_completion_tokens_per_second_over_run": _safe_divide(
                total_completion_tokens, wall_time_seconds
            ),
        },
        "patch_target": patch_target,
        "patch_local_dir": patch_local_dir,
        "patch_remote_dir": patch_remote_dir,
        "server_command": shlex.join(server_command),
        "failed_requests": failed_requests,
        "request_files": [
            os.path.join(run_dir, f"request_{result['request_index']:05d}.json")
            for result in results
        ],
    }
    _write_json(os.path.join(run_dir, "summary.json"), summary)
    _write_json(
        os.path.join(run_dir, "predictions.jsonl.tmp"),
        [
            {
                "request_index": result.get("request_index"),
                "eval_index": result.get("eval_index"),
                "ok": result.get("ok"),
                "correct": result.get("benchmark", {}).get("correct"),
                "gold_answer": result.get("benchmark", {}).get("gold_answer"),
                "predicted_answer": result.get("benchmark", {}).get("predicted_answer"),
                "question": result.get("benchmark", {}).get("question"),
                "model_text": result.get("benchmark", {}).get("model_text"),
            }
            for result in results
        ],
    )
    os.replace(
        os.path.join(run_dir, "predictions.jsonl.tmp"),
        os.path.join(run_dir, "predictions.json"),
    )
    return summary


def _parse_concurrency_counts(csv: str) -> list[int]:
    values = [int(item.strip()) for item in csv.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one concurrency count is required.")
    if any(value < 1 for value in values):
        raise ValueError(f"All concurrency counts must be >= 1. Got {values}.")
    return values


@app.function(
    gpu=MODAL_GPU_TYPE,
    image=image,
    timeout=MODAL_TIMEOUT,
    volumes={
        REMOTE_MODEL_CACHE_DIR: hf_volume,
        REMOTE_DUMP_DIR: dump_volume,
    },
    secrets=[HUGGINGFACE_ACCESS_TOKEN],
)
def run_experiment() -> None:
    few_shot_count = GSM_FEW_SHOT_COUNT
    problem_count = GSM_PROBLEM_COUNT
    eval_start_index = GSM_EVAL_START_INDEX
    concurrency_counts_csv: str = ",".join(str(value) for value in CONCURRENCY_COUNTS)

    concurrency_counts = _parse_concurrency_counts(concurrency_counts_csv)
    max_running_requests = max(MAX_RUNNING_REQUESTS, max(concurrency_counts))
    patch_target = SELECTED_STRATEGY
    patch_local_dir = STRATEGY_LOCAL_DIRS[SELECTED_STRATEGY]
    patch_remote_dir = PATCH_REMOTE_DIR

    eval_examples, dataset_info = _prepare_gsm_eval_examples(
        few_shot_count=few_shot_count,
        problem_count=problem_count,
        eval_start_index=eval_start_index,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(REMOTE_DUMP_DIR, f"session_{patch_target}_gsm8k_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    server_log_path = os.path.join(session_dir, "server.log")
    _write_json(os.path.join(session_dir, "dataset_info.json"), dataset_info)

    server_command = _build_server_command(max_running_requests)
    server_process = None
    summaries: list[dict[str, Any]] = []

    try:
        server_process, stdout_thread, stderr_thread = _launch_server(
            server_log_path,
            max_running_requests,
        )
        _wait_for_server_health(
            process=server_process,
            base_url=SGLANG_SERVER_BASE_URL,
            timeout_seconds=SGLANG_STARTUP_TIMEOUT,
        )

        for concurrency_count in concurrency_counts:
            result_folder = f"{patch_target}_gsm8k_c{concurrency_count}_n{problem_count}_fs{few_shot_count}"
            result_folder_path = os.path.join(session_dir, result_folder)
            os.makedirs(result_folder_path, exist_ok=True)

            results = asyncio.run(
                _run_async_experiment(
                    SGLANG_SERVER_BASE_URL,
                    result_folder_path,
                    concurrency_count,
                    eval_examples,
                )
            )

            summary = _write_summary(
                run_dir=result_folder_path,
                server_log_path=server_log_path,
                results=results,
                result_folder=result_folder,
                result_folder_path=result_folder_path,
                concurrency_count=concurrency_count,
                patch_target=patch_target,
                patch_local_dir=patch_local_dir,
                patch_remote_dir=patch_remote_dir,
                max_running_requests=max_running_requests,
                server_command=server_command,
                dataset_info=dataset_info,
            )
            summaries.append(summary)
            print(f"Experiment summary written to {result_folder_path}/summary.json", flush=True)
    finally:
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server_process.kill()

    session_summary_path = os.path.join(session_dir, "session_summary.json")
    session_summary = {
        "ok": all(summary["ok"] for summary in summaries),
        "session_dir": session_dir,
        "session_summary_path": session_summary_path,
        "server_log_path": server_log_path,
        "base_url": SGLANG_SERVER_BASE_URL,
        "model_path": MODEL_PATH,
        "dllm_algorithm": DLLM_ALGORITHM,
        "patch_target": patch_target,
        "patch_local_dir": patch_local_dir,
        "patch_remote_dir": patch_remote_dir,
        "dataset": dataset_info,
        "concurrency_count_configurations": concurrency_counts,
        "max_running_requests": max_running_requests,
        "server_command": shlex.join(server_command),
        "runs": [
            {
                "ok": summary["ok"],
                "concurrency_count": summary["concurrency_count"],
                "result_folder": summary["result_folder"],
                "result_folder_path": summary["result_folder_path"],
                "run_dir": summary["run_dir"],
                "summary_path": os.path.join(summary["run_dir"], "summary.json"),
                "predictions_path": os.path.join(summary["run_dir"], "predictions.json"),
                "score_metrics": summary["score_metrics"],
                "latency_metrics": summary["latency_metrics"],
                "throughput_metrics": summary["throughput_metrics"],
                "failed_requests": summary["failed_requests"],
            }
            for summary in summaries
        ],
    }
    _write_json(session_summary_path, session_summary)
    print(f"Session summary written to {session_summary_path}", flush=True)
    print(json.dumps(session_summary, indent=2, ensure_ascii=False))
    dump_volume.commit()


@app.local_entrypoint()
def main() -> None:
    run_experiment.remote()
