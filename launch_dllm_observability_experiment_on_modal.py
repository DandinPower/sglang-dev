import asyncio
import json
import os
import shlex
import subprocess
import threading
import time
import traceback
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import modal


MODAL_APP_NAME = "SGLang"
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

PATCH_CHOICES = ["sglang", "sglang-baseline"]
LOCAL_PATCH_DIR_BY_TARGET = {
    "sglang": "/net/home/liaw/COSMOSLab-dLLM/sglang",
    "sglang-baseline": "/net/home/liaw/COSMOSLab-dLLM/sglang-baseline",
}
REMOTE_PATCH_DIR_BY_TARGET = {
    "sglang": "/sgl-workspace/patches/sglang",
    "sglang-baseline": "/sgl-workspace/patches/sglang-baseline",
}

DLLM_ALGORITHM = "LowConfidence"
LOCAL_DLLM_ALGORITHM_CONFIG_PATH = "algorithm_config.yaml"
REMOTE_DLLM_ALGORITHM_CONFIG_PATH = "/sgl-workspace/algorithm_config.yaml"

MODEL_PATH = "inclusionAI/LLaDA2.0-mini"
REQUEST_COUNTS = [1, 8, 16]
REQUEST_WINDOW_SECONDS = 4.0
OBSERVABILITY_MAX_NEW_TOKENS = 512
OBSERVABILITY_SEED = 42
FORWARD_COUNTS_KEY = "dllm_forward_counts_per_block"
BLOCK_LATENCY_KEY = "dllm_block_completion_latencies"

COUNTRIES = [
    "France",
    "Germany",
    "Japan",
    "Canada",
    "Australia",
    "Brazil",
    "India",
    "Italy",
    "Spain",
    "Egypt",
    "South Korea",
    "Mexico",
    "Argentina",
    "Thailand",
    "the Netherlands",
    "Sweden",
]

PROMPT_TEMPLATE = (
    "Human: What is the capital of {country} and how is that city like. "
    "Give me detail information about that city. "
    "Write in a format of plaintext.\nAssistant:"
)

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
        LOCAL_PATCH_DIR_BY_TARGET["sglang"],
        REMOTE_PATCH_DIR_BY_TARGET["sglang"],
        copy=False,
    )
    .add_local_dir(
        LOCAL_PATCH_DIR_BY_TARGET["sglang-baseline"],
        REMOTE_PATCH_DIR_BY_TARGET["sglang-baseline"],
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


def _build_server_command(max_running_requests: int) -> list[str]:
    assert max_running_requests >= 1, "max_running_requests must be at least 1."

    return [
        "sglang",
        "serve",
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

def _build_server_env(patch_remote_dir: str) -> dict[str, str]:
    env = os.environ.copy()
    env["HF_HOME"] = REMOTE_MODEL_CACHE_DIR
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    pythonpath_parts = [os.path.join(patch_remote_dir, "python")]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def _tee_process_output(stream, log_path: str) -> None:
    with open(log_path, "a", buffering=1) as log_file:
        for line in iter(stream.readline, ""):
            print(line, end="", flush=True)
            log_file.write(line)
    stream.close()


def _launch_server(
    log_path: str,
    patch_remote_dir: str,
    max_running_requests: int,
) -> tuple[subprocess.Popen, threading.Thread, threading.Thread]:
    command = _build_server_command(
        max_running_requests,
    )
    with open(log_path, "a", buffering=1) as log_file:
        log_file.write(f"server_command: {shlex.join(command)}\n")

    print(f"Launching SGLang server: {shlex.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_build_server_env(patch_remote_dir),
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


def _write_json(path: str, data: dict[str, Any]) -> None:
    with open(path, "w") as fout:
        json.dump(data, fout, indent=2)
        fout.write("\n")


def _safe_divide(numerator: int | float | None, denominator: int | float | None):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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
    prompt: str,
) -> dict[str, Any]:
    scheduled_delay = request_index * request_interval_seconds
    await asyncio.sleep(max(0.0, experiment_start_time + scheduled_delay - time.perf_counter()))
    payload = {
        "text": prompt,
        "sampling_params": {
            "sampling_seed": OBSERVABILITY_SEED,
            "temperature": 0.0,
            "max_new_tokens": OBSERVABILITY_MAX_NEW_TOKENS,
            "ignore_eos": True,
        },
        "stream": True,
    }
    result_path = os.path.join(run_dir, f"request_{request_index:02d}.json")
    request_start_time = time.perf_counter()
    request_start_realtime = datetime.now().isoformat(timespec="milliseconds")

    chunks: list[dict[str, Any]] = []
    final_meta_info: dict[str, Any] = {}
    first_new_token_elapsed = None
    block_arrival_elapsed_seconds: list[float] = []
    first_block_output_tokens = None
    previous_output_tokens = 0
    previous_block_count = 0

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

                output_tokens = _output_token_count(chunk)
                if (
                    first_new_token_elapsed is None
                    and output_tokens > previous_output_tokens
                ):
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

        end_time = time.perf_counter()
        e2e_seconds = end_time - request_start_time
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

        first_block_elapsed = (
            block_arrival_elapsed_seconds[0] if block_arrival_elapsed_seconds else None
        )
        tokens_after_first_block = None
        generated_tokens_per_second_after_first_block = None
        if (
            completion_tokens is not None
            and first_block_output_tokens is not None
            and first_block_elapsed is not None
        ):
            tokens_after_first_block = max(
                0, completion_tokens - first_block_output_tokens
            )
            generated_tokens_per_second_after_first_block = _safe_divide(
                tokens_after_first_block,
                e2e_seconds - first_block_elapsed,
            )

        result = {
            "ok": True,
            "request_index": request_index,
            "prompt": prompt,
            "payload": payload,
            "scheduled_delay_seconds": scheduled_delay,
            "request_start_realtime": request_start_realtime,
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
                "end_to_end_tokens_per_second": _safe_divide(
                    total_tokens, e2e_seconds
                ),
                "total_tokens_per_second": _safe_divide(total_tokens, e2e_seconds),
                "completion_tokens_per_second": _safe_divide(
                    completion_tokens, e2e_seconds
                ),
                "tokens_after_first_block": tokens_after_first_block,
                "generated_tokens_per_second_after_first_block": (
                    generated_tokens_per_second_after_first_block
                ),
            },
            "final_meta_info": final_meta_info,
            "chunks": chunks,
        }
    except Exception as exc:
        e2e_seconds = time.perf_counter() - request_start_time
        result = {
            "ok": False,
            "request_index": request_index,
            "prompt": prompt,
            "payload": payload,
            "scheduled_delay_seconds": scheduled_delay,
            "request_start_realtime": request_start_realtime,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "client_metrics": {
                "ttfb_seconds": (
                    block_arrival_elapsed_seconds[0]
                    if block_arrival_elapsed_seconds
                    else None
                ),
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
    request_count: int,
) -> list[dict[str, Any]]:
    import aiohttp
    def _build_prompts(request_count: int) -> list[str]:
        assert request_count >= 1, "request_count must be at least 1."
        assert request_count <= len(COUNTRIES), f"request_count must be at most {len(COUNTRIES)}."
        return [PROMPT_TEMPLATE.format(country=country) for country in COUNTRIES[:request_count]]

    prompts = _build_prompts(request_count)
    request_interval_seconds = REQUEST_WINDOW_SECONDS / request_count
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
    connector = aiohttp.TCPConnector(limit=request_count)
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
                    prompt=prompt,
                )
            )
            for request_index, prompt in enumerate(prompts)
        ]
        return await asyncio.gather(*tasks)


def _write_summary(
    run_dir: str,
    server_log_path: str,
    results: list[dict[str, Any]],
    result_folder: str,
    result_folder_path: str,
    request_count: int,
    patch_target: str,
    patch_local_dir: str,
    patch_remote_dir: str,
    max_running_requests: int,
    server_command: list[str],
) -> dict[str, Any]:
    failed_requests = [result["request_index"] for result in results if not result["ok"]]
    request_interval_seconds = REQUEST_WINDOW_SECONDS / request_count
    summary = {
        "ok": not failed_requests,
        "run_dir": run_dir,
        "result_folder": result_folder,
        "result_folder_path": result_folder_path,
        "server_log_path": server_log_path,
        "base_url": SGLANG_SERVER_BASE_URL,
        "model_path": MODEL_PATH,
        "dllm_algorithm": DLLM_ALGORITHM,
        "request_count": request_count,
        "max_running_requests": max_running_requests,
        "request_window_seconds": REQUEST_WINDOW_SECONDS,
        "request_interval_seconds": request_interval_seconds,
        "patch_target": patch_target,
        "patch_local_dir": patch_local_dir,
        "patch_remote_dir": patch_remote_dir,
        "server_command": shlex.join(server_command),
        "failed_requests": failed_requests,
        "request_files": [
            os.path.join(run_dir, f"request_{result['request_index']:02d}.json")
            for result in results
        ],
    }
    _write_json(os.path.join(run_dir, "summary.json"), summary)
    return summary


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
def run_experiment(patch_target: str) -> None:
    request_counts = REQUEST_COUNTS
    max_running_requests = max(request_counts)
    patch_local_dir = LOCAL_PATCH_DIR_BY_TARGET[patch_target]
    patch_remote_dir = REMOTE_PATCH_DIR_BY_TARGET[patch_target]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(REMOTE_DUMP_DIR, f"session_{patch_target}_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    server_log_path = os.path.join(session_dir, "server.log")
    
    server_command = _build_server_command(
        max_running_requests,
    )
    server_process = None
    
    server_process, stdout_thread, stderr_thread = _launch_server(
        server_log_path,
        patch_remote_dir,
        max_running_requests,
    )
    _wait_for_server_health(
        process=server_process,
        base_url=SGLANG_SERVER_BASE_URL,
        timeout_seconds=SGLANG_STARTUP_TIMEOUT,
    )

    summaries: list[dict[str, Any]] = []
    for request_count in request_counts:
        result_folder = f"{patch_target}_{request_count}"
        result_folder_path = os.path.join(session_dir, result_folder)
        os.makedirs(result_folder_path, exist_ok=True)

        results = asyncio.run(
            _run_async_experiment(SGLANG_SERVER_BASE_URL, result_folder_path, request_count)
        )

        summary = _write_summary(
            run_dir=result_folder_path,
            server_log_path=server_log_path,
            results=results,
            result_folder=result_folder,
            result_folder_path=result_folder_path,
            request_count=request_count,
            patch_target=patch_target,
            patch_local_dir=patch_local_dir,
            patch_remote_dir=patch_remote_dir,
            max_running_requests=max_running_requests,
            server_command=server_command,
        )
        summaries.append(summary)
        print(
            f"Experiment summary written to {result_folder_path}/summary.json",
            flush=True,
        )

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
        "request_count_configurations": request_counts,
        "max_running_requests": max_running_requests,
        "server_command": shlex.join(server_command),
        "runs": [
            {
                "ok": summary["ok"],
                "request_count": summary["request_count"],
                "result_folder": summary["result_folder"],
                "result_folder_path": summary["result_folder_path"],
                "run_dir": summary["run_dir"],
                "summary_path": os.path.join(summary["run_dir"], "summary.json"),
                "failed_requests": summary["failed_requests"],
            }
            for summary in summaries
        ],
    }
    _write_json(session_summary_path, session_summary)
    print(f"Session summary written to {session_summary_path}", flush=True)
    print(json.dumps(session_summary, indent=2))
    dump_volume.commit()


@app.local_entrypoint()
def main() -> None:
    for patch_target in PATCH_CHOICES:
        run_experiment.remote(patch_target=patch_target)
