import os
import shlex
import subprocess
import modal

from datetime import datetime

MODAL_APP_NAME = "SGLang"
MODAL_GPU_TYPE = "H100!:1"
MODAL_TIMEOUT = 60 * 60

HUGGINGFACE_ACCESS_TOKEN = modal.Secret.from_name("huggingface-access-token")
VOL_MOUNT_PATH = "/model_cache"

PATCH_LOCAL_DIRS = "/net/home/liaw/COSMOSLab-dLLM/sglang-baseline"
PATCH_REMOTE_DIR = "/sgl-workspace/sglang"

LOG_BASE_NAME = "test_observability_baseline"

TEST_FILE_REMOTE_PATH = "/sgl-workspace/sglang/test/registered/dllm/test_observability.py"

SAVED_SPECIAL_FOLDER = "/sgl-workspace/sglang/sglang_dllm_req_dumps"

# for reproducibility, don't use the lmsysorg/sglang:dev image which is updated regularly, but use the image with a specific sha256 digest
SGLANG_DEV_DOCKER_HUB_IMAGE = "lmsysorg/sglang@sha256:462b58d2363a51603c9c2c2c38201bf144bad799e0bc4722be97c1a530131274" 

app = modal.App(MODAL_APP_NAME)
image = (
    modal.Image.from_registry(SGLANG_DEV_DOCKER_HUB_IMAGE, force_build=False)
    .uv_pip_install("huggingface_hub", "hf_transfer")
    .env(
        {
            "HF_HOME": VOL_MOUNT_PATH,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    )
    .add_local_dir(PATCH_LOCAL_DIRS, PATCH_REMOTE_DIR, copy=False)
)

hf_volume = modal.Volume.from_name("hfcache", create_if_missing=True)
dump_volume = modal.Volume.from_name("sglang-dllm-req-dumps", create_if_missing=True)

@app.function(
    gpu=MODAL_GPU_TYPE,
    image=image,
    timeout=MODAL_TIMEOUT,
    volumes={
        VOL_MOUNT_PATH: hf_volume,
        SAVED_SPECIAL_FOLDER: dump_volume,
    },
    secrets=[HUGGINGFACE_ACCESS_TOKEN],
)
def run_test() -> None:
    os.makedirs(SAVED_SPECIAL_FOLDER, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(SAVED_SPECIAL_FOLDER, f"{LOG_BASE_NAME}_{timestamp}.log")
    
    cmd = (
        "set -o pipefail; "
        f"python -u {shlex.quote(TEST_FILE_REMOTE_PATH)} "
        f"2>&1 | tee {shlex.quote(log_path)}"
    )

    process = subprocess.run(["bash", "-c", cmd], check=False)

    dump_volume.commit()

    if process.returncode != 0:
        raise RuntimeError(
            f"Test failed with return code {process.returncode}. "
            f"Log saved to {log_path}"
        )


@app.local_entrypoint()
def main() -> None:
    run_test.remote()
    print("run_test success")
    return
