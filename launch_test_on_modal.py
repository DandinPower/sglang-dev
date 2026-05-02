import subprocess
import modal

MODAL_APP_NAME = "SGLang"
MODAL_GPU_TYPE = "H100!:1"
MODAL_TIMEOUT = 60 * 60

HUGGINGFACE_ACCESS_TOKEN = modal.Secret.from_name("huggingface-access-token")
VOL_MOUNT_PATH = "/model_cache"

PATCH_LOCAL_DIRS = "/net/home/liaw/COSMOSLab-dLLM/sglang"
PATCH_REMOTE_DIR = "/sgl-workspace/sglang"

TEST_FILE_REMOTE_PATH = "/sgl-workspace/sglang/test/registered/dllm/test_scheduler.py"

SAVED_SPECIAL_FOLDER = "/sgl-workspace/sglang/sglang_dllm_req_dumps"

app = modal.App(MODAL_APP_NAME)
image = (
    modal.Image.from_registry("lmsysorg/sglang:dev", force_build=False)
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
    process = subprocess.run(
        ["python", TEST_FILE_REMOTE_PATH],
        check=False,
    )

    dump_volume.commit()

    if process.returncode != 0:
        raise RuntimeError(f"{process.returncode}")


@app.local_entrypoint()
def main() -> None:
    run_test.remote()
    print("run_test success")
    return
