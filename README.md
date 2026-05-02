## Prerequisites

1. Install and authenticate the Modal CLI.
    ```bash
    uv venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    modal setup
    ```

2. Create Modal secret `huggingface-access-token`.
    ```bash
    modal secret create huggingface-access-token HF_TOKEN=<your_hf_token>
    ```

## 1) Run single sglang test file on modal

```bash
modal run launch_test_on_modal.py
```