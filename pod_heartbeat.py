#!/usr/bin/env python3
"""Upload a pod heartbeat file to Hugging Face.

    python pod_heartbeat.py A
    python pod_heartbeat.py B

Reads HF_REPO, HF_SUBDIR, RUNPOD_POD_ID, and HF_TOKEN from the environment.
Writes {HF_SUBDIR}/pod_startup/{pod_id}-{A|B}.txt on the model repo.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

STAGES = {"A", "B"}
RETRIES = 4


def env(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def upload_with_hub(repo, path, text, token):
    from huggingface_hub import HfApi

    HfApi(token=token).upload_file(
        path_or_fileobj=text.encode("utf-8"),
        path_in_repo=path,
        repo_id=repo,
        repo_type="model",
        commit_message=f"pod heartbeat {path}",
    )


def upload_with_http(repo, path, text, token):
    import base64
    import json
    import urllib.error
    import urllib.request

    url = f"https://huggingface.co/api/models/{repo}/commit/main"
    payload = {
        "summary": f"pod heartbeat {path}",
        "files": [
            {
                "path": path,
                "encoding": "base64",
                "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
            }
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "tdv-lewm-pod-heartbeat",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HF commit {exc.code}: {body}") from exc


def upload(repo, path, text, token):
    try:
        upload_with_hub(repo, path, text, token)
        return
    except ImportError:
        pass
    try:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        upload_with_hub(repo, path, text, token)
        return
    except Exception:
        pass
    upload_with_http(repo, path, text, token)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        raise SystemExit("usage: python pod_heartbeat.py A|B")

    stage = sys.argv[1]
    repo = env("HF_REPO")
    subdir = env("HF_SUBDIR")
    pod_id = env("RUNPOD_POD_ID")
    token = env("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

    if not repo or not subdir:
        print("pod_heartbeat: skip (HF_REPO / HF_SUBDIR not set)")
        return
    if not pod_id:
        print("pod_heartbeat: skip (RUNPOD_POD_ID not set)")
        return
    if not token:
        print("pod_heartbeat: skip (HF_TOKEN not set)")
        return

    rel = f"{subdir.strip('/')}/pod_startup/{pod_id}-{stage}.txt"
    text = (
        f"stage={stage}\n"
        f"pod_id={pod_id}\n"
        f"timestamp={datetime.now(timezone.utc).isoformat()}\n"
    )

    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            upload(repo, rel, text, token)
            print(f"pod_heartbeat: uploaded {repo}/{rel}")
            return
        except Exception as exc:
            last_error = exc
            wait = min(8, 2 ** (attempt - 1))
            print(
                f"pod_heartbeat: attempt {attempt}/{RETRIES} failed: {exc} "
                f"(retry in {wait}s)"
            )
            time.sleep(wait)

    print(f"pod_heartbeat: giving up after {RETRIES} attempts: {last_error}")


if __name__ == "__main__":
    main()
