#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import NamedTuple


REGION = "eu-ro-1"
ENDPOINT = "https://s3api-eu-ro-1.runpod.io"
BUCKET = "levqvpoqax"
PREFIX = "runs/2026-06-09_15-20-54/"
DEST = Path("2026-06-09_15-20-54")


class RemoteFile(NamedTuple):
    key: str
    size: int


def aws_base() -> list[str]:
    return ["aws", "--region", REGION, "--endpoint-url", ENDPOINT]


def normalize_prefix(prefix: str) -> str:
    prefix = prefix.strip()
    if prefix.startswith("s3://"):
        raise ValueError("Pass prefix like 'runs/...' not a full s3:// URL")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return prefix


def run_text(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def ls_prefix(prefix: str) -> tuple[list[str], list[RemoteFile]]:
    s3_url = f"s3://{BUCKET}/{prefix}"
    cmd = aws_base() + ["s3", "ls", s3_url]
    output = run_text(cmd)

    subdirs: list[str] = []
    files: list[RemoteFile] = []

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("PRE "):
            name = line[4:].strip()
            if name:
                subdirs.append(prefix + name)
            continue

        parts = line.split(maxsplit=3)
        if len(parts) == 4:
            filename = parts[3]
            if filename:
                files.append(RemoteFile(key=prefix + filename, size=int(parts[2])))

    return subdirs, files


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def should_skip(local_path: Path, remote_size: int) -> bool:
    return local_path.exists() and local_path.is_file() and local_path.stat().st_size == remote_size


def download_with_cp(src: str, local_path: Path) -> tuple[bool, str]:
    cmd = aws_base() + ["s3", "cp", src, str(local_path)]
    code, stdout, stderr = run_cmd(cmd)
    if code == 0:
        return True, stdout
    return False, stderr or stdout


def download_with_get_object(key: str, local_path: Path) -> tuple[bool, str]:
    cmd = aws_base() + [
        "s3api",
        "get-object",
        "--bucket",
        BUCKET,
        "--key",
        key,
        str(local_path),
    ]
    code, stdout, stderr = run_cmd(cmd)
    if code == 0:
        return True, stdout
    return False, stderr or stdout


def copy_one(remote_file: RemoteFile, root_prefix: str, dest: Path) -> bool:
    key = remote_file.key

    if not key.startswith(root_prefix):
        raise ValueError(f"Key {key!r} does not start with root prefix {root_prefix!r}")

    relative = key[len(root_prefix) :]
    if not relative:
        return True

    local_path = dest / relative
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if should_skip(local_path, remote_file.size):
        print(f"SKIP {local_path} (already downloaded, size matches {remote_file.size})")
        return True

    src = f"s3://{BUCKET}/{key}"
    print(f"COPY {src} -> {local_path}")
    ok, message = download_with_cp(src, local_path)
    if ok:
        return True

    print(f"WARN cp failed for {src}: {message.strip()}")
    print(f"RETRY get-object s3://{BUCKET}/{key} -> {local_path}")
    ok, message = download_with_get_object(key, local_path)
    if ok:
        return True

    print(f"ERROR could not download {src}: {message.strip()}")
    if local_path.exists() and local_path.stat().st_size == 0:
        local_path.unlink()
    return False


def walk_and_copy(root_prefix: str, dest: Path) -> None:
    todo = [root_prefix]
    seen_prefixes: set[str] = set()
    copied = 0
    skipped = 0
    failed: list[str] = []

    while todo:
        prefix = todo.pop(0)
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)

        print(f"LS s3://{BUCKET}/{prefix}")
        subdirs, files = ls_prefix(prefix)
        todo.extend(subdirs)

        for remote_file in files:
            local_rel = remote_file.key[len(root_prefix) :]
            local_path = dest / local_rel
            already_present = should_skip(local_path, remote_file.size)
            ok = copy_one(remote_file, root_prefix, dest)
            if ok and already_present:
                skipped += 1
            elif ok:
                copied += 1
            else:
                failed.append(remote_file.key)

    print(f"Done. Downloaded {copied} files into {dest}")
    print(f"Skipped {skipped} existing files")
    if failed:
        print(f"Failed to download {len(failed)} files:")
        for key in failed:
            print(f"  {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Runpod S3 prefix using only ls + cp")
    parser.add_argument("--bucket", default=BUCKET)
    parser.add_argument("--prefix", default=PREFIX)
    parser.add_argument("--dest", default=str(DEST))
    parser.add_argument("--region", default=REGION)
    parser.add_argument("--endpoint", default=ENDPOINT)
    return parser.parse_args()


def main() -> None:
    global BUCKET, REGION, ENDPOINT

    args = parse_args()
    BUCKET = args.bucket
    REGION = args.region
    ENDPOINT = args.endpoint

    prefix = normalize_prefix(args.prefix)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    walk_and_copy(prefix, dest)


if __name__ == "__main__":
    main()
