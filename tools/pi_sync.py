#!/usr/bin/env python3
"""Sync selected USV source files to the Orange Pi and optionally run a command."""

from __future__ import annotations

import argparse
from pathlib import Path

import paramiko


DEFAULT_FILES = (
    "include/usv/SlamExecutionLayer.h",
    "src/SlamExecutionLayer.cc",
    "src/MainProcessor.cc",
    "usv_test_client.py",
    "client_test.py",
    "tests/SelD435iSmokeTest.cc",
    "tests/run_d435i_smoke_test.sh",
    "tools/D435iImuProbe.cc",
)


def put_file(sftp: paramiko.SFTPClient, local_root: Path, remote_root: str, rel: str) -> None:
    local = local_root / rel
    if not local.exists():
        return
    remote = f"{remote_root}/{rel.replace('\\', '/')}"
    parent = remote.rsplit("/", 1)[0]
    sftp.mkdir(parent) if False else None
    parts = parent.strip("/").split("/")
    current = ""
    for part in parts:
        current += "/" + part
        try:
            sftp.mkdir(current)
        except OSError:
            pass
    sftp.put(str(local), remote)
    print(f"uploaded {rel}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.31.31")
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default="orangepi")
    parser.add_argument("--remote-root", default="/root/usv_runtime/usv_better")
    parser.add_argument("--cmd", default="")
    args = parser.parse_args()

    local_root = Path(__file__).resolve().parents[1]
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(args.host, username=args.user, password=args.password, timeout=10)
    try:
        sftp = ssh.open_sftp()
        try:
            for rel in DEFAULT_FILES:
                put_file(sftp, local_root, args.remote_root, rel)
        finally:
            sftp.close()

        if args.cmd:
            stdin, stdout, stderr = ssh.exec_command(args.cmd)
            del stdin
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            if out:
                print(out, end="")
            if err:
                print(err, end="")
            return stdout.channel.recv_exit_status()
        return 0
    finally:
        ssh.close()


if __name__ == "__main__":
    raise SystemExit(main())
