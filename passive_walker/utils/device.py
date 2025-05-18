# passive_walker/utils/device.py
import os
import argparse

def add_device_arg(p: argparse.ArgumentParser):
    p.add_argument("--gpu", action="store_true", help="…")

def parse_and_set_device():
    p = argparse.ArgumentParser(add_help=False)
    add_device_arg(p)
    args, _ = p.parse_known_args()
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu" if args.gpu else "cpu")
    return args.gpu
