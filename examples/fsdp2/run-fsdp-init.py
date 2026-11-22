"""Launch the initialization milestone on a private, locally owned Ray cluster."""

import os
from pathlib import Path
import runpy
import sys
import tempfile

import ray

repo = Path(__file__).resolve().parents[2]
slime = Path(os.environ.get("SLIME_SOURCE_DIR", repo / "slime")).absolute()
devices = [value.strip() for value in os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")]
if len(devices) != 4 or len(set(devices)) != 4 or not all(value.isdigit() for value in devices):
    raise ValueError("This initialization example requires four distinct NPU device IDs")
# Keep session logs for troubleshooting; ray.shutdown owns only this cluster.
ray_temp = tempfile.mkdtemp(prefix="slime-fsdp-init-", dir="/tmp")
ray.init(address="local", namespace="slime-fsdpturbo-init", num_cpus=12,
         num_gpus=0, resources={"NPU": len(devices)}, include_dashboard=False,
         _node_ip_address="127.0.0.1", _temp_dir=ray_temp,
         object_store_memory=512 * 1024**2)
try:
    sys.argv[0] = str(slime / "train.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    ray.shutdown()
