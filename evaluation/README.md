# Evaluation Benchmarks

This directory contains evaluation benchmarks for Flash-WAM model.

## Benchmarks

### 1. LIBERO

LIBERO benchmark for evaluating robot manipulation policies.

**Files:**
- `client.py` - Main evaluation client
- `launch_client.sh` - Script to launch evaluation client
- `launch_server.sh` - Script to launch inference server

**Usage:**

1. Start the inference server:
```bash
bash evaluation/libero/launch_server.sh
```

2. Run evaluation:
```bash
bash evaluation/libero/launch_client.sh
```

Or run directly:
```bash
python evaluation/libero/client.py \
    --libero-benchmark libero_10 \
    --port 29056 \
    --test-num 50 \
    --task-range 0 10 \
    --out-dir outputs/libero
```

### 2. RobotWin

RobotWin benchmark for evaluating dual-arm robot manipulation policies.

**Files:**
- `eval_polict_client_openpi.py` - Main evaluation client
- `calc_stat.py` - Calculate success statistics
- `geometry.py` - Geometry utilities (Euler/Quaternion conversions)
- `test_render.py` - Sapien render test
- `launch_client.sh` - Script to launch evaluation client
- `launch_server.sh` - Script to launch inference server

**Usage:**

1. Start the inference server:
```bash
bash evaluation/robotwin/launch_server.sh
```

2. Run evaluation:
```bash
bash evaluation/robotwin/launch_client.sh
```

3. Calculate statistics:
```bash
python evaluation/robotwin/calc_stat.py <results_folder>
```

## Server Architecture

Both benchmarks use a client-server architecture:
- **Server**: Runs the model inference (Flash-WAM)
- **Client**: Runs the simulation environment and communicates with the server via WebSocket

The WebSocket communication is handled by `wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy`.
