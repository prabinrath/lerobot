# Franka FR3 Policy Deployment

This directory contains scripts for deploying your trained LeRobot policies on the Franka FR3 robot using asynchronous inference.

## Supported Policies

### 🤖 **Policy Types**
- **ACT** (Action Chunking Transformer)
- **Diffusion** (Diffusion Policy)
- **VQ-BeT** (Vector Quantized Behavior Transformer)
- **SmolVLA** (Small Vision Language Action model)
- **GROOT** (Generative Robotic Object Manipulation)
- **PI0/PI05** (Imitation Learning policies)

## Overview

The deployment uses a client-server architecture:
- **Policy Server** (`deploy_policy_server.py`): Loads your trained model and serves inference requests
- **Robot Client** (`deploy_robot_client.py`): Connects to your Franka FR3 robot and communicates with the policy server

This async approach eliminates "wait-for-inference" delays, allowing the robot to continue executing actions while the next action chunk is being computed.

## Prerequisites

1. **Install LeRobot with async dependencies**:
   ```bash
   pip install -e ".[async]"
   ```

2. **Trained Policy**: You should have a trained checkpoint at:
   ```
   outputs/train/your_policy_name/checkpoints/last/
   ```

3. **Robot Implementation**: Complete your robot class implementation:
   ```
   src/lerobot/robots/your_robot/your_robot.py
   ```

4. **Camera Setup**: Configure cameras that match your training data
   - Identify camera indices with: `python -m lerobot.find_cameras`

## Quick Start

### Step 1: Start the Policy Server

In terminal 1:
```bash
python deploy_policy_server.py --host 127.0.0.1 --port 8080 --fps 10
```

### Step 2: Start the Robot Client

In terminal 2:
```bash
python deploy_robot_client.py \
    --server_address 127.0.0.1:8080 \
    --robot_ip 172.16.0.2 \
    --robot_id franka_fr3 \
    --policy_type act \
    --checkpoint_path outputs/train/act_franka_fr3_softtoy/checkpoints/last \
    --task "pick up the soft toy and place it in the drawer" \
    --cameras "{'camera_wrist': {'type': 'opencv', 'index_or_path': 0, 'width': 480, 'height': 640, 'fps': 30}, 'camera_top': {'type': 'opencv', 'index_or_path': 1, 'width': 480, 'height': 640, 'fps': 30}}" \
    --actions_per_chunk 10 \
    --chunk_size_threshold 0.5 \
    --fps 10
```

## Frequency Control and Data Collection Matching

**Critical**: The deployment frequency must match your training data collection frequency!

### How Frequency Control Works

The async inference system controls timing through multiple layers:

1. **Control Loop Frequency (`fps`)**: Sets the base execution rate
   - `environment_dt = 1/fps` seconds between control loop cycles
   - Actions are executed and observations captured at this rate

2. **Inference Request Frequency**: Controlled by `chunk_size_threshold`
   - Observations sent to policy server when action queue drops below threshold
   - Decoupled from control loop frequency for efficiency

3. **Action Chunking**: Policy generates multiple actions per inference
   - `actions_per_chunk` actions generated per server request
   - Actions consumed over time while next chunk is computed

### For 10Hz Training Data

Since your ACT policy was trained on data collected at 10Hz:
- Set `--fps 10` in both server and client
- This ensures robot moves at same speed as training
- `environment_dt = 0.1` seconds between actions

```bash
# Correct configuration for 10Hz training data
python deploy_act_policy_server.py --fps 10
python deploy_act_robot_client.py --fps 10 [other options]
```

## Configuration Parameters

### Policy Server Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--host` | 127.0.0.1 | Server host address |
| `--port` | 8080 | Server port |
| `--fps` | 10 | Control frequency (match training data!) |

### Robot Client Options

#### Required Parameters
- `--policy_type`: Type of policy (act, diffusion, vqbet, smolvla, groot, pi0, pi05)
- `--checkpoint_path`: Path to your trained model
- `--cameras`: Camera configuration (see format below)

#### Franka FR3 Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--robot_ip` | 172.16.0.2 | Franka FR3 IP address |
| `--robot_id` | franka_fr3 | Robot identifier |
| `--max_relative_target` | None | Max joint movement per step (safety) |

#### Policy Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--policy_device` | cuda | Inference device (cuda/cpu/mps) |
| `--task` | "" | Task description |
| `--actions_per_chunk` | 10 | Actions per inference chunk |
| `--chunk_size_threshold` | 0.5 | When to request new actions (0-1) |

#### Performance Tuning
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fps` | 10 | Control frequency (must match training data!) |
| `--aggregate_fn_name` | weighted_average | Action aggregation method |

### Camera FPS vs Control FPS

**Important distinction**:
- **Control FPS** (`--fps 10`): How often the robot executes actions (must match training)
- **Camera FPS** (in camera config): How fast cameras capture (can be higher for smoother video)

```bash
# Control at 10Hz (matching training), but cameras can capture at 30Hz
--fps 10 \
--cameras "{'camera_wrist': {'type': 'opencv', 'index_or_path': 0, 'width': 480, 'height': 640, 'fps': 30}}"
```

The system will automatically downsample camera feeds to match the control frequency.

## Camera Configuration Format

The `--cameras` parameter expects a Python dictionary string:

```python
"{'camera_name': {'type': 'opencv', 'index_or_path': INDEX, 'width': WIDTH, 'height': HEIGHT, 'fps': FPS}}"
```

**Examples**:

Single camera:
```bash
--cameras "{'camera_wrist': {'type': 'opencv', 'index_or_path': 0, 'width': 480, 'height': 640, 'fps': 30}}"
```

Multiple cameras:
```bash
--cameras "{'camera_wrist': {'type': 'opencv', 'index_or_path': 0, 'width': 480, 'height': 640, 'fps': 30}, 'camera_top': {'type': 'opencv', 'index_or_path': 1, 'width': 480, 'height': 640, 'fps': 30}}"
```

## Performance Tuning

### Key Parameters

1. **`actions_per_chunk`** (5-20 recommended for ACT):
   - Higher values: Fewer network requests, less responsive
   - Lower values: More network requests, more responsive

2. **`chunk_size_threshold`** (0.3-0.7 recommended):
   - Higher values: More frequent updates, more responsive
   - Lower values: Fewer updates, more sequential behavior

### Tuning Process

1. Start with defaults (`actions_per_chunk=10`, `chunk_size_threshold=0.5`)
2. Use `--debug_visualize_queue_size` to monitor action queue
3. Adjust parameters based on your robot's performance:
   - Queue often empty? Increase `actions_per_chunk` or decrease `chunk_size_threshold`
   - Robot too sluggish? Decrease `actions_per_chunk` or increase `chunk_size_threshold`

## Safety Features

- **Max Relative Target**: Limits joint movement per step
- **Automatic Disconnect**: Disables torque when client disconnects
- **Dry Run Mode**: Test configuration without connecting to robot:
  ```bash
  python deploy_act_robot_client.py [OPTIONS] --dry_run
  ```

## Troubleshooting

### Common Issues

1. **Connection Failed**:
   - Check robot IP address and network connectivity
   - Ensure Franka robot is powered on and unlocked
   - Verify firewall settings

2. **Policy Loading Failed**:
   - Verify checkpoint path exists
   - Check that checkpoint is compatible with current LeRobot version
   - Ensure sufficient GPU memory for model loading

3. **Camera Issues**:
   - Run `python -m lerobot.find_cameras` to verify camera indices
   - Check camera permissions and USB connections
   - Ensure camera resolution matches training data

4. **Poor Performance**:
   - Monitor action queue size with `--debug_visualize_queue_size`
   - Adjust `actions_per_chunk` and `chunk_size_threshold`
   - Check network latency between server and client

### Debugging Options

- `--debug_visualize_queue_size`: Show action queue size plot
- `--dry_run`: Validate configuration without connecting
- Increase logging verbosity by setting `LEROBOT_LOG_LEVEL=DEBUG`

## Network Deployment

To run the policy server on a remote machine:

1. **On the inference server**:
   ```bash
   python deploy_act_policy_server.py --host 0.0.0.0 --port 8080
   ```

2. **On the robot controller**:
   ```bash
   python deploy_act_robot_client.py --server_address SERVER_IP:8080 [OTHER_OPTIONS]
   ```

## Examples

## Examples

## Examples

### ACT Policy
```bash
# Terminal 1: Start server
python deploy_policy_server.py --fps 10

# Terminal 2: Start client
python deploy_robot_client.py \
    --robot_ip 172.16.0.2 \
    --policy_type act \
    --checkpoint_path outputs/train/act_franka_fr3_softtoy/checkpoints/last \
    --cameras "{'camera_wrist': {'type': 'opencv', 'index_or_path': 0, 'width': 480, 'height': 640, 'fps': 30}}" \
    --task "manipulate the soft toy"
```

### VQ-BeT Policy
```bash
python deploy_robot_client.py \
    --robot_ip 172.16.0.2 \
    --policy_type vqbet \
    --checkpoint_path outputs/train/vqbet_franka_task/checkpoints/last \
    --cameras "{'camera_top': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 480, 'fps': 30}}" \
    --actions_per_chunk 50 \
    --fps 20
```

### SmolVLA Policy
```bash
python deploy_robot_client.py \
    --robot_ip 172.16.0.2 \
    --policy_type smolvla \
    --checkpoint_path lerobot/smolvla_base \
    --cameras "{'camera_wrist': {'type': 'realsense', 'serial_number': '123456', 'width': 640, 'height': 480, 'fps': 30}}" \
    --task "fold the towel neatly" \
    --actions_per_chunk 30
```

### GROOT Policy
```bash
python deploy_robot_client.py \
    --robot_ip 172.16.0.2 \
    --policy_type groot \
    --checkpoint_path outputs/train/groot_franka/checkpoints/last \
    --cameras "{'camera_wrist': {'type': 'opencv', 'index_or_path': 0, 'width': 480, 'height': 640, 'fps': 30}}" \
    --task "precise manipulation task" \
    --actions_per_chunk 20
```

### High-Performance Setup
```bash
python deploy_robot_client.py \
    --robot_ip 172.16.0.2 \
    --policy_type diffusion \
    --checkpoint_path outputs/train/diffusion_policy/checkpoints/last \
    --cameras "{'wrist': {'type': 'opencv', 'index_or_path': 0, 'width': 480, 'height': 640, 'fps': 30}}" \
    --task "precision assembly task" \
    --actions_per_chunk 15 \
    --chunk_size_threshold 0.6 \
    --fps 50 \
    --policy_device cuda
```

### Debug Mode
```bash
python deploy_robot_client.py \
    --robot_ip 172.16.0.2 \
    --policy_type act \
    --checkpoint_path your_checkpoint \
    --cameras "{'cam': {'type': 'opencv', 'index_or_path': 0, 'width': 480, 'height': 640, 'fps': 30}}" \
    --debug_visualize_queue_size \
    --dry_run
```

For more information, see the [LeRobot async inference documentation](https://huggingface.co/docs/lerobot/async).