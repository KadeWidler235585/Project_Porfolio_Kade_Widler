# OT-2 Robotic Pipette Control

This repository contains the evaluation of reinforcement learning (RL) and PID controllers for robotic pipette control. The final integrated pipeline uses a PID controller due to superior performance in accuracy and execution time.

---

## Table of Contents

- [Setup and Libraries](#setup-and-libraries)
- [Code Files](#code-files)
- [RL Controller Evaluation](#rl-controller-evaluation)
- [PID vs RL Comparison](#pid-vs-rl-comparison)
- [Final Pipeline Integration](#final-pipeline-integration)
- [Reproduction Steps](#reproduction-steps)
- [Video Demo](#video-demo)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)

---

## Setup and Libraries

The following Python libraries are required:

- `stable-baselines3` – RL algorithm implementations (PPO, SAC)  
- `gymnasium` – Environment interface standard  
- `pybullet` – Physics simulation and robot control  
- `numpy` – Numerical computations  
- `tensorflow` – Neural network backend (U-Net segmentation model)  
- `clearml` – Remote training orchestration  
- `matplotlib` – Performance visualization  
- `opencv-python` – Image processing for CV pipeline  
- `scikit-image` – Skeletonization and morphological operations  
- `patchify` – Patch-based image segmentation  

---

## Code Files

| File | Description |
|------|-------------|
| `PID_implementation.py` | CV pipeline for root detection and coordinate mapping |
| `simulation_integration.py` | PID controller integration with simulation |
| `sim_class.py` | PyBullet simulation environment |
| `RL_evaluation_sim.py` | Simulation environment used for RL evaluation |
| `RL_evaluation_wrapper.py` | Gymnasium wrapper for RL evaluation |
| `wrapper_improved.py` | Original PPO wrapper (serves as `ot2_gym_wrapper.py`) |
| `ot2_ppo_improved_final.zip` | Best group PPO model weights (trained 5M steps, shaped reward + progress bonus) |
| `train_improved.py` | PPO training script |
| `test_improved_model.py` | PPO evaluation script |

---

## RL Controller Evaluation

Evaluation was performed using **PPO** (`ot2_ppo_improved_final.zip`) in the same simulation setup as the PID controller to allow direct comparison.

**Metrics:**

- **Total targets:** 83  
- **Successfully reached:** 0  
- **Success rate:** 0.0% (never within 1 mm)  

**Positioning accuracy:**

- **Mean error:** 2.83 mm  
- **Std error:** 0.23 mm  
- **Max error:** 3.53 mm  

**Execution time:**

- **Mean time:** 0.3 sec per plant  
- **Total time:** 21.9 sec  

> Despite extensive training (5M steps) with shaped rewards and progress bonuses, RL never achieved sub-millimeter precision and was less consistent than the PID controller.

---

## PID vs RL Comparison

| Metric | PID Controller | RL Controller |
|--------|----------------|---------------|
| Success rate (≤1mm) | 100% | 0% |
| Mean error | 0.85 mm | 2.83 mm |
| Std error | 0.12 mm | 0.23 mm |
| Max error | 1.25 mm | 3.53 mm |
| Execution time per plant | 0.05 sec | 0.3 sec |

**Conclusion:** PID outperforms RL in all metrics for this task. RL demonstrates the ability to reduce positional error but cannot match deterministic precision without carefully engineered reward functions and state representations.

---

## Final Pipeline Integration

Based on the comparison results, the **PID controller** was selected for the final integrated pipeline due to its significantly lower error rate and faster execution time.

### Pipeline Architecture

The integrated system consists of two main components:

1. **`PID_implementation.py`** – Computer vision pipeline for root tip detection and coordinate transformation
   - U-Net model segments plant roots from petri dish images
   - Detects root tips (bottom-most points) for each plant
   - Maps pixel coordinates to simulation coordinate space via `PetriDishMapper`
   - Handles 90° rotation and 180° flip between image and simulation orientations

2. **`simulation_integration.py`** – Robot control and simulation interaction
   - `PIDController` class with tuned gains (Kp=0.20, Ki=0.15, Kd=0.0005)
   - `process_with_simulation()` connects CV output to robot control
   - Targets are sorted by `pixel_x` for left-to-right dispensing order
   - Automatic texture/plate image synchronization with simulation

### Key Implementation Details

**Coordinate Mapping:**
- Pixel coordinates from cropped petri dish images are transformed to simulation space
- The `PetriDishMapper` class handles the coordinate system differences between the CV pipeline and PyBullet simulation

**Simulation-Image Synchronization:**
- The simulation randomly selects a texture from `textures/` folder
- The corresponding plate image from `textures/_plates/` is used for CV processing
- Both folders must have matching, sorted files for correct alignment

### Troubleshooting: .DS_Store Conflict

A critical issue was discovered during integration: macOS creates invisible `.DS_Store` files in directories, which caused index misalignment between the texture and plate image lists.

**Problem:** `os.listdir()` includes `.DS_Store` and subdirectories like `_plates`, causing `IndexError` when matching textures to plate images.

**Solution:** Filter hidden files and subdirectories when loading textures in `sim_class.py`:

```python
# Fixed texture loading in sim_class.py
texture_list = sorted([f for f in os.listdir("textures") 
                       if not f.startswith('.') and not f.startswith('_')])
plates_list = sorted([f for f in os.listdir("textures/_plates") 
                      if not f.startswith('.')])

num_available = min(len(texture_list), len(plates_list))
random_index = random.randint(0, num_available - 1)

random_texture = texture_list[random_index]
self.plate_image_path = f'textures/_plates/{plates_list[random_index]}'
self.textureId = p.loadTexture(f'textures/{random_texture}')
```

---

## Reproduction Steps

### 1. Run RL Evaluation (for comparison)

Required files: `RL_evaluation_wrapper.py`, `RL_evaluation_sim.py`

```python
from RL_evaluation_wrapper import ImprovedOT2Env
from stable_baselines3 import PPO

env = ImprovedOT2Env()
MODEL_PATH = "ot2_ppo_improved_final.zip" 
model = PPO.load(MODEL_PATH)
```

### 2. Run PID Integration Pipeline

Required files: `PID_implementation.py`, `simulation_integration.py`, `sim_class.py`

```python
import tensorflow as tf
from simulation_integration import *
from PID_implementation import *
from sim_class import Simulation

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Load trained U-Net model
MODEL_PATH = "path/to/your_unet_model.h5"
model = simple_unet_model(128, 128, 3)
model.load_weights(MODEL_PATH)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1])

# Learn plant positions from dataset
PLATES_FOLDER = "textures/_plates"
avg_positions = learn_plant_positions(PLATES_FOLDER, model)
zone_boundaries = create_zone_boundaries(avg_positions)
```

### 3. Run Single Plate Test

```python
# Create simulation (automatically loads random texture)
sim = Simulation(num_agents=1, render=True)

# Get matching plate image path
IMAGE_PATH = sim.get_plate_image()
print(f"Processing: {IMAGE_PATH}")

# Run CV pipeline and control robot
summary = process_with_simulation(sim, model, zone_boundaries)

# View results
print(f"Plants detected: {summary['plants_detected']}")
print(f"Plants dispensed: {summary['plants_dispensed']}")
print(f"Average error: {summary['avg_error_mm']:.2f} mm")

# Cleanup
sim.close()
```

### 4. Run Batch Processing

```python
summaries = run_batch_pipeline(model, zone_boundaries, num_plates=10, render=False)
```

---

## Future Improvements

The PID current performance < 1 mm does not need to be further improved when it comes to accuracy, execution speed or success rate.

- The model performs at accuracy where increasing would not lead to system improvements (plant inoculation).
- Also, the speed was selected to be applied for the use case. Further increases in proportion could lead to model failure (table movement, unecessary water droplets).

Possible Improvements:
1. Testing in Hades system (full integration)
2. Water physics inspection for optimal placement
3. Conveyer belt integration allowing full automation

While there are improvements to be made, the first improvements would be to the root detection pipeline, increasing robustness through adaptation to other plant species.

## Conclusion

The PID controller was selected for the final pipeline integration based on empirical evaluation showing:

- **100% success rate** at reaching targets within 1mm tolerance (vs 0% for RL)
- **3.3x lower mean error** (0.85mm vs 2.83mm)
- **6x faster execution** (0.05 sec vs 0.3 sec per plant)

The integrated pipeline successfully:
1. Detects plant root tips using U-Net segmentation
2. Transforms pixel coordinates to simulation space
3. Controls the OT-2 robot to dispense liquid at each root tip
4. Processes plates in left-to-right order for consistent operation

The main technical challenge was synchronizing simulation textures with CV input images, resolved by filtering hidden files from directory listings.
