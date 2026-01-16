import os
import time
import numpy as np
import pandas as pd
from sim_class import Simulation
from PID_implementation import (
    f1, learn_plant_positions, create_zone_boundaries,
    process_image_with_positions, PetriDishMapper
)

class PIDController:
    
    def __init__(self, Kp=0.20, Ki=0.15, Kd=0.0005):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = 1.0 / 240.0
        self.reset()
    
    def reset(self):
        self.integral = np.array([0.0, 0.0, 0.0])
        self.previous_error = np.array([0.0, 0.0, 0.0])
    
    def compute(self, target, current):
        error = np.array(target) - np.array(current)
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -1.0, 1.0)
        derivative = (error - self.previous_error) / self.dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output, error


def move_to_target(sim, controller, target_pos, max_steps=2000, tolerance=0.001):
    controller.reset()
    
    for step in range(max_steps):
        states = sim.get_states()
        robot_key = list(states.keys())[0]
        current_pos = states[robot_key]['pipette_position']
        
        velocity, error = controller.compute(target_pos, current_pos)
        error_magnitude = np.linalg.norm(error)
        
        if error_magnitude < tolerance:
            return True, error_magnitude * 1000, step
        
        action = [[velocity[0], velocity[1], velocity[2], 0]]
        sim.run(action, num_steps=1)
    
    states = sim.get_states()
    current_pos = states[robot_key]['pipette_position']
    final_error = np.linalg.norm(np.array(target_pos) - np.array(current_pos))
    return False, final_error * 1000, max_steps


def dispense_liquid(sim):
    action = [[0, 0, 0, 1]]
    sim.run(action, num_steps=1)
    time.sleep(0.1)


def run_cv_pipeline(img_path, model, zone_boundaries):
    """
    Run CV pipeline on image and return target coordinates.
    Returns list of dicts with pixel and sim coordinates, sorted left-to-right.
    """
    plant_mask, position_coords, metadata = process_image_with_positions(
        img_path, model, zone_boundaries, return_coords=True
    )
    
    detected_count = sum(1 for coord in position_coords if coord is not None)
    print(f"Detected {detected_count}/5 plants")
    
    pixel_size = metadata['cropped_shape'][0]
    mapper = PetriDishMapper(pixel_size)
    
    targets = []
    for position_id, pixel_coords in enumerate(position_coords, 1):
        if pixel_coords is not None:
            sim_x, sim_y, sim_z = mapper.pixel_to_sim(pixel_coords[0], pixel_coords[1])
            
            if mapper.validate_coords(sim_x, sim_y, sim_z):
                targets.append({
                    'position': position_id,
                    'pixel_x': pixel_coords[0],
                    'pixel_y': pixel_coords[1],
                    'sim_x': sim_x,
                    'sim_y': sim_y,
                    'sim_z': sim_z
                })
                print(f"Position {position_id}: pixel ({pixel_coords[0]:4d}, {pixel_coords[1]:4d}) -> sim ({sim_x:.5f}, {sim_y:.5f})")
    
    # Sort left-to-right by pixel_x (not sim_x due to coordinate transforms)
    targets.sort(key=lambda t: t['pixel_x'])
    print(f"{len(targets)} valid targets (sorted left-to-right)")
    
    return targets, metadata


def process_with_simulation(sim, model, zone_boundaries):
    """
    Process the plate currently loaded in simulation.
    
    Parameters:
    - sim: Existing Simulation instance (already has texture loaded)
    - model: Trained segmentation model
    - zone_boundaries: Pre-computed zone boundaries
    
    Returns:
    - Summary dict with results
    """
    # Get plate image path from simulation
    img_path = sim.get_plate_image()
    img_name = os.path.basename(img_path)
    print(f"\nProcessing plate: {img_name}")
    
    # Run CV pipeline
    print("Running CV pipeline...")
    targets, metadata = run_cv_pipeline(img_path, model, zone_boundaries)
    
    detected_count = len(targets)
    
    if len(targets) == 0:
        print("No valid targets detected")
        return {
            'image_name': img_name,
            'plants_detected': 0,
            'plants_dispensed': 0,
            'avg_error_mm': 0,
            'total_time_sec': 0,
            'results': []
        }
    
    # Initialize robot
    controller = PIDController(Kp=0.20, Ki=0.15, Kd=0.0005)
    START_POS = [0.15, 0.13, 0.20]
    sim.set_start_position(*START_POS)
    time.sleep(0.5)
    print("Robot ready")
    
    # Dispensing sequence
    print(f"\nDispensing to {len(targets)} targets (left to right)...")
    
    start_time = time.time()
    results = []
    
    for i, target in enumerate(targets, 1):
        print(f"\nTarget {i}/{len(targets)}: Position {target['position']}")
        print(f"Moving to ({target['sim_x']:.5f}, {target['sim_y']:.5f})")
        
        move_start = time.time()
        target_coords = [target['sim_x'], target['sim_y'], target['sim_z']]
        
        converged, error_mm, steps = move_to_target(
            sim, controller, target_coords, max_steps=2000, tolerance=0.001
        )
        
        move_time = time.time() - move_start
        
        if converged:
            print(f"Arrived (error: {error_mm:.2f}mm, time: {move_time:.2f}s)")
            dispense_liquid(sim)
            print("Dispensed")
            dispensed = True
        else:
            print(f"Timeout (error: {error_mm:.2f}mm)")
            dispensed = False
        
        results.append({
            'position': target['position'],
            'pixel_x': target['pixel_x'],
            'pixel_y': target['pixel_y'],
            'target_x': target['sim_x'],
            'target_y': target['sim_y'],
            'target_z': target['sim_z'],
            'converged': converged,
            'error_mm': error_mm,
            'steps': steps,
            'time_sec': move_time,
            'dispensed': dispensed
        })
    
    total_time = time.time() - start_time
    
    # Wait for drops to settle
    print("\nWaiting for drops to settle...")
    for _ in range(240):
        sim.run([[0, 0, 0, 0]], num_steps=1)
    
    dispensed_count = sum(1 for r in results if r['dispensed'])
    avg_error = np.mean([r['error_mm'] for r in results]) if results else 0
    
    print(f"\nComplete: {dispensed_count}/{len(targets)} dispensed")
    print(f"Average error: {avg_error:.2f} mm")
    print(f"Total time: {total_time:.2f} seconds")
    
    return {
        'image_name': img_name,
        'plants_detected': detected_count,
        'plants_dispensed': dispensed_count,
        'avg_error_mm': avg_error,
        'total_time_sec': total_time,
        'results': results
    }


def run_full_pipeline(model, zone_boundaries, render=True):
    """
    Complete pipeline: create simulation, get image, run CV, control robot.
    
    Parameters:
    - model: Trained segmentation model  
    - zone_boundaries: Pre-computed zone boundaries
    - render: Whether to show GUI
    
    Returns:
    - Summary dict with results
    """
    # Create simulation (automatically loads random texture)
    print("Initializing simulation...")
    sim = Simulation(num_agents=1, render=render)
    
    # Get the plate image that corresponds to loaded texture
    img_path = sim.get_plate_image()
    print(f"Loaded plate: {img_path}")
    
    # Process with this simulation instance
    summary = process_with_simulation(sim, model, zone_boundaries)
    
    # Cleanup
    time.sleep(2)
    sim.close()
    
    return summary


def run_batch_pipeline(model, zone_boundaries, num_plates=10, render=False):
    """
    Process multiple random plates sequentially.
    """
    all_summaries = []
    
    for i in range(num_plates):
        print(f"\n{'='*60}")
        print(f"PLATE {i+1}/{num_plates}")
        print('='*60)
        
        summary = run_full_pipeline(model, zone_boundaries, render=render)
        all_summaries.append(summary)
    
    # Print batch statistics
    total_detected = sum(s['plants_detected'] for s in all_summaries)
    total_dispensed = sum(s['plants_dispensed'] for s in all_summaries)
    avg_errors = [s['avg_error_mm'] for s in all_summaries if s['avg_error_mm'] > 0]
    
    print(f"\n{'='*60}")
    print("BATCH SUMMARY")
    print('='*60)
    print(f"Plates processed: {len(all_summaries)}")
    print(f"Total plants detected: {total_detected}")
    print(f"Total plants dispensed: {total_dispensed}")
    if avg_errors:
        print(f"Mean error: {np.mean(avg_errors):.2f} mm")
    
    return all_summaries