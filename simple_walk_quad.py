import mujoco
import mujoco.viewer
import numpy as np
import os
import csv
from datetime import datetime


def simple_walk_demo(model_path):
    """Simple hardcoded walking demonstration for Unitree Go2"""
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Print available joints
    print("\nAvailable joints:")
    leg_joints = {}
    for i in range(model.njnt):
        jname = model.joint(i).name
        print(f"  [{i}] {jname}")
        
        # Group by leg
        if jname.startswith('FL_'):
            leg_joints.setdefault('FL', []).append(i)
        elif jname.startswith('FR_'):
            leg_joints.setdefault('FR', []).append(i)
        elif jname.startswith('RL_'):
            leg_joints.setdefault('RL', []).append(i)
        elif jname.startswith('RR_'):
            leg_joints.setdefault('RR', []).append(i)
    
    # Sort joints by position
    for leg in leg_joints:
        leg_joints[leg] = sorted(leg_joints[leg], key=lambda j: model.jnt_qposadr[j])
    
    print(f"\nLeg joints grouped: {leg_joints}")
    
    # Reset to standing pose
    mujoco.mj_resetData(model, data)
    
    # Set initial standing configuration
    for leg, joints in leg_joints.items():
        if len(joints) >= 3:
            hip_idx = model.jnt_qposadr[joints[0]]
            thigh_idx = model.jnt_qposadr[joints[1]]
            calf_idx = model.jnt_qposadr[joints[2]]
            
            data.qpos[hip_idx] = 0.0      # Hip neutral
            data.qpos[thigh_idx] = 0.8    # Thigh forward
            data.qpos[calf_idx] = -1.5    # Calf bent
    
    mujoco.mj_forward(model, data)
    print(f"Initial height: {data.qpos[2]:.3f}m\n")
    
    # ==== DATA LOGGING SETUP ====
    # Create CSV file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"simple_quad_walk_data_{timestamp}.csv"
    
    # Prepare CSV header
    header = ['timestamp', 'pos_x', 'pos_y', 'pos_z', 
              'quat_w', 'quat_x', 'quat_y', 'quat_z',
              'vel_x', 'vel_y', 'vel_z',
              'ang_vel_x', 'ang_vel_y', 'ang_vel_z']
    
    # Add action headers for each joint
    for leg_name in ['FL', 'FR', 'RL', 'RR']:
        header.extend([f'{leg_name}_hip_action', f'{leg_name}_thigh_action', f'{leg_name}_calf_action'])
    
    # Add target angle headers
    for leg_name in ['FL', 'FR', 'RL', 'RR']:
        header.extend([f'{leg_name}_hip_target', f'{leg_name}_thigh_target', f'{leg_name}_calf_target'])
    
    # Initialize data storage
    logged_data = []
    
    print(f"📝 Logging data to: {csv_filename}")
    print("=" * 60)
    # ============================
    
    # Walking parameters
    freq = 2.5  # Walking frequency (Hz)
    max_simulation_time = 10.0  # Stop after 10 seconds
    
    # Trot gait - BACK TO ORIGINAL
    def get_phase_offset(leg_name):
        if leg_name in ['FL', 'RR']:
            return np.pi
        else:  # FR, RL
            return 0.0
    
    # Simple joint trajectories for trotting
    def get_joint_angles(t, leg_name):
        phase = 2 * np.pi * freq * t + get_phase_offset(leg_name)
        
        # Swing phase when sin(phase) > 0
        swing = np.sin(phase)
        
        # Hip: NO hip motion - this prevents rotation
        hip = 0.0
        
        # Thigh: NEGATE to walk forward instead of backward
        # Original was: 0.8 + 0.4 * swing (walked backward)
        # So reverse it to: use negative swing values
        thigh = 0.8 - 0.4 * swing  # Flip the direction
        
        # Calf: Keep original
        if swing > 0:  # Swing phase
            calf = -1.8 + 0.5 * swing
        else:  # Stance phase
            calf = -1.5
        
        return [hip, thigh, calf]
    
    print("Starting walk demo...")
    print("Simulation will stop after 10 seconds\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_print = 0
        
        while viewer.is_running() and data.time < max_simulation_time:
            t = data.time
            
            # ==== PREPARE DATA ROW ====
            data_row = [t]
            
            # Position (x, y, z)
            data_row.extend(data.qpos[:3].tolist())
            
            # Orientation (quaternion: w, x, y, z)
            data_row.extend(data.qpos[3:7].tolist())
            
            # Linear velocity (x, y, z)
            data_row.extend(data.qvel[:3].tolist())
            
            # Angular velocity (x, y, z)
            data_row.extend(data.qvel[3:6].tolist())
            
            # Store actions and targets
            actions = []
            targets = []
            # ==========================
            
            # Compute target angles for each leg
            for leg_name, joints in leg_joints.items():
                if len(joints) < 3:
                    continue
                
                target_angles = get_joint_angles(t, leg_name)
                
                # Store target angles
                targets.extend(target_angles)
                
                # Apply PD control to each joint
                leg_actions = []
                for i, joint_id in enumerate(joints[:3]):
                    # Find actuator for this joint
                    action_value = 0.0
                    for act_id in range(model.nu):
                        if model.actuator_trnid[act_id, 0] == joint_id:
                            qpos_idx = model.jnt_qposadr[joint_id]
                            qvel_idx = model.jnt_dofadr[joint_id]
                            
                            # PD control
                            kp = 100.0  # Position gain
                            kd = 10.0   # Velocity gain
                            
                            error = target_angles[i] - data.qpos[qpos_idx]
                            error_dot = -data.qvel[qvel_idx]
                            
                            action_value = kp * error + kd * error_dot
                            data.ctrl[act_id] = action_value
                            break
                    
                    leg_actions.append(action_value)
                
                actions.extend(leg_actions)
            
            # Add actions and targets to data row
            data_row.extend(actions)
            data_row.extend(targets)
            
            # Append to logged data
            logged_data.append(data_row)
            
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Print status every 2 seconds
            if t - last_print > 2.0:
                pos = data.qpos[:3]
                height = pos[2]
                dist = np.sqrt(pos[0]**2 + pos[1]**2)
                print(f"t={t:5.1f}s | pos=[{pos[0]:6.2f}, {pos[1]:6.2f}] | height={height:.3f}m | dist={dist:.2f}m")
                last_print = t
    
    # ==== SAVE DATA TO CSV ====
    print("\n" + "=" * 60)
    print(f"💾 Saving {len(logged_data)} data points to {csv_filename}...")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(logged_data)
    
    print(f"✅ Data saved successfully!")
    print(f"📊 Total simulation time: {data.time:.2f}s")
    print(f"📁 File location: {os.path.abspath(csv_filename)}")
    print("=" * 60)
    # ==========================


if __name__ == "__main__":
    # Use the path from your code
    base_path = "/Users/gagandeshad/Desktop/unitree/mujoco_menagerie"
    model_path = f"{base_path}/unitree_go2/scene.xml"
    
    simple_walk_demo(model_path)