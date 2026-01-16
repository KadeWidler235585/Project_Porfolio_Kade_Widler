"""Minimal PyBullet simulation - no textures, no extras"""
import pybullet as p
import pybullet_data

class MinimalSimulation:
    def __init__(self, render=False):
        # Connect to PyBullet
        mode = p.GUI if render else p.DIRECT
        self.physics_client = p.connect(mode)
        
        if not render:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load robot
        position = [0, 0, 0.03]
        self.robot_id = p.loadURDF(
            "ot_2_simulation_v6.urdf",
            position,
            [0, 0, 0, 1],
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )
        
        # Fix robot in place
        start_pos, start_orn = p.getBasePositionAndOrientation(self.robot_id)
        p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=start_pos,
            childFrameOrientation=start_orn
        )
        
        self.pipette_offset = [0.073, 0.0895, 0.0895]
    
    def run(self, action):
        """Apply velocity command and step simulation"""
        vx, vy, vz = action
        
        p.setJointMotorControl2(self.robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=-vx, force=500)
        p.setJointMotorControl2(self.robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=-vy, force=500)
        p.setJointMotorControl2(self.robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=vz, force=800)
        
        p.stepSimulation()
        
        return self.get_pipette_position()
    
    def get_pipette_position(self):
        """Get current pipette position"""
        robot_pos = list(p.getBasePositionAndOrientation(self.robot_id)[0])
        joint_states = p.getJointStates(self.robot_id, [0, 1, 2])
        
        robot_pos[0] -= joint_states[0][0]
        robot_pos[1] -= joint_states[1][0]
        robot_pos[2] += joint_states[2][0]
        
        pipette_pos = [
            robot_pos[0] + self.pipette_offset[0],
            robot_pos[1] + self.pipette_offset[1],
            robot_pos[2] + self.pipette_offset[2]
        ]
        
        return pipette_pos
    
    def set_position(self, x, y, z):
        """Set pipette to specific position"""
        robot_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        
        adjusted_x = x - robot_pos[0] - self.pipette_offset[0]
        adjusted_y = y - robot_pos[1] - self.pipette_offset[1]
        adjusted_z = z - robot_pos[2] - self.pipette_offset[2]
        
        p.resetJointState(self.robot_id, 0, targetValue=adjusted_x)
        p.resetJointState(self.robot_id, 1, targetValue=adjusted_y)
        p.resetJointState(self.robot_id, 2, targetValue=adjusted_z)
    
    def close(self):
        p.disconnect()
