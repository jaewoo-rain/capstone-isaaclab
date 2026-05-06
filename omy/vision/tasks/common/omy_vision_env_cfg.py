from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg

from source.omy.omy_robot_cfg import OMY_CFG


@configclass
class OmyVisionEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    episode_length_s: float = 8.0
    action_space: int = 7
    observation_space: int = 34
    state_space: int = 0

    enable_yolo: bool = True
    dataset_mode: bool = False

    use_camera: bool = True
    use_depth: bool = True
    yolo_model_path: str = 'checkpoints/yolo/best.pt'
    yolo_device: str = 'cuda'
    yolo_imgsz: int = 640
    yolo_conf: float = 0.35
    yolo_iou: float = 0.45
    yolo_tracker_cfg: str = 'bytetrack.yaml'
    max_stale_frames: int = 5

    camera_width: int = 848
    camera_height: int = 480
    camera_hfov_deg: float = 84.0
    camera_vfov_deg: float = 58.0
    camera_min_depth_m: float = 0.07
    camera_max_depth_m: float = 0.50

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode='multiply', restitution_combine_mode='multiply', static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=2.8, replicate_physics=True)
    robot = OMY_CFG.replace(prim_path='/World/envs/env_.*/Robot')

    object_size_xyz = (0.044, 0.118, 0.139)
    _common_spawn = dict(
        size=object_size_xyz,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=False, max_depenetration_velocity=5.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=3.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    )
    object_a: RigidObjectCfg = RigidObjectCfg(prim_path='/World/envs/env_.*/ObjectA', spawn=sim_utils.CuboidCfg(**_common_spawn, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 0.2))), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.42, -0.15, object_size_xyz[2] * 0.5)))
    object_b: RigidObjectCfg = RigidObjectCfg(prim_path='/World/envs/env_.*/ObjectB', spawn=sim_utils.CuboidCfg(**_common_spawn, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.9))), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.47, 0.00, object_size_xyz[2] * 0.5)))
    object_c: RigidObjectCfg = RigidObjectCfg(prim_path='/World/envs/env_.*/ObjectC', spawn=sim_utils.CuboidCfg(**_common_spawn, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.55, 0.2))), init_state=RigidObjectCfg.InitialStateCfg(pos=(0.52, 0.15, object_size_xyz[2] * 0.5)))

    slot_size_xyz = (0.05, 0.13, 0.01)
    slot_grid_rows: int = 3
    slot_grid_cols: int = 3
    slot_spacing_x: float = 0.12
    slot_spacing_y: float = 0.16
    slot_origin_xy: tuple[float, float] = (0.70, -0.16)

    camera: CameraCfg = CameraCfg(
        prim_path='/World/envs/env_.*/Robot/OMY/link6/d405_like_camera',
        update_period=0.0,
        height=camera_height,
        width=camera_width,
        data_types=['rgb', 'distance_to_camera'],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=0.25, horizontal_aperture=20.955, clipping_range=(camera_min_depth_m, 2.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.1, 0.084), 
            rot=(0.0, 0.0, 0.7071068, -0.7071068), 
            convention='ros'),
    )

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path='/World/ground', terrain_type='plane', collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(friction_combine_mode='multiply', restitution_combine_mode='multiply', static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )

    action_scale: float = 1.8
    dof_velocity_scale: float = 1.0
    object_pos_noise: float = 0.03
    lift_height_threshold: float = 0.22
    success_bonus: float = 15.0
    ee_body_name: str = 'link6'
    left_finger_body_name: str = 'rh_p12_rn_l1'
    right_finger_body_name: str = 'rh_p12_rn_r1'
    left_tip_body_name: str = 'rh_p12_rn_l2'
    right_tip_body_name: str = 'rh_p12_rn_r2'
    gripper_master_joint_name: str = 'rh_r1_joint'

    n_steps: int = 512
    batch_size: int = 4096
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    n_epochs: int = 5
