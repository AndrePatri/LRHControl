import numpy as np
import pinocchio as pin
from pinocchio import GeometryType
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf


class RhcMeshcatPublisher:
    """
    Lightweight Meshcat publisher built on top of pinocchio's MeshcatVisualizer.

    Assumes a floating base: the configuration vector is [xyz, quaternion(xyzw), joints].
    The user must pass root orientation as (w, x, y, z); we convert to (x, y, z, w)
    for pinocchio.
    """

    def __init__(
        self,
        urdf_path: str,
        mesh_dir: str = None,
        base_link_name: str = "base_link",
        host: str = "0.0.0.0",
        port: int = 7000,
        visualize_collisions: bool = False,
        robot_joint_names=None,
    ):
        self.urdf_path = urdf_path
        self.mesh_dir = mesh_dir
        self.base_link_name = base_link_name
        self.visualize_collisions = visualize_collisions
        self.robot_joint_names = list(robot_joint_names) if robot_joint_names is not None else []

        # Load pinocchio model + geometry
        
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, mesh_dir, pin.JointModelFreeFlyer()
        )
        for i, joint in enumerate(self.model.joints):
            print(i, self.model.names[i])
        self.data = self.model.createData()
        self.visual_data = self.visual_model.createData()
        self.collision_data = self.collision_model.createData()

        # Meshcat visualizer
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        # Connect/start server (Meshcat server should be running; open browser if possible)
        self.viz.initViewer(open=True)
        # Load geometry (visuals and collisions if requested)
        # Older pinocchio versions do not accept geometryTypes kwarg; fall back to default load.
        self.viz.loadViewerModel()
        # Display initial pose (fallback to zero/neutral if no "home")
        if hasattr(self.model, "referenceConfigurations") and "home" in self.model.referenceConfigurations:
            q0 = self.model.referenceConfigurations["home"]
        else:
            q0 = pin.neutral(self.model)
        self.viz.display(q0)

        # build joint name map: model (after free-flyer) -> RobotState joint ordering
        self._joint_index_map, self._joint_valid_mask, self._actuated_joint_ids = self._build_joint_index_map()

    def _build_joint_index_map(self):
        """
        Map pinocchio joint order (after free-flyer) to incoming RobotState joint order.
        Returns:
            mapping: np.ndarray of shape (n_model_actuated,) with source indices into robot_joint_names (or -1).
            valid_mask: boolean mask for entries that were matched.
            joint_ids: list of pinocchio joint ids (model order, after free-flyer).
        """
        model_joint_names = []
        model_joint_ids = []
        # pin model.names includes "universe", "root_joint" (free-flyer), then actuated joints
        for jidx in range(2, self.model.njoints):  # skip universe (0) and free-flyer (1)
            model_joint_names.append(self.model.names[jidx])
            model_joint_ids.append(jidx)
        name_to_idx = {n: i for i, n in enumerate(self.robot_joint_names)}
        mapping_list = []
        unmatched = []
        for mj in model_joint_names:
            src = name_to_idx.get(mj, -1)
            mapping_list.append(src)
            if src < 0:
                unmatched.append(mj)
        if unmatched:
            print(f"[meshcat] Warning: {len(unmatched)} joints missing in RobotState: {unmatched[:5]}{'...' if len(unmatched)>5 else ''}")
        mapping = np.array(mapping_list, dtype=int)
        valid_mask = mapping >= 0
        return mapping, valid_mask, model_joint_ids

    def pinocchio_q(self, base_pos_w, base_quat_wxyz, joint_positions):
        """Assemble pinocchio configuration vector."""
        q = np.zeros(self.model.nq)
        q[0:3] = base_pos_w
        # pin expects xyzw
        q[3:7] = np.array([base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]])
        # map joints
        jp = np.asarray(joint_positions, dtype=float).reshape(-1)
        for k, joint_id in enumerate(self._actuated_joint_ids):
            src = self._joint_index_map[k]
            if src < 0 or src >= jp.size:
                continue
            qpos = self.model.idx_qs[joint_id]
            nq = self.model.joints[joint_id].nq
            if nq == 1:
                q[qpos] = jp[src]
            elif nq == 2:
                # treat as continuous revolute -> cos/sin parameterization
                ang = jp[src]
                q[qpos] = np.cos(0.5 * ang)
                q[qpos + 1] = np.sin(0.5 * ang)
            else:
                # fallback: copy as many as available
                end = min(qpos + nq, 7 + jp.size)
                span = end - qpos
                if span > 0:
                    q[qpos:end] = jp[src:src + span]

        # Normalize configuration (wrap revolute/continuous joints, renormalize base quat)
        q_norm = q.copy()
        pin.normalize(self.model, q_norm)
        return q_norm

    def display(self, base_pos_w, base_quat_wxyz, joint_positions):
        q = self.pinocchio_q(base_pos_w, base_quat_wxyz, joint_positions)
        self.viz.display(q)
        if self.visualize_collisions:
            # ensure collision geometry poses are updated
            self.viz.displayCollisions(q)

    def display_heightmap(self, points_xyz, color=[0.2, 0.6, 1.0, 1.0], radius=0.01):
        """
        Render height samples as a point cloud or small spheres.
        points_xyz: (N,3) in world frame.
        """
        if points_xyz is None or len(points_xyz) == 0:
            return
        vis = getattr(self.viz, "viewer", None) or getattr(self.viz, "vis", None)
        if vis is None:
            return
        for i, p in enumerate(points_xyz):
            T = tf.translation_matrix([p[0], p[1], p[2]])
            handle = vis[f"heightmap/{i}"]
            handle.set_object(g.Sphere(radius), g.MeshLambertMaterial(color=color))
            handle.set_transform(T)
