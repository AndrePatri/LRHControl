import numpy as np
import pinocchio as pin
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
        self.data = self.model.createData()
        self.visual_data = self.visual_model.createData()
        self.collision_data = self.collision_model.createData()

        # Meshcat visualizer
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)
        # Connect/start server (Meshcat server should be running; open browser if possible)
        self.viz.initViewer(open=True)
        # Load geometry (API varies by pinocchio version; use minimal signature)
        self.viz.loadViewerModel()
        # Display initial pose (fallback to zero/neutral if no "home")
        if hasattr(self.model, "referenceConfigurations") and "home" in self.model.referenceConfigurations:
            q0 = self.model.referenceConfigurations["home"]
        else:
            q0 = pin.neutral(self.model)
        self.viz.display(q0)

        # build joint name map: model (after free-flyer) -> RobotState joint ordering
        self._joint_index_map = self._build_joint_index_map()

    def _build_joint_index_map(self):
        """
        Map pinocchio joint order (after free-flyer) to incoming RobotState joint order.
        Returns list of indices into robot_joint_names; -1 if not found.
        """
        model_joint_names = []
        # pin model.names includes "universe", "root_joint" (free-flyer), then actuated joints
        for jidx in range(2, self.model.njoints):  # skip universe (0) and free-flyer (1)
            model_joint_names.append(self.model.names[jidx])
        name_to_idx = {n: i for i, n in enumerate(self.robot_joint_names)}
        mapping = []
        unmatched = []
        for mj in model_joint_names:
            src = name_to_idx.get(mj, -1)
            mapping.append(src)
            if src < 0:
                unmatched.append(mj)
        if unmatched:
            print(f"[meshcat] Warning: {len(unmatched)} joints missing in RobotState: {unmatched[:5]}{'...' if len(unmatched)>5 else ''}")
        return mapping

    def pinocchio_q(self, base_pos_w, base_quat_wxyz, joint_positions):
        """Assemble pinocchio configuration vector."""
        q = np.zeros(self.model.nq)
        q[0:3] = base_pos_w
        # pin expects xyzw
        q[3:7] = np.array([base_quat_wxyz[1], base_quat_wxyz[2], base_quat_wxyz[3], base_quat_wxyz[0]])
        # map joints (truncate or pad to match model)
        jp = np.asarray(joint_positions).reshape(-1)
        for k, src_idx in enumerate(self._joint_index_map):
            if src_idx >= 0 and src_idx < jp.size:
                q[7 + k] = jp[src_idx]
        return q

    def display(self, base_pos_w, base_quat_wxyz, joint_positions):
        q = self.pinocchio_q(base_pos_w, base_quat_wxyz, joint_positions)
        self.viz.display(q)

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
