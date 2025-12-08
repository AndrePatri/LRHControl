import socket
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

    Apptainer/Singularity notes:
      - If the Meshcat server runs inside the same container, connect to 127.0.0.1.
      - If it runs on the host, use the host's IP or '127.0.0.1' with --net=host.
      - Do not pass '0.0.0.0' as a client address; it's a bind address and not routable.
      - The browser opens the *web* port (default 7000), not the ZMQ port.
    """

    def __init__(
        self,
        urdf_path: str,
        mesh_dir: str = None,
        base_link_name: str = "base_link",
        zmq_host: str = "127.0.0.1",
        zmq_port: int = 6000,
        web_host: str = None,      # advertised URL host (what users open)
        web_port: int = 7000,
        web_bind_host: str = None, # actual bind address for Meshcat's web server
        visualize_collisions: bool = False,
        robot_joint_names=None,
    ):
        self.urdf_path = urdf_path
        self.mesh_dir = mesh_dir
        self.base_link_name = base_link_name
        self.visualize_collisions = visualize_collisions
        self.robot_joint_names = list(robot_joint_names) if robot_joint_names is not None else []

        # Normalize hosts/ports; never allow 0.0.0.0 on the client side.
        self.zmq_host = "127.0.0.1" if zmq_host in ("0.0.0.0", "", None) else zmq_host
        self.zmq_port = int(zmq_port)

        # What you print/share with users (may be a LAN IP if remote access)
        self.web_host = (web_host or self.zmq_host)
        self.web_host = "127.0.0.1" if self.web_host in ("0.0.0.0", "", None) else self.web_host
        self.web_port = int(web_port)

        # What the local server actually binds to (can legitimately be 0.0.0.0)
        self.web_bind_host = web_bind_host or self.web_host

        # Load pinocchio model + geometry (free-flyer root)
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, mesh_dir, pin.JointModelFreeFlyer()
        )
        self.data = self.model.createData()
        self.visual_data = self.visual_model.createData()
        self.collision_data = self.collision_model.createData()

        # Pinocchio Meshcat visualizer wrapper
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)

        # Try to connect to an existing meshcat server quickly.
        mc_viewer = self._connect_or_start_meshcat()

        # Initialize viewer and load models
        self.viz.initViewer(viewer=mc_viewer, open=True)
        self.viz.loadViewerModel()

        # Display initial pose (home if available, else neutral)
        if hasattr(self.model, "referenceConfigurations") and "home" in self.model.referenceConfigurations:
            q0 = self.model.referenceConfigurations["home"]
        else:
            q0 = pin.neutral(self.model)
        self.viz.display(q0)

        # Build joint name map: model (after free-flyer) -> RobotState joint ordering
        self._joint_index_map, self._joint_valid_mask, self._actuated_joint_ids = self._build_joint_index_map()

    # -------------------------------------------------------------------------
    # Meshcat connection helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _tcp_listening(host: str, port: int, timeout: float = 0.25) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _connect_or_start_meshcat(self):
        import meshcat
        zmq_url = f"tcp://{self.zmq_host}:{self.zmq_port}"
        advertised_web_url = f"http://{self.web_host}:{self.web_port}"

        print(f"[meshcat] Target ZMQ: {zmq_url}")
        print(f"[meshcat] Open in your browser: {advertised_web_url}")

        # Fast path: connect if an external server is already listening.
        if self._tcp_listening(self.zmq_host, self.zmq_port):
            try:
                return meshcat.Visualizer(zmq_url=zmq_url)
            except Exception as e:
                print(f"[meshcat] Warning: connection to existing server failed: {e}")

        # Fallback: start our own server without server_args (avoid list.append crash).
        print("[meshcat] No ZMQ server detected; starting an internal Meshcat server with defaults")
        try:
            vis = meshcat.Visualizer(zmq_url=None)  # <-- no server_args, let meshcat decide
            # Meshcat prints the actual URL itself; also try to show it here if available.
            web_url_attr = getattr(getattr(vis, "window", None), "web_url", None)
            if web_url_attr:
                print(f"[meshcat] Internal server web URL: {web_url_attr}")
            else:
                # Fallback to the advertised URL (often correct with defaults)
                print(f"[meshcat] You can open the visualizer at: {advertised_web_url}")
            return vis
        except Exception as ee:
            print(f"[meshcat] Warning: failed to start meshcat server: {ee}")
        return None

    # -------------------------------------------------------------------------
    # Joint index mapping and configuration assembly
    # -------------------------------------------------------------------------
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
        # Pinocchio model.names includes "universe", "root_joint" (free-flyer), then actuated joints
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
            head = unmatched[:5]
            print(f"[meshcat] Warning: {len(unmatched)} joints missing in RobotState: {head}{'...' if len(unmatched) > 5 else ''}")

        mapping = np.array(mapping_list, dtype=int)
        valid_mask = mapping >= 0
        return mapping, valid_mask, model_joint_ids

    def pinocchio_q(self, base_pos_w, base_quat_wxyz, joint_positions):
        """Assemble pinocchio configuration vector."""
        q = np.zeros(self.model.nq)
        q[0:3] = base_pos_w
        # pin expects xyzw
        q[3:7] = np.array([
            base_quat_wxyz[1],
            base_quat_wxyz[2],
            base_quat_wxyz[3],
            base_quat_wxyz[0],
        ])

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
                end = qpos + nq
                span = min(nq, max(0, 7 + jp.size - qpos))
                if span > 0:
                    q[qpos:qpos + span] = jp[src:src + span]

        # Normalize configuration (wrap revolute/continuous joints, renormalize base quat)
        q_norm = q.copy()
        pin.normalize(self.model, q_norm)
        return q_norm

    def display(self, base_pos_w, base_quat_wxyz, joint_positions):
        q = self.pinocchio_q(base_pos_w, base_quat_wxyz, joint_positions)
        self.viz.display(q)
        if self.visualize_collisions:
            # ensure collision geometry poses are updated; use keyword to avoid signature confusion
            try:
                self.viz.displayCollisions(q=q.tolist())
            except TypeError:
                # fallback to positional if keyword unsupported
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

