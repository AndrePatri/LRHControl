import socket
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat
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
        web_host: str = None,       # what you share with users (LAN IP / hostname)
        web_port: int = 7000,
        web_bind_host: str = None,  # actual bind address for MeshCat's web server
        visualize_collisions: bool = False,
        robot_joint_names=None,
        open_browser: bool = False, # <-- default False so remote use doesn't pop a tab
    ):
        self.urdf_path = urdf_path
        self.mesh_dir = mesh_dir
        self.base_link_name = base_link_name
        self.visualize_collisions = visualize_collisions
        self.robot_joint_names = list(robot_joint_names) if robot_joint_names is not None else []

        # Normalize hosts/ports; never allow 0.0.0.0 on the *client* side.
        self.zmq_host = "127.0.0.1" if zmq_host in ("0.0.0.0", "", None) else zmq_host
        self.zmq_port = int(zmq_port)

        # Public-facing URL (shown to users)
        self.web_host = (web_host or self.zmq_host)
        self.web_host = "127.0.0.1" if self.web_host in ("0.0.0.0", "", None) else self.web_host
        self.web_port = int(web_port)

        # Actual bind address for the internal server (can be 0.0.0.0)
        self.web_bind_host = web_bind_host or self.web_host

        # Build pinocchio model + geometry (free-flyer root)
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, mesh_dir, pin.JointModelFreeFlyer()
        )
        self.data = self.model.createData()

        # Pinocchio Meshcat visualizer wrapper
        self.viz = MeshcatVisualizer(self.model, self.collision_model, self.visual_model)

        # Try to connect to an existing meshcat server quickly, else start one that binds for LAN.
        mc_viewer = self._connect_or_start_meshcat()

        # Initialize viewer and load models
        # IMPORTANT: open=False so we don't try to open a local browser on a headless/remote box.
        self.viz.initViewer(viewer=mc_viewer, open=False, loadModel=False)
        self.viz.loadViewerModel()

        # Display initial pose (home if available, else neutral)
        if hasattr(self.model, "referenceConfigurations") and "home" in self.model.referenceConfigurations:
            q0 = self.model.referenceConfigurations["home"]
        else:
            q0 = pin.neutral(self.model)
        self.viz.display(q0)

        # Build joint name map: model (after free-flyer) -> RobotState joint ordering
        self._joint_index_map, self._joint_valid_mask, self._actuated_joint_ids, self._joint_nqs = \
            self._build_joint_index_map()

        # Tell the user where to point the browser
        print(f"[meshcat] Open in your browser: http://{self.web_host}:{self.web_port}")

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
        zmq_url = f"tcp://{self.zmq_host}:{self.zmq_port}"

        print(f"[meshcat] Target ZMQ: {zmq_url}")

        # Fast path: connect if an external server is already listening (e.g., you ran `meshcat-server` yourself).
        if self._tcp_listening(self.zmq_host, self.zmq_port):
            try:
                return meshcat.Visualizer(zmq_url=zmq_url)
            except Exception as e:
                print(f"[meshcat] Warning: connection to existing server failed: {e}")

        # Fallback: start our own server and make it reachable from the LAN if requested.
        # Use server_args to control bind host/port for the web server and ZMQ.
        server_args = [
            "--host", str(self.web_bind_host),          # 0.0.0.0 to allow LAN access
            "--port", str(self.web_port),               # web (HTTP/WebSocket) port
            "--zmq-url", zmq_url                        # ensure the ZMQ endpoint is predictable
        ]

        print("[meshcat] No ZMQ server detected; starting an internal MeshCat server")
        print(f"[meshcat] Web bind: http://{self.web_bind_host}:{self.web_port}  (advertised as http://{self.web_host}:{self.web_port})")
        try:
            vis = meshcat.Visualizer(zmq_url=None, server_args=server_args)
            return vis
        except Exception as e:
            print(f"[meshcat] Warning: failed to start meshcat server: {e}")
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
            joint_nqs: list of nq for each joint id.
        """
        model_joint_names = []
        model_joint_ids = []
        model_joint_nqs = []
        # Pinocchio model.names includes "universe" (0) and the free-flyer root (1); skip both.
        for jidx in range(2, self.model.njoints):
            model_joint_names.append(self.model.names[jidx])
            model_joint_ids.append(jidx)
            # Prefer per-joint nq if available; otherwise use model.nqs[jidx]
            nq = getattr(self.model.joints[jidx], "nq", None)
            if nq is None:
                nq = self.model.nqs[jidx]
            model_joint_nqs.append(int(nq))

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
        return mapping, valid_mask, model_joint_ids, model_joint_nqs

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

            # destination slice for this joint in q
            qpos = self.model.idx_qs[joint_id]
            # robustly get joint nq
            try:
                nq = self.model.joints[joint_id].nq
            except Exception:
                nq = self._joint_nqs[k]

            # Copy as many values as the joint actually needs.
            # If your upstream RobotState uses 1 value per joint, ensure it matches nq=1 joints.
            end_src = min(src + nq, jp.size)
            span = end_src - src
            if span > 0:
                q[qpos:qpos + span] = jp[src:end_src]

        # Normalize configuration (wrap revolute/continuous joints, renormalize base quat)
        q_norm = q.copy()
        pin.normalize(self.model, q_norm)
        return q_norm

    def display(self, base_pos_w, base_quat_wxyz, joint_positions):
        q = self.pinocchio_q(base_pos_w, base_quat_wxyz, joint_positions)
        self.viz.display(q)
        if self.visualize_collisions:
            try:
                self.viz.displayCollisions(q)  # MeshcatVisualizer uses positional arg
            except TypeError:
                # Some versions may allow keyword; keep a gentle fallback
                self.viz.displayCollisions(q=q)

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
