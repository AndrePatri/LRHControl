#!/usr/bin/env python
import argparse
import time
import numpy as np

from EigenIPC.PyEigenIPC import VLevel
from aug_mpc.utils.rhc_meshcat.meshcat_publisher import RhcMeshcatPublisher
from mpc_hive.utilities.shared_data.rhc_data import RobotState


def _safe_client_host(h: str) -> str:
    # Never use 0.0.0.0 for client connections
    return "127.0.0.1" if h in (None, "", "0.0.0.0") else h


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Meshcat viz for RobotState.")

    # Shared memory / robot model
    parser.add_argument("--ns", type=str, required=True, help="Shared memory namespace")
    parser.add_argument("--urdf_path", type=str, required=True, help="Path to robot URDF")
    parser.add_argument("--mesh_dir", type=str, default=None, help="Meshes root (optional)")
    parser.add_argument("--base_link", type=str, default="base_link", help="Base link name")

    # Explicit endpoints
    parser.add_argument("--zmq-host", type=str, default="127.0.0.1",
                        help="Meshcat ZMQ host (client connect)")
    parser.add_argument("--zmq-port", type=int, default=6000,
                        help="Meshcat ZMQ port")
    parser.add_argument("--web-host", type=str, default=None,
                        help="Meshcat web host for browser (default: zmq-host)")
    parser.add_argument("--web-port", type=int, default=7000,
                        help="Meshcat web port (browser)")

    # Actual web bind address (can be 0.0.0.0 for LAN access)
    parser.add_argument("--web-bind", type=str, default=None,
                        help="Web server bind address (use 0.0.0.0 for LAN; default: web-host)")

    # Backward-compat flags (mapped below)
    parser.add_argument("--host", type=str, default=None,
                        help="[Deprecated] Single 'host' used previously")
    parser.add_argument("--port", type=int, default=None,
                        help="[Deprecated] Single 'port' used previously")

    # Viz behavior
    parser.add_argument("--rate", type=float, default=20.0, help="Publish rate (Hz)")
    parser.add_argument("--env_idx", type=int, default=0, help="Robot/env index to visualize")
    parser.add_argument("--show_collisions", action="store_true", help="Show collision geometries")

    args = parser.parse_args()

    # Map deprecated flags if the new ones weren't explicitly set
    if args.host is not None and (args.zmq_host == "127.0.0.1"):
        args.zmq_host = args.host
    if args.port is not None and (args.zmq_port == 6000 and args.web_port == 7000):
        # Historically there was one port; treat it as the web port users open
        args.web_port = args.port

    # Finalize hosts safely for client connects
    args.zmq_host = _safe_client_host(args.zmq_host)
    args.web_host = _safe_client_host(args.web_host if args.web_host is not None else args.zmq_host)

    # web-bind can legitimately be 0.0.0.0 (do not rewrite)
    args.web_bind = args.web_bind if args.web_bind is not None else args.web_host

    return args


def main():
    args = parse_args()

    rs = RobotState(
        namespace=args.ns,
        is_server=False,
        safe=False,
        verbose=False,
        vlevel=VLevel.V1,
        with_gpu_mirror=False,
        with_torch_view=False,
    )
    rs.run()

    # Joint name list used to map RobotState ordering -> Pinocchio model ordering
    jnt_names = rs.jnts_state.jnt_names

    # Instantiate the publisher with explicit endpoints
    pub = RhcMeshcatPublisher(
        urdf_path=args.urdf_path,
        mesh_dir=args.mesh_dir,
        base_link_name=args.base_link,
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        web_host=args.web_host,       # advertised URL host for printing
        web_port=args.web_port,
        web_bind_host=args.web_bind,  # actual bind address for the web server (can be 0.0.0.0)
        visualize_collisions=args.show_collisions,
        robot_joint_names=jnt_names,
    )

    print(
        "[meshcat] Visualization ready.\n"
        f"  ZMQ endpoint: tcp://{args.zmq_host}:{args.zmq_port}\n"
        f"  Browser URL : http://{args.web_host}:{args.web_port}\n"
        f"  Bound web   : {args.web_bind}:{args.web_port}\n"
        "Tip: Use --web-bind 0.0.0.0 for LAN access and set --web-host to the machine's LAN IP."
    )

    dt = max(1e-6, 1.0 / float(args.rate))
    try:
        while True:
            # Sync the selected environment row
            rs.root_state.synch_all(read=True, retry=True, row_index=args.env_idx, row_index_view=args.env_idx)
            rs.jnts_state.synch_all(read=True, retry=True, row_index=args.env_idx, row_index_view=args.env_idx)

            base_p = rs.root_state.get(data_type="p", gpu=False)[args.env_idx, :3]
            base_q = rs.root_state.get(data_type="q", gpu=False)[args.env_idx, :4]  # wxyz
            jpos  = rs.jnts_state.get(data_type="q", gpu=False)[args.env_idx, :]

            pub.display(np.asarray(base_p), np.asarray(base_q), np.asarray(jpos))
            time.sleep(dt)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
