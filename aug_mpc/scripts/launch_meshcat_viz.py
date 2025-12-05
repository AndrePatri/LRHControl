#!/usr/bin/env python
import argparse
import time
import numpy as np

from EigenIPC.PyEigenIPC import VLevel

from aug_mpc.utils.rhc_meshcat.meshcat_publisher import RhcMeshcatPublisher
from mpc_hive.utilities.shared_data.rhc_data import RobotState


def main():
    parser = argparse.ArgumentParser(description="Launch Meshcat viz for RobotState.")
    parser.add_argument("--ns", type=str, required=True, help="Shared memory namespace")
    parser.add_argument("--urdf_path", type=str, required=True, help="Path to robot URDF")
    parser.add_argument("--mesh_dir", type=str, default=None, help="Meshes root (optional)")
    parser.add_argument("--base_link", type=str, default="base_link", help="Base link name")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Meshcat host")
    parser.add_argument("--port", type=int, default=7000, help="Meshcat port")
    parser.add_argument("--rate", type=float, default=20.0, help="Publish rate (Hz)")
    parser.add_argument("--env_idx", type=int, default=0, help="Robot/env index to visualize")
    parser.add_argument("--show_collisions", action="store_true", help="Show collision geometries")
    args = parser.parse_args()

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
    jnt_names = getattr(rs.jnts_state, "jnt_names", None) or rs.jnts_state.jnt_names

    pub = RhcMeshcatPublisher(
        urdf_path=args.urdf_path,
        mesh_dir=args.mesh_dir,
        base_link_name=args.base_link,
        host=args.host,
        port=args.port,
        visualize_collisions=args.show_collisions,
        robot_joint_names=jnt_names,
    )

    dt = 1.0 / args.rate
    try:
        while True:
            rs.root_state.synch_all(read=True, retry=True, row_index=args.env_idx, row_index_view=args.env_idx)
            rs.jnts_state.synch_all(read=True, retry=True, row_index=args.env_idx, row_index_view=args.env_idx)

            base_p = rs.root_state.get(data_type="p", gpu=False)[args.env_idx, :3]
            base_q = rs.root_state.get(data_type="q", gpu=False)[args.env_idx, :4]  # wxyz
            jpos = rs.jnts_state.get(data_type="q", gpu=False)[args.env_idx, :]

            pub.display(np.asarray(base_p), np.asarray(base_q), np.asarray(jpos))
            time.sleep(dt)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
