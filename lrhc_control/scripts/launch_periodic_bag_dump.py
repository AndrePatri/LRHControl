import argparse

from lrhc_control.utils.bag_dumper import RosBagDumper
import time 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--ros_bridge_dt', type=float, default=0.01, help='Update interval in seconds for ros topics, default is 0.01')
    parser.add_argument('--bag_sdt', type=float, default=90.0, help='sim time dt over which each bag will run')
    parser.add_argument('--abort_wallmin', type=float, default=5.0, help='abort bridge if no response wihtin this timeout')
    parser.add_argument('--dump_dt_min', type=float, default=60.0, help='wait these min before dumping a new bag')

    parser.add_argument('--ns', type=str, help='Namespace to be used for cluster shared memory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode, default is False')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False, help='Enable verbose mode, default is True')
    parser.add_argument('--ros2', action=argparse.BooleanOptionalAction, default=True, help='Use ROS 2')
    parser.add_argument('--with_agent_refs', action=argparse.BooleanOptionalAction, default=False, help='also forward agent refs to rhcviz')
    parser.add_argument('--rhc_refs_in_h_frame', type=bool, default=True, help='set to true if rhc refs are \
                        specified in the horizontal frame')
    parser.add_argument('--agent_refs_in_h_frame', type=bool, default=False, help='set to true if agent refs are \
                        specified in the horizontal frame')
    parser.add_argument('--env_idx', type=int, help='env index of which data is to be published', default=0)
    parser.add_argument('--srdf_path', type=str, help='path to SRDF path specifying homing configuration, to be used for missing joints', default=None)
    parser.add_argument('--dump_path', type=str, default="/tmp", help='where bag will be dumped')
    parser.add_argument('--use_shared_drop_dir', action=argparse.BooleanOptionalAction, default=True, 
        help='if true use the shared drop dir to drop the data where all the other training data is dropeer')

    args = parser.parse_args()

    training_done=False
    while True:
        
        bag_dumper=RosBagDumper(ns=args.ns,
            ros_bridge_dt=args.ros_bridge_dt,
            bag_sdt=args.bag_sdt,
            debug=args.debug,
            verbose=args.verbose,
            dump_path=args.dump_path,
            use_shared_drop_dir=args.use_shared_drop_dir,
            ros2=args.ros2,
            env_idx=args.env_idx,
            srdf_path=args.srdf_path,
            abort_wallmin=args.abort_wallmin,
            with_agent_refs=args.with_agent_refs,
            rhc_refs_in_h_frame=args.rhc_refs_in_h_frame,
            agent_refs_in_h_frame=args.agent_refs_in_h_frame)
        training_done=bag_dumper.training_done()
        if training_done:
            bag_dumper.close()
            break

        start_time=time.monotonic() 
        bag_dumper.run()
        bag_dumper.close()

        elapsed_min=(time.monotonic()-start_time)*1.0/60.0
        remaining_min=args.dump_dt_min-elapsed_min
        if remaining_min>0.0: # wait 
            time.sleep(remaining_min*60.0)

        