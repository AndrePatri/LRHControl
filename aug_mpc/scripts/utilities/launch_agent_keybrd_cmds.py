#!/usr/bin/env python3
import argparse

if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--actions',action='store_true', help='whether to send agent actions instead of refs')
    parser.add_argument('--ns', type=str, help='Namespace to be used for shared memory')
    parser.add_argument('--agent_refs_world',action='store_true', 
        help='whether to set the agent ref in world frame (it will be internally adjucted to base frame)')
    parser.add_argument('--cmapping', type=str, help='contact mapping to, respectively, keys 7 9 1 and 3', default="0;1;2;3")
    parser.add_argument('--env_idx', type=int,default=None)
    parser.add_argument('--from_stdin', action='store_true')

    parser.add_argument('--joy', action='store_true')
    parser.add_argument("--bind", default="0.0.0.0:5556", help="JoyListenerZMQ bind address (host:port). Default 0.0.0.0:5556")
    parser.add_argument("--topic", default="joy", help="Topic to subscribe to (default 'joy')")
    parser.add_argument("--poll-interval", type=float, default=0.01, help="Poll interval seconds (default 0.01)")
    parser.add_argument('--add_remote_exit', action='store_true', help='create a client to the remote exit flag')
    
    args = parser.parse_args()
    
    keyb_cmds=None
    if args.actions:
        if not args.joy:
            from aug_mpc.utils.keyboard_cmds import AgentActionsFromKeyboard

            keyb_cmds = AgentActionsFromKeyboard(namespace=args.ns, 
                                verbose=True,
                                contact_mapping=args.cmapping,
                                env_idx=args.env_idx)
            
            keyb_cmds.run(read_from_stdin=args.from_stdin)
        else:
            raise NotImplemented("Agent actions from joy not implemented yet")

    else:

        if not args.joy:
            from aug_mpc.utils.keyboard_cmds import AgentRefsFromKeyboard
            

            keyb_cmds = AgentRefsFromKeyboard(namespace=args.ns, 
                                verbose=True,
                                agent_refs_world=args.agent_refs_world,
                                env_idx=args.env_idx)
            
        
            keyb_cmds.run(read_from_stdin=args.from_stdin)
        
        else:

            from aug_mpc.utils.joy_cmds import AgentRefsFromJoy
            from EigenIPC.PyEigenIPC import VLevel, dtype, Journal, LogType

            joy_cmds = AgentRefsFromJoy(namespace=args.ns, 
                                verbose=True,
                                agent_refs_world=args.agent_refs_world,
                                env_idx=args.env_idx)
            
            # optional safety flag wrapper
            safety_flag = None
            if args.add_remote_exit:
                from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import SharedTWrapper
                safety_flag = SharedTWrapper(namespace = args.ns,
                        basename = "IbridoRemoteEnvExitFlag",
                        is_server = False,
                        verbose = True,
                        vlevel = VLevel.V2,
                        safe = True,
                        dtype=dtype.Bool)
                safety_flag.run()

                # callback will be called each loop as callback(joy_listener, callback_arg)
                def safety_callback(joy_listener, safety_flag_wrapper):
                    """
                    Read the joystick menu/guide/back/start button (back_start_home[2]) and,
                    if pressed, set the remote exit flag in the provided safety_flag_wrapper.
                    """
                    # read current back/start/home array from listener
                    cur_bsh = joy_listener.back_start_home.copy()
                    if bool(cur_bsh[2]) and (safety_flag_wrapper is not None):
                        Journal.log("launch_agent_KEYBRD_CMDS", "[]", "triggering remote exit flag", LogType.WARN)
                        mirror = safety_flag_wrapper.get_numpy_mirror()
                        mirror.flat[0] = True
                        safety_flag_wrapper.synch_all(read=False, retry=True)
                        return False
                    else:
                        return True
    
                # run with callback and ensure cleanup
                joy_cmds.run(connect=args.bind, topic=args.topic, poll_interval=args.poll_interval,
                                callback=safety_callback, callback_arg=safety_flag)
                safety_flag.close()
            else:
                # run without safety callback
                joy_cmds.run(connect=args.bind, topic=args.topic, poll_interval=args.poll_interval,
                             callback=None, callback_arg=None)
