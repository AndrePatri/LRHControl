#!/usr/bin/env python3
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run keyboard or joystick refs writer.")
    parser.add_argument('--ns', type=str, required=True, help='Namespace to be used for shared memory')
    parser.add_argument('--cmapping', type=str, help='contact mapping to, respectively, keys 7 9 1 and 3', default="0;1;2;3")
    parser.add_argument('--env_idx', type=int, default=None)
    parser.add_argument('--from_stdin', action='store_true',
                        help='If set and running keyboard mode, read key events from stdin-based listener')
    # New joystick options:
    parser.add_argument('--joy', action='store_true', help='Run in joystick mode (RefsFromJoy)')
    parser.add_argument('--connect', type=str, default='localhost:5556', help='JoyListenerZMQ connect address (host:port)')
    parser.add_argument('--topic', type=str, default='joy', help='ZeroMQ topic for joystick messages')
    parser.add_argument('--poll-interval', type=float, default=0.01, help='Joy listener poll interval (s)')
    parser.add_argument('--hold-time', type=float, default=0.15, help='Hold time for toggles (seconds) when using joystick')

    parser.add_argument('--add_remote_exit', action='store_true', help='When in joystick mode, create a client to the remote exit flag')

    args = parser.parse_args()

    from mpc_hive.utilities.shared_data.rhc_data import RhcRefs
    from EigenIPC.PyEigenIPC import VLevel, dtype, Journal, LogType

    # create shared refs (same for both modes)
    shared_refs = RhcRefs(namespace=args.ns,
                          is_server=False,
                          safe=False,
                          verbose=True,
                          vlevel=VLevel.V2)

    # import both classes (assumes they are available at these paths)
    from mpc_hive.utilities.keyboard_cmds import RefsFromKeyboard
    from mpc_hive.utilities.joy_cmds import RefsFromJoy

    # instantiate the appropriate controller
    if args.joy:
        # joystick-driven
        joy_cmds = RefsFromJoy(namespace=args.ns,
                               shared_refs=shared_refs,
                               verbose=True,
                               agent_refs_world=True,
                               env_idx=args.env_idx,
                               hold_time=args.hold_time)

        # apply contact mapping (parsed from string)
        try:
            parsed = [int(x) for x in args.cmapping.strip().split(';') if x != '']
            if len(parsed) == 4:
                joy_cmds._contact_mapping = parsed
        except Exception:
            # keep default if parse fails
            pass

        # If requested, create a remote-exit safety_flag and provide a callback
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

            # callback that will be called from inside RefsFromJoy.run each loop.
            # signature: callback(joy_listener, safety_flag)
            def safety_callback(joy_listener, safety_flag_wrapper):
                """
                Read the joystick menu/guide button and, if pressed, set the remote exit flag.
                This callback does NOT return any value (the run loop ignores return values).
                """
                # read current back/start/home from the live joy_listener object
                cur_back_start_home = joy_listener.back_start_home.copy()
        
                # menu button is back_start_home[2] per mapping
                exit_pressed = bool(cur_back_start_home[2])
                if exit_pressed and (safety_flag_wrapper is not None):
                    Journal.log("launch_rhc_keybrd_cmds", "[]", "triggering remote exit flag", LogType.WARN)
                    mirror = safety_flag_wrapper.get_numpy_mirror()
                    mirror.flat[0] = True
                    safety_flag_wrapper.synch_all(read=False, retry=True)
                    return False
                else:
                    return True
        
            # Run with callback and ensure cleanup
            joy_cmds.run(connect=args.connect, topic=args.topic, poll_interval=args.poll_interval,
                            callback=safety_callback, callback_arg=safety_flag)
            safety_flag.close()
        else:
            # No remote-exit callback; pass callback_arg=None
            joy_cmds.run(connect=args.connect, topic=args.topic, poll_interval=args.poll_interval,
                         callback=None, callback_arg=None)

    else:
        # keyboard-driven (original behavior)
        keyb_cmds = RefsFromKeyboard(namespace=args.ns,
                                     shared_refs=shared_refs,
                                     verbose=True,
                                     contact_mapping=args.cmapping,
                                     env_idx=args.env_idx)

        keyb_cmds.run(read_from_stdin=args.from_stdin)

