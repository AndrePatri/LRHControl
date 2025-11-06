
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
    parser.add_argument("--connect", default="localhost:5556", help="Publisher address to connect to (host:port). Default localhost:5556")
    parser.add_argument("--topic", default="joy", help="Topic to subscribe to (default 'joy')")
    parser.add_argument("--poll-interval", type=float, default=0.01, help="Poll interval seconds (default 0.01)")
    
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

            joy_cmds = AgentRefsFromJoy(namespace=args.ns, 
                                verbose=True,
                                agent_refs_world=args.agent_refs_world,
                                env_idx=args.env_idx)
            
            joy_cmds.run(args.connect, args.topic, args.poll_interval)