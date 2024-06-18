from lrhc_control.utils.keyboard_cmds import AgentRefsFromKeyboard
import argparse

if __name__ == "__main__":  

    parser = argparse.ArgumentParser(description="Set CPU affinity for the script.")
    parser.add_argument('--ns', type=str, help='Namespace to be used for shared memory')
    args = parser.parse_args()
    
    keyb_cmds = AgentRefsFromKeyboard(namespace=args.ns, 
                            verbose=True)

    keyb_cmds.run()