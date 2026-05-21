import argparse

from aug_mpc.utils.description_adapter import prepare_xrdf_input


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare source-tree xacro descriptions for AugMPC.")
    parser.add_argument("xacro_path", help="URDF/SRDF xacro path to adapt")
    parser.add_argument(
        "--dump_path",
        default="/tmp",
        help="Directory used for the temporary adapted package copy")
    args = parser.parse_args()

    adapted_path, package_map = prepare_xrdf_input(
        xacro_path=args.xacro_path,
        dump_path=args.dump_path)

    print(adapted_path)
    for package_name, package_path in sorted(package_map.items()):
        print(f"{package_name}: {package_path}")
