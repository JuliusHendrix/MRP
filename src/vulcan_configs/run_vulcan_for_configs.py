import glob
import os
from tqdm import tqdm
import importlib.util
import shutil


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(script_dir, 'configs')
    VULCAN_dir = os.path.expanduser('~/git/VULCAN/')

    # Change the current working directory
    os.chdir(VULCAN_dir)

    config_files = glob.glob(os.path.join(configs_dir, '*.py'))

    print('running VULCAN for configs...')
    for config_file in tqdm(config_files):
        # copy config file to VULCAN directory
        shutil.copyfile(config_file, os.path.join(VULCAN_dir, 'vulcan_cfg.py'))

        # run VULCAN
        spec = importlib.util.spec_from_file_location("module.name", os.path.join(VULCAN_dir, "vulcan.py"))
        vulcan_py = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vulcan_py)
        break


if __name__ == "__main__":
    main()
