import glob
import os
import sys
from tqdm import tqdm
import shutil
import time
from contextlib import contextmanager

# to suppress output
# from https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(script_dir, 'configs')
    VULCAN_dir = os.path.expanduser('~/git/VULCAN/')

    # Change the current working directory
    os.chdir(VULCAN_dir)
    sys.path.append(VULCAN_dir)

    config_files = glob.glob(os.path.join(configs_dir, '*.py'))

    # TODO: VULCAN not parallelizing on ALICE, so maybe parallelize here.
    print('running VULCAN for configs...')
    for config_file in tqdm(config_files):
        # copy config file to VULCAN directory
        start = time.time()
        shutil.copyfile(config_file, os.path.join(VULCAN_dir, 'vulcan_cfg.py'))

        # run VULCAN
        with suppress_stdout():
            exec(open(os.path.join(VULCAN_dir, "vulcan.py")).read())

        print(f'one run takes: {time.time() - start} s')


if __name__ == "__main__":
    main()
