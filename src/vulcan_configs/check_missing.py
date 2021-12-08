import os
import glob
from pathlib import Path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MRP_dir = str(Path(script_dir).parents[1])
    std_output = os.path.join(MRP_dir, 'data/vulcan_output_small/std_output')
    output_dir = os.path.join(MRP_dir, 'data/vulcan_output_small/')

    std_out_files = glob.glob(os.path.join(std_output, '*.txt'))
    std_out_file_names = [f'{os.path.basename(f)[11:-4]}' for f in std_out_files]
    output_files = glob.glob(os.path.join(output_dir, '*.vul'))
    output_file_names = [f'{os.path.basename(f)[7:-4]}' for f in output_files]

    print(f'{len(std_out_files) = }')
    print(f'{len(output_files) = }')

    for i, std_out_file_name in enumerate(std_out_file_names):
        if std_out_file_name not in output_file_names:
            print(f'file {i}: {std_out_file_name}')


if __name__ == "__main__":
    main()
