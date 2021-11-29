import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import glob


def inspect_vul():
    vulcan_file_dir = os.path.expanduser('/data/vulcan_output_parallel/')

    vulcan_files = glob.glob(os.path.join(vulcan_file_dir, '*.vul'))

    vulcan_file = np.random.choice(vulcan_files)

    # extract data
    with open(vulcan_file, 'rb') as handle:
        data = pickle.load(handle)


    for key, value in data.items():
        print(key)
        for k, v in value.items():
            print(k)
        print('\n')

    ins_data = data['variable']['y_time']

    print(ins_data)
    print(ins_data.shape)
    print(type(ins_data))


def plot_times():
    # std_dir = os.path.expanduser('~/git/MRP/data/vulcan_output_sequential_bench/std_output/')
    std_dir = os.path.expanduser('/data/vulcan_output_parallel_4_bench/std_output/')
    std_files = glob.glob(os.path.join(std_dir, '*.txt'))

    # extract running times from files
    times = []
    for std_file in std_files:
        with open(std_file, 'r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1]
            words = last_line.split(sep=' ')
            times.append(float(words[-2]))

    plt.figure()
    times = np.array(times)/60.
    plt.hist(times, density=False, bins=int(len(times)/1.5), rwidth=0.9)
    # plt.xticks(np.arange(0, max(times) + 5, 5.0))
    plt.xlabel('time (hours)')
    plt.ylabel('count')
    plt.title(f'time per VULCAN run for {len(times)} configurations\naverage time = {round(np.mean(times),2)} hours')
    # plt.savefig('histogram sequential.png', dpi=600)
    plt.savefig('histogram parallel 4.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    inspect_vul()
    # plot_times()
