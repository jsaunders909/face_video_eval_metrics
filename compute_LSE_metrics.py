import os
import sys
import argparse
import subprocess
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, 'evaluation_metrics', 'LSE_d', 'syncnet_python')

os.chdir(dir_path)
sys.path.append(dir_path)

def get_LSE_metrics(inputs):

    cmd = f'bash calculate_scores_real_videos.sh {inputs}'
    subprocess.run(cmd, shell=True)


    with open('all_scores.txt', 'r') as f:
        lines = f.readlines()
        
    ds, cs, offsets = [], [], []
    for line in lines:
        _, d, c, offset, *_ = line.split(' ')
        ds.append(float(d))
        cs.append(float(c))
        offsets.append(int(offset))

    return {
        'LSE_d': np.mean(ds),
        'LSE_c': np.mean(cs),
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', type=str, required=True)
    args = parser.parse_args()

    print(get_LSE_metrics(args.inputs))
