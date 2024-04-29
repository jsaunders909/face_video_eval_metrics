import sys
import cv2
import os
import subprocess
import shutil
from tqdm import tqdm
import numpy as np


def videos_to_images(video, output_dir, rand=False):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video)
    N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    print('Writing frames to disk...')
    for i in tqdm(range(N_FRAMES)):
        ret, frame = cap.read()
        if not ret:
            break
        
        if rand:
            frame = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
        
        frame = cv2.resize(frame, (256, 256))
        cv2.imwrite(f'{output_dir}/{frame_idx}.png', frame)
        frame_idx += 1
    cap.release()
        
def calc_fid(real_img_dir, fake_image_dir):
    cmd = f"python -m pytorch_fid {real_img_dir} {fake_image_dir}"
    subprocess.run(cmd, shell=True)
    

def get_FID_from_videos(real_video, fake_video):
    
    temp_dir = './temp'
    
    temp_real_image_dir = os.path.join(temp_dir, 'real_images')
    temp_fake_image_dir = os.path.join(temp_dir, 'fake_images')
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    videos_to_images(real_video, temp_real_image_dir)
    videos_to_images(fake_video, temp_fake_image_dir, rand=True)
    
    calc_fid(temp_real_image_dir, temp_fake_image_dir)
    
    print('Cleaning up...')
    
    shutil.rmtree(temp_real_image_dir)
    shutil.rmtree(temp_fake_image_dir)

if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--real_video', type=str, required=True)
    parser.add_argument('-f', '--fake_video', type=str, required=True)
    args = parser.parse_args()
    
    print(get_FID_from_videos(args.real_video, args.fake_video))
    