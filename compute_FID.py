import sys
import cv2
import os
import subprocess
import shutil
from tqdm import tqdm
from glob import glob


def videos_to_images(videos, output_dir):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frame_idx = 0
    for video in videos:
        cap = cv2.VideoCapture(video)
        N_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Writing {N_FRAMES} frames to disk...')
        for i in tqdm(range(N_FRAMES)):
            ret, frame = cap.read()
            if not ret:
                break
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
    
    real_videos = glob(os.path.join(real_video, '*.mp4'))
    fake_videos = glob(os.path.join(fake_video, '*.mp4'))
    
    real_videos_ = []
    for video in real_videos:
        name = os.path.basename(video).split('.')[0]
        if name in [os.path.basename(video).split('.')[0] for video in fake_videos]:
            real_videos_.append(video)
    real_videos = real_videos_
    
    print(f'Found {len(real_videos)} videos')
        
    videos_to_images(real_videos, temp_real_image_dir)
    videos_to_images(fake_videos, temp_fake_image_dir)
    
    calc_fid(temp_real_image_dir, temp_fake_image_dir)
    
    print('Cleaning up...')
    
    shutil.rmtree(temp_real_image_dir)
    shutil.rmtree(temp_fake_image_dir)

if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--real_videos', type=str, required=True)
    parser.add_argument('-f', '--fake_videos', type=str, required=True)
    args = parser.parse_args()
    
    try:
        print(get_FID_from_videos(args.real_videos, args.fake_videos))
    except ValueError as e:
        print(e)
        print('This error can be resolved by downgrading scipy to version 1.11.1 Run the following command: pip install scipy==1.1.0')