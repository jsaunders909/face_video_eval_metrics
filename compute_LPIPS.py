import sys
import cv2
import os
import subprocess
import shutil
from tqdm import tqdm
from glob import glob
import lpips

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
        
def calc_LPIPS(real_img_dir, fake_image_dir):
    
    lpips_sum, count = 0, 0
    files = os.listdir(fake_image_dir)
    
    loss_fn = lpips.LPIPS(net='alex').cuda()
    
    for file in tqdm(files):
        if '.png' in file:
            if os.path.exists(os.path.join(real_img_dir, file)):
                real_img = os.path.join(real_img_dir, file)
                fake_img = os.path.join(fake_image_dir, file)
                real_img = lpips.im2tensor(lpips.load_image(real_img)).cuda()
                fake_img = lpips.im2tensor(lpips.load_image(fake_img)).cuda()
                
                lpips_sum += loss_fn.forward(real_img, fake_img).sum().item()
                count += 1
    print('LPIPS: ', lpips_sum / count)
    return lpips_sum / count
    

def get_LPIPS_from_videos(real_video, fake_video):
    
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
    
    lpips = calc_LPIPS(temp_real_image_dir, temp_fake_image_dir)
    
    print('Cleaning up...')
    
    shutil.rmtree(temp_real_image_dir)
    shutil.rmtree(temp_fake_image_dir)
    
    return {'LPIPS': lpips}

if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--real_videos', type=str, required=False, default='/efs/D4E/experiments/video_only/real')
    parser.add_argument('-f', '--fake_videos', type=str, required=True)
    args = parser.parse_args()
    
    print(get_LPIPS_from_videos(args.real_videos, args.fake_videos))
    