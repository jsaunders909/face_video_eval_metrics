# Create a visual showing the absolute difference between the real and generated video
import cv2
from tqdm import tqdm

import numpy as np

def main(args):
    
    in_video_real = cv2.VideoCapture(args.real)
    in_video_fake = cv2.VideoCapture(args.fake)
    width = int(in_video_real.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(in_video_real.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_video = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))
    
    n_frames = int(in_video_real.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(n_frames, int(in_video_fake.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    for i in tqdm(range(n_frames)):
        
        ret_real, frame_real = in_video_real.read()
        ret_fake, frame_fake = in_video_fake.read()
        
        if not ret_real or not ret_fake:
            break
        
        frame = np.abs(frame_real.astype(np.float32) - frame_fake.astype(np.float32))
        frame = frame.astype(np.uint8)
        cv2.imwrite('test.png', frame)
        
        out_video.write(frame)
        
    out_video.release()
        
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--real', type=str, required=True)
    parser.add_argument('-f', '--fake', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    
    args = parser.parse_args()
    
    main(args)