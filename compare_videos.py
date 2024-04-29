import cv2
import os
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Combine two videos side by side with differnce')
parser.add_argument('-r', '--real', type=str, help='Path to real video')
parser.add_argument('-f', '--fake', type=str, help='Path to fake video')
parser.add_argument('-o', '--output', type=str, help='Path to output video', default='test.mp4')
args = parser.parse_args()

real_cap = cv2.VideoCapture(args.real)
fake_cap = cv2.VideoCapture(args.fake)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = real_cap.get(cv2.CAP_PROP_FPS)
width = int(real_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(real_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(args.output, fourcc, fps, (width * 3, height))

while real_cap.isOpened() and fake_cap.isOpened():
    real_ret, real_frame = real_cap.read()
    fake_ret, fake_frame = fake_cap.read()
    
    if not real_ret or not fake_ret:
        break
    
    fake_frame = cv2.resize(fake_frame, (width, height))
    diff = cv2.absdiff(real_frame, fake_frame)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    
    out_frame = np.hstack((real_frame, diff, fake_frame))
    out.write(out_frame)
