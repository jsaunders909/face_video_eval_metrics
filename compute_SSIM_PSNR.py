import PIL.Image as Image
import cv2
import os
from tqdm import tqdm
import numpy as np
from skimage.metrics import structural_similarity

import mediapipe as mp

class FaceDetector:
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection()
        
    def forward(self, image):
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections is None:
            return None
        return results.detections[0].location_data.relative_bounding_box

    def crop(self, image, bbox):
        h, w, _ = image.shape
        x1, y1, x2, y2 = bbox.xmin, bbox.ymin, bbox.xmin + bbox.width, bbox.ymin + bbox.height
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        return image[y1:y2, x1:x2]

def get_ssim(img1, img2):
    return structural_similarity(img1, img2, multichannel=True, channel_axis=2)


def get_psnr(img1, img2):
    return cv2.PSNR(img1, img2)

def resize_images(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return img1, img2


def get_metrics_video(real_vid, fake_vid, detector=None):

    real_cap = cv2.VideoCapture(real_vid)
    fake_cap = cv2.VideoCapture(fake_vid)

    ssim = 0
    psnr = 0
    l1 = 0
    n = 0
    
    n_frames = int(real_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in tqdm(range(n_frames)):
        ret1, real = real_cap.read()
        ret2, fake = fake_cap.read()
        
        if not ret1 or not ret2:
            break
        
        fake = cv2.resize(fake, (real.shape[1], real.shape[0]))
        
        if detector is not None:
            bbox = detector.forward(real)  
            if bbox is None:
                continue
            
            real = detector.crop(real, bbox)
            fake = detector.crop(fake, bbox)
            
            cv2.imwrite('real.jpg', real)
        
        real, fake = resize_images(real, fake)
        
        ssim += get_ssim(real, fake)
        psnr += get_psnr(real, fake)
        l1 += np.mean(np.abs(real - fake)) / 255
        n += 1
    real_cap.release()
    fake_cap.release()
    if n == 0:
        return 0, 0, 0
    
    return ssim / n, psnr / n, l1 / n


def get_metrics_dirs(real, fake, use_detector=True):
    ssim = 0
    psnr = 0
    l1 = 0
    n = 0
    
    videos = os.listdir(real)
    videos = [video for video in videos if video.endswith('.mp4')]
    
    detector = FaceDetector()
    
    for video in tqdm(videos):
        ssim_, psnr_, l1 = get_metrics_video(os.path.join(real, video), os.path.join(fake, video), detector=detector)
        if l1 == 0 and ssim_ == 0 and psnr_ == 0:
            continue
        
        ssim += ssim_
        psnr += psnr_
        l1 += l1
        n += 1
        
    return ssim / n, psnr / n, l1 / n
    

def main(args):
    if args.video_only:
        ssim, psnr, l1 = get_metrics_video(args.real, args.fake)
    else:
        ssim, psnr, l1 = get_metrics_dirs(args.real, args.fake)
    print(f'L1: {l1}, SSIM: {ssim}, PSNR: {psnr}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--real', type=str, default=None)
    parser.add_argument('-f', '--fake', type=str, default=None)
    parser.add_argument('-v', '--video_only', action='store_true')
    args = parser.parse_args()
    main(args)
