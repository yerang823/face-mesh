# +
# data_dst
data_dst = '/data/dfl/DeepFaceLab_Linux/workspace/images/data_dst/aligned/'
# eyes_img
eyes_img = '/data/dfl/DeepFaceLab_Linux/workspace/images/data_src01_eyes/aligned/'
# nose_img
nose_img = '/data/dfl/DeepFaceLab_Linux/workspace/images/data_src02_nose/aligned/'
# mouth_img
mouth_img = '/data/dfl/DeepFaceLab_Linux/workspace/images/data_src03_mouth/aligned/'
# dst_eyes
dst_eyes = '/data/dfl/DeepFaceLab_Linux/workspace/images/data_dst/mask_XSeg_02eyes/'
# dst_nose
dst_nose = '/data/dfl/DeepFaceLab_Linux/workspace/images/data_dst/mask_XSeg_03nose'
# dst_mouth
dst_mouth = '/data/dfl/DeepFaceLab_Linux/workspace/images/data_dst/mask_XSeg_04mouth'
# src_eyes
src_eyes = '/data/dfl/DeepFaceLab_Linux/workspace/images/data_src01_eyes/mask_XSeg_02eyes'
# src_nose
src_nose = '/data/dfl/DeepFaceLab_Linux/workspace/images/data_src02_nose/mask_XSeg_03nose'
# src_mouth
src_mouth = '/data/dfl/DeepFaceLab_Linux/workspace/images/data_src03_mouth/mask_XSeg_04mouth'
# merge_path
merge_path = '/data/dfl/DeepFaceLab_Linux/workspace/merge/'
# dilation_path
dilation_path = '/data/dfl/DeepFaceLab_Linux/workspace/dilation/'
# blur_path
blur_path = '/data/dfl/DeepFaceLab_Linux/workspace/blur/'
# blend_path
blend_path = '/data/dfl/DeepFaceLab_Linux/workspace/blend/'

import os
import cv2
from tqdm import tqdm
from glob import glob
import numpy as np

os.makedirs(merge_path, exist_ok=True)
os.makedirs(os.path.join(merge_path,'eyes'), exist_ok=True)
os.makedirs(os.path.join(merge_path,'nose'), exist_ok=True)
os.makedirs(os.path.join(merge_path,'mouth'), exist_ok=True)

os.makedirs(dilation_path, exist_ok=True)
os.makedirs(os.path.join(dilation_path,'eyes'), exist_ok=True)
os.makedirs(os.path.join(dilation_path,'nose'), exist_ok=True)
os.makedirs(os.path.join(dilation_path,'mouth'), exist_ok=True)

os.makedirs(blur_path, exist_ok=True)
os.makedirs(os.path.join(blur_path,'eyes'), exist_ok=True)
os.makedirs(os.path.join(blur_path,'nose'), exist_ok=True)
os.makedirs(os.path.join(blur_path,'mouth'), exist_ok=True)

os.makedirs(blend_path, exist_ok=True)

def mask_merge(f):
#     mask1 = cv2.imread(os.path.join(dst_eyes, f))
#     mask2 = cv2.imread(os.path.join(src_eyes, f))
#     mask3 = (mask1.astype(np.int32) + mask2.astype(np.int32)).clip(0,255).astype(np.uint8)
#     cv2.imwrite(os.path.join(merge_path, 'eyes', f), mask3)

#     mask1 = cv2.imread(os.path.join(dst_nose, f))
#     mask2 = cv2.imread(os.path.join(src_nose, f))
#     mask3 = (mask1.astype(np.int32) + mask2.astype(np.int32)).clip(0,255).astype(np.uint8)
#     cv2.imwrite(os.path.join(merge_path, 'nose', f), mask3)

    mask1 = cv2.imread(os.path.join(dst_mouth, f))
    mask2 = cv2.imread(os.path.join(src_mouth, f))
    mask3 = (mask1.astype(np.int32) + mask2.astype(np.int32)).clip(0,255).astype(np.uint8)
    cv2.imwrite(os.path.join(merge_path, 'mouth', f), mask3)

def dilation(f):
    kernel_size = 10
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
#     src_mask = cv2.imread(os.path.join(merge_path, 'eyes', f))
#     dilation_image = cv2.dilate(src_mask, kernel, iterations=1)
#     cv2.imwrite(os.path.join(dilation_path, 'eyes', f),dilation_image)

#     src_mask = cv2.imread(os.path.join(merge_path, 'nose', f))
#     dilation_image = cv2.dilate(src_mask, kernel, iterations=1)
#     cv2.imwrite(os.path.join(dilation_path, 'nose', f),dilation_image)

    src_mask = cv2.imread(os.path.join(merge_path, 'mouth', f))
    dilation_image = cv2.dilate(src_mask, kernel, iterations=1)
    cv2.imwrite(os.path.join(dilation_path, 'mouth', f),dilation_image)


fs = glob(os.path.join(dst_eyes, '*.jpg'))
fs = sorted(fs)
for f in tqdm(fs):
    dilation(os.path.basename(f))


def maskblur(f):
#     src_mask = cv2.imread(os.path.join(dilation_path, 'eyes', f))
#     mask_blur = cv2.GaussianBlur(src_mask, (51,51), 0)
#     cv2.imwrite(os.path.join(blur_path, 'eyes', f), mask_blur)

#     src_mask = cv2.imread(os.path.join(dilation_path, 'nose', f))
#     mask_blur = cv2.GaussianBlur(src_mask, (51,51), 0)
#     cv2.imwrite(os.path.join(blur_path, 'nose', f), mask_blur)

    src_mask = cv2.imread(os.path.join(dilation_path, 'mouth', f))
    mask_blur = cv2.GaussianBlur(src_mask, (51,51), 0)
    cv2.imwrite(os.path.join(blur_path, 'mouth', f), mask_blur)



def alphablend(f):
    background = cv2.imread(os.path.join(data_dst, f)).astype(float)

#     foreground_eyes = cv2.imread(os.path.join(eyes_img, f)).astype(float)
#     foreground_nose = cv2.imread(os.path.join(nose_img, f)).astype(float)
    foreground_mouth = cv2.imread(os.path.join(mouth_img, f)).astype(float)
    
#     alpha_eyes = cv2.imread(os.path.join(blur_path, 'eyes', f)).astype(float)/255
#     alpha_nose = cv2.imread(os.path.join(blur_path, 'nose', f)).astype(float)/255
    alpha_mouth = cv2.imread(os.path.join(blur_path, 'mouth', f)).astype(float)/255

#     foreground_eyes = cv2.multiply(alpha_eyes, foreground_eyes)
#     foreground_nose = cv2.multiply(alpha_nose, foreground_nose)
    foreground_mouth = cv2.multiply(alpha_mouth, foreground_mouth)

#     background = cv2.multiply(1.0 - alpha_eyes, background)

#     outImage1 = cv2.add(foreground_eyes, background)

#     outImage1 = cv2.multiply(1.0 - alpha_nose, outImage1)

#     outImage2 = cv2.add(foreground_nose, outImage1)


    #outImage2 = cv2.multiply(1.0 - alpha_mouth, outImage2)
    outImage2 = cv2.multiply(1.0 - alpha_mouth, background)

    outImage3 = cv2.add(foreground_mouth, outImage2)

    cv2.imwrite(os.path.join(blend_path, f), outImage3)


# +
fs = glob(os.path.join(dst_eyes, '*.jpg'))
fs = sorted(fs)
for f in tqdm(fs):
    mask_merge(os.path.basename(f))
    
fs = glob(os.path.join(dst_eyes, '*.jpg'))
fs = sorted(fs)
for f in tqdm(fs):
    dilation(os.path.basename(f))
    
for f in tqdm(fs):
    maskblur(os.path.basename(f))
for f in tqdm(fs):
    alphablend(os.path.basename(f))
# -
dst_mouth
src_mouth
src_mask
data_dst
mouth_img



    mask1 = cv2.imread(os.path.join(dst_mouth, f))
    mask2 = cv2.imread(os.path.join(src_mouth, f))
    mask3 = (mask1.astype(np.int32) + mask2.astype(np.int32)).clip(0,255).astype(np.uint8)
    
    kernel_size = 10
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
#     dilation_image = cv2.dilate(src_mask, kernel, iterations=1)
    dilation_image = cv2.dilate(mask3, kernel, iterations=1)
    
    #mask_blur = cv2.GaussianBlur(src_mask, (51,51), 0)
    mask_blur = cv2.GaussianBlur(dilation_image, (51,51), 0)
    
    
    background = cv2.imread(os.path.join(data_dst, f)).astype(float) ## simswap image
    foreground_mouth = cv2.imread(os.path.join(mouth_img, f)).astype(float)    ## sberswap image 
    #alpha_mouth = cv2.imread(os.path.join(blur_path, 'mouth', f)).astype(float)/255
    alpha_mouth = mask_blur.astype(float)/255
    foreground_mouth = cv2.multiply(alpha_mouth, foreground_mouth)
    outImage2 = cv2.multiply(1.0 - alpha_mouth, background)
    outImage3 = cv2.add(foreground_mouth, outImage2)
#     cv2.imwrite(os.path.join(blend_path, f), outImage3)

