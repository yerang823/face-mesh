# -*- coding: utf-8 -*-
# +
import os
import glob
from PIL import Image
from IPython.display import Image as PIM
import matplotlib.pyplot as plt
import itertools
import skimage
import numpy as np

from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from pathlib import Path

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates


# -

def lip_seg(vid_path):
    res = []
    lips_alpha = []
    inter_lips_alpha = []
    #LIPS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
    LIPS_INDEXES = [164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57, 186, 92, 165, 167]
    INTER_LIPS_INDEXES = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(vid_path)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        idx = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            # if idx > 35:
            #     break
            #image = cv2.resize(image, (3840,2160))

            #image = image[270:2160-270*3, 960:3840-960]
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                cords = []
                for face_landmarks in results.multi_face_landmarks:
                    for lip_id in LIPS_INDEXES:
                        lid = face_landmarks.landmark[lip_id]
                        #cord = _normalized_to_pixel_coordinates(lid.x,lid.y,image.shape[1],image.shape[0])
                        cords.append(_normalized_to_pixel_coordinates(lid.x,lid.y,image.shape[1],image.shape[0]))
                        xs = list(zip(*cords))[0]
                        ys = list(zip(*cords))[1]
                        _len = len(xs)
                        cx = (max(xs) + min(xs)) / 2
                        cy = (max(ys) + min(ys)) / 2
                        lx = (max(xs) - min(xs)) / 2
                        ly = (max(ys) - min(ys)) / 2
                        Y, X = skimage.draw.polygon(ys, xs)
                    cropped_img = np.zeros(image.shape, dtype=np.uint8)
                    for i in range(len(X)):
                        cropped_img[Y[i], X[i]] = min( (max((ly - abs(cy-Y[i])), 0)/ly)*255, (max((lx - abs(cx-X[i])), 0)/lx)*255 )
                        #cropped_img[Y[i], X[i]] = np.sqrt( np.square((max((ly - abs(cy-Y[i])), 0)/ly)*255) + np.square((max((lx - abs(cx-X[i])), 0)/lx)*255) )
                lips_alpha.append(cropped_img)
                cords = []
                for face_landmarks in results.multi_face_landmarks:
                    for lip_id in INTER_LIPS_INDEXES:
                        lid = face_landmarks.landmark[lip_id]
                        #cord = _normalized_to_pixel_coordinates(lid.x,lid.y,image.shape[1],image.shape[0])
                        cords.append(_normalized_to_pixel_coordinates(lid.x,lid.y,image.shape[1],image.shape[0]))
                        Y, X = skimage.draw.polygon(list(zip(*cords))[1], list(zip(*cords))[0])
                    cropped_img = np.zeros(image.shape, dtype=np.uint8)
                    cropped_img[Y, X] = 255 #image[Y, X]
                inter_lips_alpha.append(cropped_img)
            else:
                #print("no lips")
                lips_alpha.append(np.zeros(image.shape, dtype=np.uint8))
                inter_lips_alpha.append(np.zeros(image.shape, dtype=np.uint8))
            res.append(image)
            idx += 1

    cap.release()
    
    return (lips_alpha, inter_lips_alpha, res)


def get_lip_vid(vid_path2):
    vid_path = vid_path2
    res_over = []
    cap = cv2.VideoCapture(vid_path)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        #image = cv2.resize(image, (1920,1080))
        #image = image[270:2160-270*3, 960:3840-960]
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = True

        res_over.append(image)

    cap.release()
    return res_over


def blend(res, res_over, lips_alpha, inter_lips_alpha):
    
    blending_res = []
    for idx in range(len(res_over)): # len(res)
        foreground = res_over[idx].astype(float)
        background = res[idx].astype(float)
        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = (lips_alpha[idx].astype(float)/255)*0.3 #*0.6
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, background)
        temp_outImage = cv2.add(foreground, background)

        background = cv2.add(foreground, background)
        foreground = res_over[idx].astype(float)
        alpha = (inter_lips_alpha[idx].astype(float)/255)*0.7
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, background)
        outImage = cv2.add(foreground, background)
        blending_res.append(outImage)
        
    return blending_res


def write_audio(src_path, save_wv_path):
    vc = VideoFileClip(src_path) #'../simswap/jtbc_template/csy.mp4'
    vc.audio.write_audiofile(save_wv_path)


def write_video(composed, wav_path, fps, output_path, slow_write, verbose=False):
    duration = len(composed)/fps
    
    ac = AudioFileClip(wav_path)
    if verbose:
        print(ac.duration, duration, abs(ac.duration- duration))
    #assert(abs(ac.duration - duration) < 0.1)
    print(ac.duration, duration)
    
    clip = ImageSequenceClip(composed, fps=fps)
    #clip = clip.set_audio(ac.subclip(ac.duration-duration, ac.duration))
    h, w, _ = composed[0].shape
    if h > 1920:
        clip = clip.resize((w//2, h//2))

    ffmpeg_params = None
    if slow_write:
        ffmpeg_params=['-acodec', 'aac', '-preset', 'veryslow', '-crf', '17']
        
    temp_out = output_path
    Path(temp_out).parent.mkdir(exist_ok=True)
    if verbose:
        clip.write_videofile(temp_out, ffmpeg_params=ffmpeg_params)
    else:
        clip.write_videofile(temp_out, ffmpeg_params=ffmpeg_params, verbose=verbose, logger=None)
    
    clip.close()
    ac.close()
    del clip
    del ac


if __name__=='__main__':

    sim_li = glob.glob('../GPEN/jtbc_output_mp4_sim/*.mp4')
    sber_li = glob.glob('../GPEN/jtbc_output_mp4_sber/*.mp4')
    sim_li.sort()
    sber_li.sort()
    
    for i in range(len(sim_li)):
        
        vid_path1 = sim_li[i]
        vid_path2 = sber_li[i]
        print(f'{i}/{len(sim_li)} vid_path1={vid_path1}')
        print(f'{i}/{len(sim_li)} vid_path2={vid_path2}')
        
        vid_name = vid_path1.split('/')[-1].split('.')[0]
        src_name = vid_path1.split('/')[-1][:3]
        wv_src_path = f'../simswap/jtbc_template/{src_name}.mp4'
        save_wv_path = f'./src_wave/{src_name}.wav'
        save_vid_path = f'./output_nowav/{vid_name}.mp4'
        save_fin_path = f'./output/{vid_name}.mp4'
        
        if os.path.isfile(save_fin_path):
            continue
        if vid_name =='hnh_lsj': ############ 파일오류
            continue

        lips_alpha, inter_lips_alpha, res = lip_seg(vid_path1)
        res_over = get_lip_vid(vid_path2)
        blending_res = blend(res, res_over, lips_alpha, inter_lips_alpha)

        write_audio(wv_src_path, save_wv_path)
        write_video(blending_res, save_wv_path, 30, save_vid_path, slow_write=False)
        os.system(f'ffmpeg -i {save_wv_path} -i {save_vid_path} -map 0:a -map 1:v {save_fin_path}')
