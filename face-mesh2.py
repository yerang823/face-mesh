# -*- coding: utf-8 -*-
# +
import os
import glob
import click
from PIL import Image
from IPython.display import Image as PIM
import matplotlib.pyplot as plt
import itertools
import skimage
import numpy as np

from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from pathlib import Path

import cv2
import copy
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

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
    with mp_face_mesh.FaceMesh( max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5) as face_mesh:
 
        idx=0
        while cap.isOpened():
            print('idx=',idx)
            success, image = cap.read()
            if not success:
                break
            h = image.shape[0]
            w = image.shape[1]
            
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = copy.deepcopy(image_ori)
            image2 = copy.deepcopy(image)
            image3 = copy.deepcopy(image)
            
            # <<<< 이미지에서 얼굴들 detection >>>>
            with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
                det_res = face_detection.process(image)#cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                #print('len(det_res.detections) ===',len(det_res.detections))
                for i in range(len(det_res.detections)):
                    bbox = det_res.detections[i].location_data.relative_bounding_box
                    x1 = round(bbox.xmin * w)
                    y1 = round(bbox.ymin * h)
                    x2 = round(x1 + bbox.width*w)
                    y2 = round(y1 + bbox.height*h)


                    #print(x1,y1,x2,y2)
                    im = image[y1:y2, x1:x2]       

#                     im.flags.writeable = False
#                     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    mesh_res = face_mesh.process(im)

                    # Draw the face mesh annotations on the image.
                    im.flags.writeable = True

                    if mesh_res.multi_face_landmarks:

                        cords = []
                        for face_landmarks in mesh_res.multi_face_landmarks:
                            for lip_id in LIPS_INDEXES:
                                lid = face_landmarks.landmark[lip_id]
                                #cord = _normalized_to_pixel_coordinates(lid.x,lid.y,image.shape[1],image.shape[0])
                                cords.append(_normalized_to_pixel_coordinates(lid.x,lid.y,im.shape[1],im.shape[0]))
                                xs = list(zip(*cords))[0]
                                ys = list(zip(*cords))[1]
                                _len = len(xs)
                                cx = (max(xs) + min(xs)) / 2
                                cy = (max(ys) + min(ys)) / 2
                                lx = (max(xs) - min(xs)) / 2
                                ly = (max(ys) - min(ys)) / 2
                                Y, X = skimage.draw.polygon(ys, xs)
                            cropped_img = np.zeros(im.shape, dtype=np.uint8)
                            for i in range(len(X)):
                                cropped_img[Y[i], X[i]] = min( (max((ly - abs(cy-Y[i])), 0)/ly)*255, (max((lx - abs(cx-X[i])), 0)/lx)*255 )
                                #cropped_img[Y[i], X[i]] = np.sqrt( np.square((max((ly - abs(cy-Y[i])), 0)/ly)*255) + np.square((max((lx - abs(cx-X[i])), 0)/lx)*255) )
                        #lips_alpha.append(cropped_img)
                        image2[y1:y2, x1:x2] = cropped_img 

                        cords = []
                        for face_landmarks in mesh_res.multi_face_landmarks:
                            for lip_id in INTER_LIPS_INDEXES:
                                lid = face_landmarks.landmark[lip_id]
                                #cord = _normalized_to_pixel_coordinates(lid.x,lid.y,image.shape[1],image.shape[0])
                                cords.append(_normalized_to_pixel_coordinates(lid.x,lid.y,im.shape[1],im.shape[0]))
                                Y, X = skimage.draw.polygon(list(zip(*cords))[1], list(zip(*cords))[0])
                            cropped_img = np.zeros(im.shape, dtype=np.uint8)
                            cropped_img[Y, X] = 255 #image[Y, X]
                        #inter_lips_alpha.append(cropped_img)
                        image3[y1:y2, x1:x2] = cropped_img


                    else:
                        print("no lips")
                        #lips_alpha.append(np.zeros(image.shape, dtype=np.uint8))
                        #inter_lips_alpha.append(np.zeros(image.shape, dtype=np.uint8))
                        image2 = np.zeros(image.shape, dtype=np.uint8)
                        image3 = np.zeros(image.shape, dtype=np.uint8)
            
            # <<< 얼굴들 원본에 붙여서 전체이미지 한장으로 만들기>>>
                        
            res.append(image)
            lips_alpha.append(image2)
            inter_lips_alpha.append(image3)
            idx+=1


    cap.release()
    
    return (lips_alpha, inter_lips_alpha, res)


def blend(res, res_over, lips_alpha, inter_lips_alpha):
    
    blending_res = []
    for idx in range(len(res_over)): # len(res)
        foreground = res_over[idx].astype(float)
        background = res[idx].astype(float)
        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = (lips_alpha[idx].astype(float)/255)*0 #*0.3 #*0.6
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, background)
        temp_outImage = cv2.add(foreground, background)

        background = cv2.add(foreground, background)
        foreground = res_over[idx].astype(float)
        
        # ===========================
        mask1 = lips_alpha[idx]
        mask2 = inter_lips_alpha[idx]
        mask3 = (mask1.astype(np.int32) + mask2.astype(np.int32)).clip(0,255).astype(np.uint8)
        kernel_size = 10
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilation_image = cv2.dilate(mask3, kernel, iterations=1)
        mask_blur = cv2.GaussianBlur(dilation_image, (51,51), 0)
        alpha_mouth = mask_blur.astype(float)/255
        
        # ===========================
        alpha = (inter_lips_alpha[idx].astype(float)/255)*1.0 #*0.7        
        #alpha = (inter_lips_alpha[idx].astype(float)/255)*1.0 #*0.7        
        
        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1.0 - alpha, background)
        outImage = cv2.add(foreground, background)
        blending_res.append(outImage)
        
    return blending_res


def postproc(res, res2, inter_lips_alpha, inter_lips_alpha2):
    
    postproc_res = []
    
    for idx in range(len(res2)):
        mask1 = inter_lips_alpha[idx]
        mask2 = inter_lips_alpha2[idx]

        background = res[idx].astype(float)
        #foreground_mouth = res_over[idx].astype(float)
        foreground_mouth = res2[idx].astype(float)

        mask3 = (mask1.astype(np.int32) + mask2.astype(np.int32)).clip(0,255).astype(np.uint8)

        kernel_size = 10
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilation_image = cv2.dilate(mask3, kernel, iterations=1)

        mask_blur = cv2.GaussianBlur(dilation_image, (51,51), 0)
        alpha_mouth = mask_blur.astype(float)/255
        
        foreground_mouth = cv2.multiply(alpha_mouth, foreground_mouth)
        outImage2 = cv2.multiply(1.0 - alpha_mouth, background)
        outImage3 = cv2.add(foreground_mouth, outImage2)
        
        postproc_res.append(outImage3)
        #postproc_res.append(foreground_mouth)

    return postproc_res


def write_audio(src_path, save_wv_path):
    vc = VideoFileClip(src_path) 
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


# +
@click.command()
@click.option('--sim_vid_out', help='existing simswap output dir', required=True, metavar='DIR')
@click.option('--sber_vid_out', help='existing sberswap output dir', required=True, metavar='DIR')
@click.option('--src_wv_dir', help='existing wave source dir', required=True, metavar='DIR')
@click.option('--out_dir', help='Directory to save', metavar='DIR')

def main(sim_vid_out, sber_vid_out, src_wv_dir, out_dir):
    
    import time, math


    start = time.time()
    math.factorial(100000)
    
    sim_li = glob.glob(f'{sim_vid_out}/*.mp4')
    sber_li = glob.glob(f'{sber_vid_out}/*.mp4')
    sim_li.sort()
    sber_li.sort()
    
    for i in range(len(sim_li)):
        
        vid_path1 = sim_li[i]
        vid_path2 = sber_li[i]
        #name = sim_li[i].split('/')[-1].split('_')[0] ###
        #vid_path2 = f'{sber_out}/{name}.mp4' ####
        
        print(f'{i}/{len(sim_li)} vid_path1={vid_path1}')
        print(f'{i}/{len(sim_li)} vid_path2={vid_path2}')
        
        vid_name = vid_path1.split('/')[-1].split('.')[0]
        src_name = vid_path1.split('/')[-1].split('.')[0].split('_')[0]
        #wv_src_path = f'../simswap/jtbc_template/{src_name}.mp4'
        wv_src_path = f'{src_wv_dir}/{src_name}.mp4'
        save_wv_path = f'./src_wave/{src_name}.wav'
        save_vid_path = f'./output_nowav/{vid_name}.mp4'
        save_fin_path = f'{out_dir}/{vid_name}.mp4'
        
        os.makedirs(f'{out_dir}', exist_ok=True)
        
        if os.path.isfile(save_fin_path):
            continue
        if vid_name =='hnh_lsj': ############ 파일오류
            continue

        print('lib seg2')
        lips_alpha2, inter_lips_alpha2, res2 = lip_seg(vid_path2)
        print('lib seg1')
        lips_alpha, inter_lips_alpha, res = lip_seg(vid_path1)
        
        
        
#         blending_res = blend(res, res2, lips_alpha, inter_lips_alpha)
        postproc_res = postproc(res, res2, inter_lips_alpha, inter_lips_alpha2)

        write_audio(wv_src_path, save_wv_path)
        write_video(postproc_res, save_wv_path, 30, save_vid_path, slow_write=True) # true 로 바꿔보기
#         write_video(blending_res, save_wv_path, 30, save_vid_path, slow_write=True) # true 로 바꿔보기
        os.system(f'ffmpeg -i {save_wv_path} -i {save_vid_path} -map 0:a -map 1:v {save_fin_path}')

        end = time.time()
        print('===========================================')
        print(f"{end - start:.5f} sec")
        print('===========================================')
    
    #return (lips_alpha,lips_alpha2,inter_lips_alpha,inter_lips_alpha2,res,res2,postproc_res)
# -

if __name__=='__main__':
    main()



# +
#l1,l2, i1,i2, r1,r2, p_res = main('../GPEN/tmp_out_mp4_sim/','../GPEN/tmp_out_mp4_sber/', '../simswap/jtbc_template/','./khe_out_mp4/')
# main('../GPEN/tmp_out_mp4_sim/','../GPEN/tmp_out_mp4_sber/', '../simswap/jtbc_template/','./khe_out_mp4/')
