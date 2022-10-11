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
import copy

from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from pathlib import Path
from insightface_func.face_detect_crop_multi import Face_detect_crop

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates


# -

def lip_seg(path_dir):
    res = []
    lips_alpha = []
    inter_lips_alpha = []
    #LIPS_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
    LIPS_INDEXES = [164, 393, 391, 322, 410, 287, 273, 335, 406, 313, 18, 83, 182, 106, 43, 57, 186, 92, 165, 167]
    INTER_LIPS_INDEXES = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]
    
    EYEREGION_INDEX = [151,108,69,104,68,70,156,143,111,117,118,101,100,5,  329,330,347,346,340,372,383,301,298,333,299,337]
    
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    #cap = cv2.VideoCapture(vid_path)
    
    img_li = sorted(glob.glob(path_dir+'/*.*g'))
    
    ############################
    
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    align, mat, bbox, kps = app.get(frame,crop_size)
    align_li.append(align)
    mat_li.append(mat)
    bbox_li.append(bbox)
    kps_li.append(kps)
    
    if not np_exist:
        np.save(f'../result/tmp_npy/{vid_name}_align.npy', np.array(align_li))
        np.save(f'../result/tmp_npy/{vid_name}_mat.npy', np.array(mat_li))
        np.save(f'../result/tmp_npy/{vid_name}_bbox.npy', np.array(bbox_li))
        np.save(f'../result/tmp_npy/{vid_name}_kps.npy', np.array(kps_li))
    
    
    #############################
    
    try:
        bbox_li = np.load(f'../result/tmp_npy/{vid_name}_bbox.npy')
        np_exist = True
        print('\nFOUND EXISTING NPY!\n')
        
    except:
        align_li, mat_li, bbox_li, kps_li = [],[],[],[]
        np_exist = False
        
        app = Face_detect_crop(name='antelope', root='./insightface_func/models')
        app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
        print('\nCAN\'T FIND EXISTING NPY, GENERATING...\n')
        
    
    with mp_face_mesh.FaceMesh( max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5) as face_mesh:
        for idx, img_path in enumerate(img_li):
            image = cv2.imread(img_path)
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # ================= FACE DETECTION ===================
            # 얼굴 검출 이후에 다음 프로세스 진행, 한장에 얼굴 하나만 있다는 가정 하에
            if np_exist:
                bbox = bbox_li[idx]
            else:
                align, mat, bbox, kps = app.get(frame,crop_size)
                align_li.append(align)
                mat_li.append(mat)
                bbox_li.append(bbox)
                kps_li.append(kps)
                
            x1 = int(bbox[0][0])
            y1 = int(bbox[0][1])
            x2 = int(bbox[0][2])
            y2 = int(bbox[0][3])
            
            image_ori = copy.deepcopy(image)
            image = image[y1:y2, x1:x2]
            
            # =====================================================
            
            
            
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True

            if results.multi_face_landmarks:
                cords = []
                for face_landmarks in results.multi_face_landmarks:
                    for lip_id in EYEREGION_INDEX:
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
                
                
#                 cords = []
#                 for face_landmarks in results.multi_face_landmarks:
#                     for lip_id in EYEREGION_INDEX:
#                         lid = face_landmarks.landmark[lip_id]
#                         #cord = _normalized_to_pixel_coordinates(lid.x,lid.y,image.shape[1],image.shape[0])
#                         cords.append(_normalized_to_pixel_coordinates(lid.x,lid.y,image.shape[1],image.shape[0]))
#                         Y, X = skimage.draw.polygon(list(zip(*cords))[1], list(zip(*cords))[0])
#                     cropped_img = np.zeros(image.shape, dtype=np.uint8)
#                     cropped_img[Y, X] = 255 #image[Y, X]
#                 lips_alpha.append(cropped_img)
                
            else:
                print("FACE DETECTION FAILED ! ")
                lips_alpha.append(np.zeros(image.shape, dtype=np.uint8))
                inter_lips_alpha.append(np.zeros(image.shape, dtype=np.uint8))
            res.append(image)


    
    return (lips_alpha, res)


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


def postproc(res, res2, lips_alpha, lips_alpha2, out_dir):
    
    #postproc_res = []
    
    for idx in range(len(res2)):
        mask1 = lips_alpha[idx]
        mask2 = lips_alpha2[idx]

        background = res[idx].astype(float)
        #foreground_mouth = res_over[idx].astype(float)
        foreground_mouth = res2[idx].astype(float)

        mask3 = (mask1.astype(np.int32) + mask2.astype(np.int32)).clip(0,255).astype(np.uint8)
        
#         cv2.imwrite(out_dir+'/mask1.jpg', mask1)
#         cv2.imwrite(out_dir+'/mask2.jpg', mask2)
#         cv2.imwrite(out_dir+'/mask3.jpg', mask3)
        #import time
        #time.sleep(100)

        kernel_size = 10
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilation_image = cv2.dilate(mask3, kernel, iterations=1)

        mask_blur = cv2.GaussianBlur(dilation_image, (31,31), 0) # 51,51
        alpha_mouth = mask_blur.astype(float)/255
        
        #cv2.imwrite(out_dir+'/mask_blur.jpg', mask_blur)
        
        
        foreground_mouth = cv2.multiply(alpha_mouth, foreground_mouth)
        outImage2 = cv2.multiply(1.0 - alpha_mouth, background)
        outImage3 = cv2.add(foreground_mouth, outImage2)
        
        
        
        cv2.imwrite(out_dir+'/%07d.jpg'%idx, outImage3)
        
        #postproc_res.append(outImage3)
        #postproc_res.append(foreground_mouth)

    #return postproc_res


# def write_audio(src_path, save_wv_path):
#     vc = VideoFileClip(src_path) 
#     vc.audio.write_audiofile(save_wv_path)


# def write_video(composed, wav_path, fps, output_path, slow_write, verbose=False):
#     duration = len(composed)/fps
    
#     ac = AudioFileClip(wav_path)
#     if verbose:
#         print(ac.duration, duration, abs(ac.duration- duration))
#     #assert(abs(ac.duration - duration) < 0.1)
#     print(ac.duration, duration)
    
#     clip = ImageSequenceClip(composed, fps=fps)
#     #clip = clip.set_audio(ac.subclip(ac.duration-duration, ac.duration))
#     h, w, _ = composed[0].shape
#     if h > 1920:
#         clip = clip.resize((w//2, h//2))

#     ffmpeg_params = None
#     if slow_write:
#         ffmpeg_params=['-acodec', 'aac', '-preset', 'veryslow', '-crf', '17']
        
#     temp_out = output_path
#     Path(temp_out).parent.mkdir(exist_ok=True)
#     if verbose:
#         clip.write_videofile(temp_out, ffmpeg_params=ffmpeg_params)
#     else:
#         clip.write_videofile(temp_out, ffmpeg_params=ffmpeg_params, verbose=verbose, logger=None)
    
#     clip.close()
#     ac.close()
#     del clip
#     del ac


# +
@click.command()
@click.option('--bg_dir', help='existing simswap output dir', required=True, metavar='DIR')
@click.option('--toswap_dir', help='existing sberswap output dir', required=True, metavar='DIR')
#@click.option('--src_wv_dir', help='existing wave source dir', required=True, metavar='DIR')
@click.option('--out_dir', help='Directory to save', metavar='DIR')

def main(bg_dir, toswap_dir, out_dir):
    
    import time, math


    start = time.time()
    math.factorial(100000)
    
    bg_li = glob.glob(f'{bg_dir}/*')
    toswap_li = glob.glob(f'{toswap_dir}/*')
    bg_li.sort()
    toswap_li.sort()
    
    for i in range(len(bg_li)):
        
        path1 = bg_li[i]
        path2 = toswap_li[i]
        #name = sim_li[i].split('/')[-1].split('_')[0] ###
        #vid_path2 = f'{sber_out}/{name}.mp4' ####
        
        print(f'{i+1}/{len(bg_li)} path1={path1}')
        print(f'{i+1}/{len(toswap_li)} path2={path2}')
        
        nameonly = path1.split('/')[-1]
       
        os.makedirs(f'{out_dir}/{nameonly}', exist_ok=True)
 

        lips_alpha, res = lip_seg(path1)
        lips_alpha2, res2 = lip_seg(path2)
        
        
#         blending_res = blend(res, res2, lips_alpha, inter_lips_alpha)
        out_path = f'{out_dir}/{nameonly}'
        postproc(res, res2, lips_alpha, lips_alpha, out_path)

#         write_audio(wv_src_path, save_wv_path)
#         write_video(postproc_res, save_wv_path, 30, save_vid_path, slow_write=False) # true 로 바꿔보기
#         write_video(blending_res, save_wv_path, 30, save_vid_path, slow_write=True) # true 로 바꿔보기
#         os.system(f'ffmpeg -i {save_wv_path} -i {save_vid_path} -map 0:a -map 1:v {save_fin_path}')

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
