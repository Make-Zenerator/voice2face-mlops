import cv2
import os
import torch
import shutil
import numpy as np
from tqdm import tqdm
from util.reverse2original import reverse2wholeimage
import moviepy.editor as mp
from moviepy.editor import AudioFileClip, VideoFileClip 
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
import imageio.v3 as iio
import tempfile, requests

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

def mp4_swap(video_path, id_vetor, swap_model, detect_model, save_path, temp_results_dir='./temp_results', crop_size=224, no_simswaplogo=False, use_mask=False):

        r = requests.get(video_path)
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(r.content)
            tmp_video_path = tmpfile.name
        video_forcheck = VideoFileClip(tmp_video_path)
        
        if video_forcheck.audio is None:
            no_audio = True
        else:
            no_audio = False

        del video_forcheck

        if not no_audio:
            video_audio_clip = AudioFileClip(video_path)
        
        video = cv2.VideoCapture(video_path)
        logoclass = None
        ret = True
        frame_index = 0

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        video_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        
        fps = video.get(cv2.CAP_PROP_FPS)
        print(fps)
        save_video = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(video_WIDTH,video_HEIGHT))
        if  os.path.exists(temp_results_dir):
                shutil.rmtree(temp_results_dir)

        spNorm =SpecificNorm()
        if use_mask:
            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.cuda()
            save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
            net.load_state_dict(torch.load(save_pth))
            net.eval()
        else:
            net =None

        frames = []
        # while ret:
        for frame_index in tqdm(range(frame_count)): 
            ret, frame = video.read()
            if  ret:
                detect_results = detect_model.get(frame,crop_size)

                if detect_results is not None:
                    # print(frame_index)
                    # if not os.path.exists(temp_results_dir):
                    #         os.mkdir(temp_results_dir)
                    frame_align_crop_list = detect_results[0]
                    frame_mat_list = detect_results[1]
                    swap_result_list = []
                    frame_align_crop_tenor_list = []
                    for frame_align_crop in frame_align_crop_list:

                        # BGR TO RGB
                        # frame_align_crop_RGB = frame_align_crop[...,::-1]

                        frame_align_crop_tenor = _totensor(cv2.cvtColor(frame_align_crop,cv2.COLOR_BGR2RGB))[None,...].cuda()

                        swap_result = swap_model(None, frame_align_crop_tenor, id_vetor, None, True)[0]
                        # cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), frame) # 원본 프레임 img 저장
                        swap_result_list.append(swap_result)
                        frame_align_crop_tenor_list.append(frame_align_crop_tenor)


                    img = reverse2wholeimage(frame_align_crop_tenor_list,swap_result_list, frame_mat_list, crop_size, frame, logoclass,\
                            os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), no_simswaplogo, pasring_model=net, use_mask=use_mask, norm=spNorm)
                    
                    frame = frame.astype(np.uint8)
                    if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                    # video.write(os.path.join(temp_results_dir, 'frame_{:0>7d}.jpg'.format(frame_index)), img)
                    save_video.write(img)
                    # frames.append(img.tobytes())
                    frames.append(img)

                else:
                    if not os.path.exists(temp_results_dir):
                        os.mkdir(temp_results_dir)
                    frame = frame.astype(np.uint8)
                    # if not no_simswaplogo:
                        # frame = logoclass.apply_frames(frame)
                    save_video.write(frame)
            else:
                break
        
        
        # iio.imwrite(save_path,
        #                 frames)
        return True
    # except:
    #     return False

