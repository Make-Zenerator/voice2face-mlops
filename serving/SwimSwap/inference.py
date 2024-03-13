'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-24 19:00:38
Description: 
'''

import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.mp4swap import mp4_swap
from util.upload_minio import upload_object
import os
import mlflow
from minio import Minio

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://223.130.133.236:9000/"
os.environ["MLFLOW_TRACKING_URI"] = "http://223.130.133.236:5001/"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

BUCKET_NAME = "voice2face"
ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
MINIO_API_HOST = "223.130.133.236:9000"
client = Minio(MINIO_API_HOST, ACCESS_KEY, SECRET_KEY, secure=False)

model = mlflow.pytorch.load_model("runs:/31999afcd8784fbfad77aab54f075a84/swimswap_pytorch").eval()

def face_synthesis_gif(face_image_url,base_video_url,request_id,result_id):
    def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0


    transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    crop_size = 224
    
    temp_folder_path = "./temp/"
    os.makedirs(temp_folder_path,exist_ok=True)
    save_path = os.path.join(temp_folder_path,os.path.basename(face_image_url))
    object_path = "/".join(face_image_url.split("/")[4:])
    
    client.fget_object(BUCKET_NAME, object_path, save_path)
    
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode="{}")
    with torch.no_grad():
        pic_a = save_path
        print(pic_a)
        # img_a = Image.open(pic_a).convert('RGB')
        img_a_whole = cv2.imread(pic_a)

        try:
            img_a_align_crop, _ = app.get(img = img_a_whole,crop_size=crop_size)
        except TypeError:
            return 400
            
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0],cv2.COLOR_BGR2RGB)) 
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
        # convert numpy to tensor
        img_id = img_id.cuda()


        #create latent id
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)
        save_url = f"web_artifact/output/{request_id}_{result_id}_video.mp4"
        os.makedirs(os.path.dirname(save_url),exist_ok=True)
        make_flag = mp4_swap(base_video_url, latend_id, model, app, save_url,\
                        no_simswaplogo=True, use_mask=True, crop_size=crop_size)
        # print("dssdlkksld",make_flag)
        if make_flag:
            with open(save_url, 'rb') as file_data:
                file_stat = os.stat(save_url)
                upload_object(client, save_url, file_data,file_stat.st_size,BUCKET_NAME)
                os.remove(save_url)
    return 200


if __name__ == '__main__':
    face_image_url,base_video_url = "http://223.130.133.236:9000/voice2face/web_artifact/output/realface.jpg","http://223.130.133.236:9000/voice2face-public/site/main/hj_24fps_square.mp4"
    face_synthesis_gif(face_image_url,base_video_url,0,0)

