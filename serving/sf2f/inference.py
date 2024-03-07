import models
import torch
import os, glob
import mlflow
from utils.wav2mel import wav_to_mel
from datasets import imagenet_deprocess_batch, set_mel_transform, \
    deprocess_and_save, window_segment
import mlflow
from PIL import Image

model_url = "runs:/0d43501443014c88b9337427c83a0d8f/sf2f_pytorch"

#docker compose에서 지정해줘야함 Fastapi 
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"

model = mlflow.pytorch.load_model(model_url).cuda().eval()

def load_voice_to_face(model_url):
    global model
    model =  mlflow.pytorch.load_model(model_url).cuda().eval()
    
def generate_voice_to_face(voice_url):

    mel_transform = set_mel_transform("vox_mel")
    image_normalize_method = 'imagenet'
    log_mel = wav_to_mel(voice_url)
    log_mel = mel_transform(log_mel).type(torch.cuda.FloatTensor)

    log_mel_segs = window_segment(log_mel, window_length=125, stride_length=63)
    log_mel = log_mel.unsqueeze(0)

    with torch.no_grad():
        imgs_fused, others = model(log_mel_segs.unsqueeze(0))
    if isinstance(imgs_fused, tuple):
        imgs_fused = imgs_fused[-1]
    imgs_fused = imgs_fused.cpu().detach()
    imgs_fused = imagenet_deprocess_batch(imgs_fused, normalize_method=image_normalize_method)
    for j in range(imgs_fused.shape[0]):
        img_np = imgs_fused[j].numpy().transpose(1, 2, 0) # 64x64x3
    print(type(img_np))
    pil_image = Image.fromarray(img_np)
    pil_image.save("./ss_korean.png")
    return img_np
# generate_voice_to_face("/home/hojun/Documents/project/boostcamp/final_project/mlops/pipeline/serving/sf2f/녹음_남자목소리_여잘노래.wav")
generate_voice_to_face("/home/hojun/Documents/project/boostcamp/final_project/mlops/mlflow/train/mlflow/sf2f/data/녹음.wav")