import models
import torch
import os, glob
import mlflow
from utils.wav2mel import wav_to_mel
from utils.upload_minio import upload_object
from datasets import imagenet_deprocess_batch, set_mel_transform, \
    deprocess_and_save, window_segment
import mlflow
from PIL import Image
import io
from minio import Minio

model_url = "runs:/2413bd3a87c64b139da84f9c6b78813c/sf2f_pytorch"

#docker compose에서 지정해줘야함 Fastapi 
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://223.130.133.236:9000"
os.environ["MLFLOW_TRACKING_URI"] = "http://223.130.133.236:5001"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "miniostorage"
os.environ["MINIO_BUCKET"] = "voice2face"
os.environ["MINIO_ENDPOINT"] = "223.130.133.236:9000"

BUCKET_NAME = "voice2face"
ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
MINIO_API_HOST = "223.130.133.236:9000"
client = Minio(MINIO_API_HOST, ACCESS_KEY, SECRET_KEY, secure=False)

model = mlflow.pytorch.load_model(model_url).cuda().eval()

def load_voice_to_face(model_url):
    global model
    model =  mlflow.pytorch.load_model(model_url).cuda().eval()
    
def generate_voice_to_face(voice_url,request_id,result_id):

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

    pil_image = Image.fromarray(img_np)
    a = Image.open("ss_korean.png")
    print(a.format)
    # Save the image to an in-memory file
    in_mem_file = io.BytesIO()
    pil_image.save(in_mem_file, format="PNG")
    in_mem_file.seek(0)
    img_byte_arr = in_mem_file.getvalue()
    
    
    upload_object(client, f"web_aritifact/output/{request_id}_{result_id}_image.png",in_mem_file,len(img_byte_arr),BUCKET_NAME)
    return img_np
# generate_voice_to_face("/home/hojun/Documents/project/boostcamp/final_project/mlops/pipeline/serving/sf2f/녹음_남자목소리_여잘노래.wav")
generate_voice_to_face("/workspace/people_audio.wav",0,0)