from flask import Flask, request
from flask_cors import CORS
from sf2f import inference as sf2f
from SwimSwap import inference as simswap 

app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 용량제한
app.config.update(DEBUG=True)

CORS(app, resources={r'*': {'origins': '*'}}, supports_credentials=True)

@app.route('/', methods=['GET'])
def inference():
    params = request.args.to_dict()
    request_id = params['request_id']
    result_id = params['result_id']
    age = params['age']
    gender = params['gender']
    voice_url = params['voice_url']

    gif_dict = {"woman" : 'hj', "man" : 'tae'}

    try: 
        result = sf2f.generate_voice_to_face(voice_url, request_id, result_id)
        if result == 400:
            return {'status_code' : result}
        
        voice_image_url = f"https://storage.makezenerator.com:9000/voice2face-public/web_artifact/output/{request_id}_{result_id}_image.png"
        video_url = f"https://storage.makezenerator.com:9000/voice2face-public/site/result/{gif_dict[gender]}_24fps_square.mp4"
        
        result = simswap.face_synthesis_gif(voice_image_url, video_url, request_id, result_id)
        if result == 400:
            return {'status_code' : result}

        voice_gif_url = f"https://storage.makezenerator.com:9000/voice2face-public/web_artifact/output/{request_id}_{result_id}_video.mp4"

        return {'status_code' : 200,
                'voice_image_url' : voice_image_url,
                'voice_gif_url' : voice_gif_url}
    except Exception as ex:
        print(ex)
        return {"status_code" : 400, "error": str(ex)} #false->400

if __name__ == "__main__":
    app.run(port=5050, debug=True)
