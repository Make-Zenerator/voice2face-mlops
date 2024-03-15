from flask import Flask, request
from flask_cors import CORS
from inference import generate_voice_to_face
import json
import requests
app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 용량제한
app.config.update(DEBUG=True)

CORS(app, resources={r'*': {'origins': '*'}}, supports_credentials=True)

@app.route('/makevideo', methods=['POST'])
def inference():
    make_json = request.get_json()
    print("2k1l2kl1")
    request_id = make_json['request_id']
    result_id = make_json['result_id']
    age = make_json['age']
    gender = make_json['gender']
    voice_url = make_json['voice_url']

    gif_dict = {"woman" : 'hj', "man" : 'tae'}

    try:
        result, voice_image_url = generate_voice_to_face(voice_url, request_id, result_id)
        if result == 400:
            return {'status_code' : result}

        
        video_make_json = json.dumps({
            "request_id" : request_id,
            "result_id" : result_id,
            "voice_image_url" : voice_image_url,
            "age":age,
            "gender":gender
        })
        headers = {"user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36","Content-Type":"application/json"}
        imagetovideo_response = requests.post("http://127.0.0.1:3001/imagetovideo",data=video_make_json,headers=headers)
        print("asasasasaaaaaaa",imagetovideo_response)
        # if imagetovideo_result == 400:
        #     return {'status_code' : imagetovideo_result}

        # voice_gif_url = 
        print(voice_image_url)
        return {'status_code' : 200,
                'voice_image_url' : voice_image_url,
                'voice_gif_url' : ""}
    except Exception as ex:
        print(ex)
        return {"status_code" : 400, "error": str(ex)} #false->400

if __name__ == "__main__":
    app.run(port=5050, debug=True)
