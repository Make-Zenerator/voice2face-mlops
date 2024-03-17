from flask import Flask, request
from flask_cors import CORS
from inference import face_synthesis_gif

app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 용량제한
app.config.update(DEBUG=True)

CORS(app, resources={r'*': {'origins': '*'}}, supports_credentials=True)

@app.route('/imagetovideo', methods=['POST'])
def inference():
    make_json = request.get_json()
    request_id = make_json['request_id']
    result_id = make_json['result_id']
    voice_image_url = make_json['voice_image_url']
    gender = make_json['gender']
    age = make_json['age']
    gif_dict = {"woman" : 'hj', "man" : 'tae'}
    
    try: 
        video_url = f"https://storage.makezenerator.com:9000/voice2face-public/site/result/{gif_dict[gender]}_24fps_square.mp4"
        
        result,voice_video_url = face_synthesis_gif(voice_image_url, video_url, request_id, result_id)
        
        if result == 400:
            return {'status_code' : result}

        return {'status_code' : 200,
                'voice_image_url' : voice_image_url,
                'voice_video_url' : voice_video_url}
    except Exception as ex:
        print(ex)
        return {"status_code" : 400, "error": str(ex)} #false->400
    except:
        return {"status_code" : 400} #false->400

if __name__ == "__main__":
    app.run(port=3001, debug=True)
