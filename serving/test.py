import requests
import json
data = {
    "request_id" : 3,
    "result_id" : 19,
    "age": 25,
    "gender": "woman",
    "voice_url":"https://storage.makezenerator.com:9000/voice2face-public/example/sy.wav" 
}
headers = {"user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36","Content-Type":"application/json"}
response = requests.post("http://1.227.152.189:3002/makevideo",data=json.dumps(data),headers=headers)
response_data = response.json()
print(response_data)
print(response_data.get('status_code'))
print(response_data.get('error'))