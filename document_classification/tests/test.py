import requests
import json
session = requests.Session()
url = "http://127.0.0.1:8000/score"
data = {
    "title": "download this song",
    "content": "Nigerian super talented artiste Joeboy who has given us many singles and dope body of work, is out with another fresh body of work titled Somewhere Between Beauty & Magic , off the 14 tracked Album is this rack he titled Sugar Mama, listen and download below Download Music: Joeboy – Sugar Mama https://430Box.com/wp-content/uploads/2021/02/Joeboy_-_Sugar_Mama-(430Box.com).mp3 Download Full ALBUM: Joeboy – Somewhere Between Beauty & Magic The post Music: Joeboy – Sugar Mama appeared first on 430box.com ."
}
resp = session.post(url,json=data,verify=False,timeout=10)
print(resp.text)