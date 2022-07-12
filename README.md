# od-endpoint
🍾 Flask server for detecting objects in images, intended to be called by a 🤖 bot

----
## commands

### usaging the API

post an image to the endpoint
```
curl http://url:port/api/detect -X POST -F "image=@testimg.jpg" -F "output=1"
```

post an image URL to the endpoint
```
curl http://url:port/api/detect -X POST -F "input=https://pbs.twimg.com/media/FXeZSe1UIAE44c0?format=jpg" -F "output=1"
```

### deployment

#### package installation

create a new venv  
```
python -m venv venv
```

activate the venv  
```
source venv/bin/activate
```

install packages 
```
pip install -r requirements.txt
```

### generate the css files and create a minimal bundle

this is run automatically before running the flask app
```
tailwindcss -i ./static/src/main.css -o ./static/dist/main.css --minify
```


#### run the server

```
python app.py
```