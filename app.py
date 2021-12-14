import json
import os
import sys

from flask import Flask, render_template, request

from face_recognition import face_recognition_api, initialize_test

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('capture_image.html')


@app.route('/detect-face', methods=["POST"])
def detect_face():
    try:
        data = json.loads(request.get_data())
        if "image" in data:
            image = data["image"].split(",")[1]
        else:
            return {"response": "Invalid Parameters", "data": {}}

    except Exception as e:
        print(e)
        return {"response": "Unexpected error occurred", "data": {}}

    response = face_recognition_api(image)
    return response


if __name__ == '__main__':
    print("Initializing backend server")
    initialize_test()
    print("Backend started successfully, initializing server")
    app.run(debug=False)
