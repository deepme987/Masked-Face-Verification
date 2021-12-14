
import io
import os
import cv2
import base64
import numpy as np
import PIL.Image as Image

from deepface import DeepFace

face_dir = "faces/"
# face_dir = "faces_archive/"

CWD = os.path.dirname(__file__)
dir = os.path.join(CWD, face_dir)
IMAGES_PER_CLASS = 2

MODEL_NAME = "VGG-Face"


class FaceDetection:
    verification_threshold = 0.08
    net, model = None, None
    image_size = 160
    embeddings = {}

    def __init__(self):
        FaceDetection.load_models()

    @staticmethod
    def load_models():
        if not FaceDetection.net:
            FaceDetection.net = FaceDetection.load_opencv()

        if not FaceDetection.model:
            FaceDetection.model = FaceDetection.load_dlib()

    # Load DeepFace model - avoids GPU memory leak
    @staticmethod
    def load_dlib():
        model = DeepFace.build_model(MODEL_NAME)

        response = DeepFace.verify(img1_path=os.path.join(CWD, "test.jpg"), img2_path=os.path.join(CWD, "test.jpg"),
                                   model_name=MODEL_NAME, model=model, enforce_detection=False)

        FaceDetection.verification_threshold = response["max_threshold_to_verify"]
        FaceDetection.verification_threshold = 0.25  # Set manual threshold
        print(response)

        return model

    @staticmethod
    def load_opencv():
        model_path = os.path.join(CWD, "./Models/OpenCV/opencv_face_detector_uint8.pb")
        model_pbtxt = os.path.join(CWD, "./Models/OpenCV/opencv_face_detector.pbtxt")
        net = cv2.dnn.readNetFromTensorflow(model_path, model_pbtxt)
        return net

    # Convert base64 to image
    @staticmethod
    def base64_to_numpy(base64_img):
        img_data = bytes(base64_img, encoding='utf-8')
        temp_path = os.path.join(CWD, "temp.png")
        with open(temp_path, "wb") as fh:
            fh.write(base64.decodebytes(img_data))

        img = cv2.imread(temp_path)
        try:
            os.remove(temp_path)
        except PermissionError:
            pass
        return img

    # Convert image to base64
    @staticmethod
    def get_response_image(image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        byte_arr = io.BytesIO()
        pil_img.save(byte_arr, format='PNG')  # convert the PIL image to byte array
        encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii')  # encode as base64
        return encoded_img

    @staticmethod
    def is_same(emb1, emb2):
        diff = np.subtract(emb1, emb2)
        diff = np.sum(np.square(diff))
        return diff < FaceDetection.verification_threshold, diff

    @staticmethod
    def detect_faces(image, detected=None):  # Make display_image to True to manually debug errors
        height, width, channels = image.shape

        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        FaceDetection.net.setInput(blob)
        detections = FaceDetection.net.forward()

        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                faces.append([x1, y1, x2 - x1, y2 - y1])

                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

                if detected:
                    cv2.putText(image, detected, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return image

    @staticmethod
    def get_name(text):
        if "-" in text:
            return text[:text.index('-')]
        else:
            return text[:text.index('.')]

    @staticmethod
    def load_face_embeddings():
        if FaceDetection.embeddings != {}:
            return FaceDetection.embeddings
        print("Loading Face Embeddings")

        embeddings = {}
        for file in os.listdir(dir):
            img_path = dir + file
            try:
                embeds = DeepFace.represent(img_path=img_path, model=FaceDetection.model, enforce_detection=False)
                name = FaceDetection.get_name(file)
                if name in embeddings:
                    embeddings[name].append(embeds)
                else:
                    embeddings[name] = [embeds]
            except Exception as e:
                print(e)
                print(img_path)
                print(f"Unable to get embeddings for file: {file}")

        FaceDetection.embeddings = embeddings
        print("Loaded embeddings successfully")
        return embeddings

    @staticmethod
    def cosine(vector_1, vector_2):
        a = np.matmul(np.transpose(vector_1), vector_2)

        b = np.matmul(np.transpose(vector_1), vector_1)
        c = np.matmul(np.transpose(vector_2), vector_2)

        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    @staticmethod
    def fetch_detections(image, embeddings):
        FaceDetection.load_face_embeddings()

        img = DeepFace.represent(img_path=os.path.join(CWD, "temp.jpg"),
                                 model=FaceDetection.model, enforce_detection=False)
        detections = {}
        for name in embeddings:
            curr_detections = [FaceDetection.cosine(img, embed) for embed in embeddings[name]]
            curr_detections.sort()
            curr_detections = curr_detections[:IMAGES_PER_CLASS // 2]
            val = sum(curr_detections) / min(IMAGES_PER_CLASS // 2, len(curr_detections))
            detections[name] = val

        detections = {k: v for k, v in detections.items() if v <= FaceDetection.verification_threshold}
        if detections:
            detected = {k: v for k, v in sorted(detections.items(), key=lambda item: item[1])}
            detected = list(detected.keys())
            detected = detected[0]
        else:
            detected = None

        image = cv2.resize(image, (520, 400))
        encoded_image = "data:image/png;base64, " + \
                        FaceDetection.get_response_image(FaceDetection.detect_faces(image, detected))
        print(detections)
        return {"data": detections, "image": encoded_image}


def face_recognition_api(image, custom=False):
    if custom:
        npimg = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    else:
        image = FaceDetection.base64_to_numpy(image)

    image = cv2.resize(image, (200, 200))
    cv2.imwrite(os.path.join(CWD, "temp.jpg"), image)

    embeddings = FaceDetection.load_face_embeddings()
    response = FaceDetection.fetch_detections(image, embeddings)
    return response


def initialize_test():
    FaceDetection.load_models()
    embeddings = FaceDetection.load_face_embeddings()
    FaceDetection.fetch_detections(cv2.imread(os.path.join(CWD, "test.jpg")), embeddings)
