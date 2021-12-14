# Masked Face Verification

Using Aggregated Verification, we attempt to use the existing Face Verification models to adapt with the masked images.

This repository was used in the blog: [A Dive into Analysis of Masked Face Verification
](https://byteiota.com/masked-face-verification/)

If you're interested in reproducing the results in the repo, you can refer to the Google Colaboratory.

Link to Colab: [MaskedFaceVerification - Analysis](https://colab.research.google.com/drive/1XE14PZ6vd6Z6Rjh0Ck19GuO1e7B9NJxJ?usp=sharing)

We look at a new approach to tackle Masked Face Verification.
Instead of re-training a new model or fine-tuning existing models, we leverage the similarity between two images that exists in a verification task.

### Demo

![alt text](http://url/to/img.png)

### To run the demo

```
git clone https://github.com/deepme987/Masked-Face-Verification.git
cd Masked-Face-Verification
pip install -r requirements.txt
python app.py
```

This will run a Flask server at: http://127.0.0.1:5000/

### Add New Train data:

To add new training data, you can do either of the following:

- Manually add 1 masked and 1 unmasked image of your face 
OR
- Add 1 unmasked image and follow the repository: MaskTheFace to automatically generate masks on current images.

Link to MaskTheFace: https://github.com/aqeelanwar/MaskTheFace

Make sure that your images are not large. To be sure, you can run `image_resizer.py` to convert all images to 200*200px

# Performance Evaluation

![alt text](http://url/to/img.png)

# References:

- [DeepFace](https://github.com/serengil/deepface/)
- [MaskTheFace](https://github.com/aqeelanwar/MaskTheFace)
- [LFW Face Database](http://vis-www.cs.umass.edu/lfw/)
- [MLFW A Database for Face Recognition on Masked Faces](https://arxiv.org/abs/2109.05804)