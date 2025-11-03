import tensorflow as tf
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from .models import UploadedImage
from .forms import ImageUploadForm

# Load model
model = tf.keras.models.load_model("waste_model.h5")

CLASS_NAMES = ["plastic", "metal", "glass", "paper", "cardboard", "trash"]

def preprocess(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)


@csrf_exempt
def classify_image(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    form = ImageUploadForm(request.POST, request.FILES)

    if not form.is_valid():
        return JsonResponse({"error": "invalid form"}, status=400)

    uploaded = form.save()
    image_path = uploaded.image.path

    # Preprocess
    img = preprocess(image_path)

    # Predict
    preds = model.predict(img)[0]
    idx = np.argmax(preds)

    label = CLASS_NAMES[idx]
    confidence = float(preds[idx])

    # Save to DB
    uploaded.predicted_label = label
    uploaded.confidence = confidence
    uploaded.save()

    return JsonResponse({
        "status": "success",
        "label": label,
        "confidence": confidence,
        "image_url": uploaded.image.url
    })
