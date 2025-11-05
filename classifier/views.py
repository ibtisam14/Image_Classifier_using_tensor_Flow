import tensorflow as tf # import tensorflow
import numpy as np  # import numpy
from django.http import JsonResponse # import JsonResponse
from django.views.decorators.csrf import csrf_exempt # import csrf_exempt
from tensorflow.keras.utils import load_img, img_to_array # import image processing utilities
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # import preprocess_input
from .models import UploadedImage  # import UploadedImage model
from .forms import ImageUploadForm # import ImageUploadForm

# ✅ Load trained model
model = tf.keras.models.load_model("waste_model.h5") # load the trained model

# ✅ Correct 10 classes (alphabetical)
CLASS_NAMES = [   # define class names
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash"
]

# ✅ Preprocessing
def preprocess(image_path):  # define preprocess function
    img = load_img(image_path, target_size=(224, 224)) # load and resize image
    img = img_to_array(img) # convert image to array
    img = preprocess_input(img) # preprocess the image
    return np.expand_dims(img, axis=0)
# ✅ View to handle image classification

@csrf_exempt
def classify_image(request): # define classify_image view 
    if request.method != "POST": # check if request method is POST
        return JsonResponse({"error": "POST only"}, status=405) # return error for non-POST requests

    form = ImageUploadForm(request.POST, request.FILES)  # instantiate form with POST data and files
 
    if not form.is_valid():  # validate form
        return JsonResponse({"error": "Invalid form data"}, status=400) # return error for invalid form

    uploaded = form.save()  # save uploaded image
    image_path = uploaded.image.path # get image path

    # ✅ Preprocess image
    img = preprocess(image_path) # preprocess the image

    # ✅ Predict
    preds = model.predict(img)[0] # make prediction
    idx = np.argmax(preds) # get index of highest prediction

    label = CLASS_NAMES[idx] # get predicted label
    confidence = float(preds[idx]) # get confidence score

    # ✅ Save to database
    uploaded.predicted_label = label # update predicted label
    uploaded.confidence = confidence # update confidence score
    uploaded.save() # save updates to database 

    return JsonResponse({ # return JSON response
        "status": "success", # indicate success
        "label": label,
        "confidence": confidence,
        "image_url": uploaded.image.url
    })
