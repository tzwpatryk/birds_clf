from flask import render_template, request
from app import app
from app.forms import ImageForm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = models.mobilenet_v2()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 25)
model.load_state_dict(torch.load('models/mbnet.pth', map_location=device))
model.eval()

classes = {0: 'Asian Green Bee-Eater',
            1: 'Brown-Headed Barbet',
            2: 'Cattle Egret',
            3: 'Common Kingfisher',
            4: 'Common Myna',
            5: 'Common Rosefinch',
            6: 'Common Tailorbird',
            7: 'Coppersmith Barbet',
            8: 'Forest Wagtail',
            9: 'Gray Wagtail',
            10: 'Hoopoe',
            11: 'House Crow',
            12: 'Indian Grey Hornbill',
            13: 'Indian Peacock',
            14: 'Indian Pitta',
            15: 'Indian Roller',
            16: 'Jungle Babbler',
            17: 'Northern Lapwing',
            18: 'Red-Wattled Lapwing',
            19: 'Ruddy Shelduck',
            20: 'Rufous Treepie',
            21: 'Sarus Crane',
            22: 'White Wagtail',
            23: 'White-Breasted Kingfisher',
            24: 'White-Breasted Waterhen'}

def process_image(image):
    transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transformation(image).unsqueeze(0)
    return image_tensor


@app.route('/')
@app.route('/home')
def home():
    form = ImageForm()
    return render_template('home.html', form=form)

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    image_tensor = process_image(Image.open(image))
    output = model(image_tensor)
    probabilities = F.softmax(output, dim=1)
    class_idx = probabilities.argmax().item()
    class_name = classes[class_idx]
    probability = probabilities.squeeze()[class_idx].item()
    return render_template('predict.html', class_name=class_name, probability=probability)
    