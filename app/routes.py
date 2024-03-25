from flask import render_template
from app import app
from app.forms import ImageForm

import torch
from torchvision import transforms

model = torch.jit.load('models/mbnet.pth')
classes = ['White Wagtail',
 'Cattle Egret',
 'House Crow',
 'Rufous Treepie',
 'Hoopoe',
 'Indian Grey Hornbill',
 'Indian Roller',
 'Red-Wattled Lapwing',
 'Asian Green Bee-Eater',
 'Brown-Headed Barbet',
 'Ruddy Shelduck',
 'Indian Peacock',
 'Northern Lapwing',
 'Forest Wagtail',
 'Common Tailorbird',
 'Coppersmith Barbet',
 'Common Myna',
 'Common Rosefinch',
 'Common Kingfisher',
 'Indian Pitta',
 'White-Breasted Waterhen',
 'Sarus Crane',
 'Jungle Babbler',
 'White-Breasted Kingfisher',
 'Gray Wagtail']

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
