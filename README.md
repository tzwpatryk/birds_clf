# Bird images classification

My goal is to perform multiclass image classification on bird images that belong to 25 classes. I used dataset available on Kaggle [25-indian-bird-species-with-226k-images](https://www.kaggle.com/datasets/arjunbasandrai/25-indian-bird-species-with-226k-images).

For now I finetuned two CNN architectures: Resnet18 and Mobilenetv2. The first one achieved over 92% accuracy and the second one 95%.

I deployed the best model using Flask. It is a simple application that allows you to upload your image, click predict and check the results.

It is possible to run the application using docker:
```
 docker container run -d -p 3000:3000 --platform linux/amd64 [image name]
 ```