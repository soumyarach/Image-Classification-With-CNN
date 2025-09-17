# Image-Classification-With-CNN
Installation
To run this code, you'll need to install the following libraries:

   pip install tensorflow numpy matplotlib scikit-learn
What i built ?
This project is a Convolutional Neural Network (CNN) that classifies images from the CIFAR-10 dataset. The model uses TensorFlow and Keras to learn features from images and predict their corresponding labels.

Why i built it ?
I built this project to demonstrate my understanding of deep learning concepts, specifically CNNs, and to practice working with image classification tasks. This project showcases how to:

Load and preprocess image data
Build and train a CNN model
Evaluate the model's performance using accuracy and loss metrics
Visualize the model's performance using plots
How i built it ?
Data Loading: I loaded the CIFAR-10 dataset using

 tf.keras.datasets.cifar10.load_data().
Data Preprocessing: I normalized the pixel values of the images to be between 0 and 1.

Model Architecture: I defined a CNN model using

tf.keras.Sequential
with Conv2D, MaxPooling2D, Flatten, and Dense layers.

Model Compilation: I compiled the model with the Adam optimizer and sparse categorical cross-entropy loss function.

Data Augmentation: I applied data augmentation using

tf.keras.preprocessing.image.ImageDataGenerator
to improve the model's performance.

Model Training: I trained the model using

model.fit()
with data augmentation and validation.

Model Evaluation: I evaluated the model's performance using accuracy and loss metrics and plotted the results.

