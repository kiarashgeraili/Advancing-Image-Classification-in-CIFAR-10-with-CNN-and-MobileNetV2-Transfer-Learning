# Deep-Learning-Image-Classification-with-CIFAR-10-Dataset

Deep Learning Image Classification with CIFAR-10 Dataset Using CNN (with Transfer learning using MobileNetV2 architecture, pre-trained on the ImageNet dataset, for object detection on CIFAR-10 )

Introduction:
This project aims to implement an image classification model using deep learning frameworks for the CIFAR-10 dataset. The dataset consists of 60,000 color images with dimensions of 32x32, divided into 50,000 training images and 10,000 test images. The images are categorized into ten classes. The project follows a convolutional neural network (CNN) architecture with specific layers and activation functions.

Dataset:
The CIFAR-10 dataset can be obtained from its official webpage and used for training and testing the models. Optionally, only the first category (1_data_batch) can be used as training data, if resource constraints apply. Please mention this choice in the report.

Model Architecture:
The CNN model comprises the following layers:

1- Convolutional layer: Kernel size 3x3, Output channels 7, Activation function: ReLU.
2- Convolutional layer: Kernel size 3x3, Output channels 9, Activation function: ReLU.
3- Max Pooling layer: Kernel size 2x2.
4- Dropout layer with 30% probability.
5- Fully connected layer with the number of neurons equal to the number of classes.

Training:
The model is trained using the Adam optimizer, categorical cross-entropy loss, and a learning rate of 0.01 (batch size 32). The training process includes validation data to monitor convergence. The model's accuracy is evaluated using the test dataset.

Experimentation:
Several modifications are applied to the base architecture and parameters to analyze their impact on performance:

1- Varying the number of convolutional layers from two to four.
2- Changing the kernel size of the first convolutional layer to 5x5 and 7x7.
3- Replacing the first and second convolutional layers with a single 5x5 kernel layer.
4- Modifying the learning rate of the optimizer to 0.0004, 0.0003, and 0.0001.
5- Trying alternative activation functions like leaky ReLU and tanh.
6- Swapping the optimizer to SGD and Adam.
7- Adding Batch Normalization as the first layer of the network.
8- Changing the batch size to 4 and 128.

Visualizing Convolutional Layers:
After training, visualize the output of the first convolutional layer for a sample test image. Display each channel as a grayscale image. Additionally, visualize the kernel weights of one of the channels.

Transfer Learning with MobileNetV2:
Implement transfer learning with MobileNetV2 architecture, pre-trained on the ImageNet dataset, for object detection on CIFAR-10. Add linear layers with 10 output neurons for classifying the CIFAR-10 categories after the feature extraction layers. Compare the convergence speed and accuracy against the base model, discussing the benefits and drawbacks of using pre-trained models.
