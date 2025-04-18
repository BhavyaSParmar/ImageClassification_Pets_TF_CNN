# Image Classification with TensorFlow CNN
This project demonstrates the use of a **Convolutional Neural Network (CNN)** built using **TensorFlow and Keras** to classify images from a dataset into distinct categories. The model is trained, validated, and evaluated for performance using visualization and metrics tracking.

--- 

## 🎯 Objective

To implement and evaluate a **deep learning model** that can:

- Classify images accurately using CNN architecture
- Analyze training and validation performance
- Visualize learning progress and results
- Test model generalization on unseen data

---

## 🖼️ Dataset

The dataset used is an image dataset divided into:

- **Training Set**
- **Validation Set**
- **Test Set**

The structure follows:
- **Training Set (train/)**
-- This subset is used to train the CNN model. It contains images divided into subfolders based on their respective classes. For example:

1. train/class_1/: Contains all training images for Class 1.

2. train/class_2/: Contains all training images for Class 2.

- **Validation Set (val/)**
-- Used during the training process to validate the model's performance and tune hyperparameters. It mirrors the structure of the training set:

1. val/class_1/: Contains validation images for Class 1.

2. val/class_2/: Contains validation images for Class 2.

- **Test Set (test/)**
This set is used after training to evaluate the model's generalization ability on unseen data. It also follows the same class-based subfolder structure:

1. test/class_1/: Contains test images for Class 1.

2. test/class_2/: Contains test images for Class 2.

Each subset ensures that the model is trained, validated, and tested on distinct and properly labeled data, supporting reliable model evaluation.

---

## ⚙️ Tools & Technologies

- **Python** 🐍
- **TensorFlow**, **Keras**
- **Matplotlib**, **NumPy**
- **Jupyter Notebook**

---

## 🏗️ Model Architecture

The model uses a basic CNN with:

- 2 Convolutional Layers with ReLU Activation
- MaxPooling Layers
- Dropout Regularization
- Dense Layers for classification
- Softmax Output Layer

---

## 📈 Training Insights

- Model compiled using **Adam optimizer** and **Sparse Categorical Crossentropy**.
- Plotted **accuracy and loss** graphs for both training and validation sets.
- Model performance evaluated using test data.

---

## 🔍 Key Observations

- 📊 Validation accuracy improved steadily, indicating good generalization.
- 🚫 Overfitting was avoided using **Dropout** layers.
- 🧪 Test accuracy confirmed the model’s ability to classify unseen images.

---

## 📁 Project Structure

The project is structured as follows:

- **img_class_tf_cnn/** 
This is the root directory containing all components of the image classification project.

- **img_class_tf_cnn.ipynb** 
The main Jupyter Notebook that includes the complete workflow—data loading, preprocessing, CNN model building, training, validation, and evaluation.

- **README.md**
The documentation file that outlines the objective, tools, methodology, key insights, and structure of the project. It serves as a quick reference and overview for users.

- **data/**
This folder contains the dataset organized into subdirectories for training, validation, and testing. Each of these is further divided into folders by class labels, enabling easy loading with image generators.

---

## 📚 References

- TensorFlow CNN Documentation
- Deep Learning with Python by François Chollet
- ImageNet and CIFAR-10 Example Datasets

---

## 🌟 Acknowledgments

Special thanks to the open-source community for their contributions and TensorFlow for their extensive tools and documentation. 🙌

---

## 📫 Contact

Created with ❤️ by **Bhavya Parmar**  
📬 Feel free to connect or drop a ⭐ if this project helped you!
