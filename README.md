Brain Tumor Detection Using CNN

This project implements a Convolutional Neural Network (CNN) to detect brain tumors from MRI images. It is designed to classify images into four categories: No Tumor, Glioma Tumor, Meningioma Tumor, and Pituitary Tumor.


Table of Contents

•	Introduction

•	Dataset

•	Installation

•	Usage

•	Model Architecture

•	Results

•	License

	Introduction 
 
Brain tumor detection and classification is crucial for early diagnosis and treatment planning. This project leverages deep learning and CNNs to automatically classify brain MRI images into different tumor types, helping in the initial assessment and diagnosis. The model is trained on a labeled dataset of MRI images and aims to provide high accuracy.
 
 Dataset
 
The dataset consists of MRI images categorized into four classes:

•	No Tumor

•	Glioma Tumor

•	Meningioma Tumor

•	Pituitary Tumor

The data is organized into Training and Testing folders, each containing subfolders for the four classes. Each image is labeled according to the type of brain tumor or as "No Tumor."

Installation

1.	Clone the Repository

git clone https://github.com/your-username/Brain_Tumor_Detection.git

cd Brain_Tumor_Detection

2.	Install Dependencies Use the following command to install required Python packages.

pip install -r requirements.txt

Your requirements.txt should include:

numpy

pandas

matplotlib

seaborn

scikit-learn

xgboost

opencv-python-headless

tensorflow

keras

pillow

3.	Prepare the Dataset

Download the dataset and organize it in the following structure:

Brain_Tumor_Dataset/

├── Training/

│   ├── no_tumor/

│   ├── glioma_tumor/

│   ├── meningioma_tumor/

│   └── pituitary_tumor/

└── Testing/

    ├── no_tumor/
		
	  ├── glioma_tumor/
	 
		├── meningioma_tumor/
	
		└── pituitary_tumor/

Usage

1. Prepare the Data

python Copy code python prepare_data.py

2. Train the Model

python Copy code python train_model.py 

3. Evaluate the Model After training, you can run the evaluation script:

python Copy code python evaluate_model.py 

 Model Architecture

The CNN model consists of:

•	Convolutional layers with ReLU activations

•	MaxPooling layers

•	Batch Normalization for stability

•	Dropout layers to prevent overfitting

•	Fully connected layers leading to the output layer with softmax activation

License

This project is licensed under the MIT License. See the LICENSE file for details.

