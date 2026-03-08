🐶 Dog Breed Identification (Deep Learning + Streamlit)

This project is a Dog Breed Identification System that uses a deep learning model to classify images into 120 different dog breeds.
The model is trained using the Kaggle Dog Breed Identification dataset, and the web interface is built using Streamlit, allowing users to upload custom dog images and receive predictions in real time.

🚀 Project Overview

The goal of this project is to build an end-to-end pipeline:

Train a deep learning model on dog breed images

Export the trained model as model.h5

Deploy a user-friendly Streamlit web app

Allow users to upload dog images and get predictions instantly

🧠 Model Details

Input image size: 224 × 224

Architecture: Custom CNN / Transfer Learning (your choice)

Number of dog breeds (classes): 120

Framework: TensorFlow / Keras

Loss function: Categorical Crossentropy

Metrics: Accuracy

Your final exported model for deployment:

models/model.h5

📁 Project Structure
DogVisionProject/
│
└── DogVisionDeploy/
    │── app/
    │   └── app.py
    │── models/
    │   └── model.h5
    │── utils/
    │   └── utils.py
    │── requirements.txt
    │── README.md   ← (THIS FILE)

🎯 Features of the Web App

✔️ Upload any dog image
✔️ Image preprocessing (resize to 224×224, normalization)
✔️ Model prediction using softmax output
✔️ Displays the top predicted dog breed
✔️ Shows confidence score
✔️ Clean UI using Streamlit

🛠️ Installation
Step 1 — Clone the Repository
git clone your-github-repo-link
cd DogVisionProject/DogVisionDeploy

Step 2 — Create Virtual Environment (recommended)
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

Step 3 — Install Dependencies
pip install -r requirements.txt

▶️ Run the Streamlit App

Run this from inside DogVisionDeploy:

streamlit run app/app.py


The app will open in your browser automatically.

📦 Deployment Options

You can deploy this Streamlit app using:

🔹 Streamlit Cloud (Free)

Simplest method — directly push your repo to GitHub and connect Streamlit Cloud.

🔹 Render / HuggingFace Spaces

Good for free GPU/CPU hosting.

🔹 AWS / GCP / Azure

For production-grade apps.

If you want deployment help, I’ll guide you step-by-step.

📚 Dataset

Kaggle Dog Breed Identification Dataset:
https://www.kaggle.com/c/dog-breed-identification/data

✨ Future Improvements

Add top 3 predicted breeds

Add Grad-CAM heatmaps

Add error logging

Deploy using Docker

Add GPU support for faster inference

🙋‍♀️ Author

Gracy Yadav
Deep Learning & Data Science Enthusiast
(Project built for portfolio, learning & deployment practice)

