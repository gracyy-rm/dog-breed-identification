**Dog Breed Identification**

A deep learning project that identifies dog breeds from uploaded images using a trained TensorFlow/Keras model.
The application is built with Streamlit and supports prediction across 120 different dog breeds in real time.
_Live Demo link:_ https://dog-breed-classifier-dw5anjiissumkphdepcffe.streamlit.app/

**Project Overview**

This project includes:

Training a deep learning model on dog breed images
Exporting the trained model as model.h5
Building a Streamlit web application
Allowing users to upload custom dog images
Predicting dog breeds instantly
Model Details
Framework: TensorFlow / Keras
Architecture: CNN / Transfer Learning
Input Image Size: 224 × 224
Number of Classes: 120 Dog Breeds
Loss Function: Categorical Crossentropy
Evaluation Metric: Accuracy
Deployment Model Path: models/model.h5

**Project Structure**
DogVisionProject/

- DogVisionDeploy/
  
  - app/
    - app.py

  - models/
    - model.h5

  - utils/
    - utils.py

  - requirements.txt

  - README.md
**Features**
Upload any dog image
Automatic image preprocessing
Real-time breed prediction
Softmax probability output
Confidence score display
Clean and interactive Streamlit interface

**Installation**
1. Clone the Repository
git clone <your-github-repo-link>
cd DogVisionProject/DogVisionDeploy
2. Create a Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Mac/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt

**Run the Application**

Run the following command inside DogVisionDeploy:

streamlit run app/app.py

The application will automatically open in your browser.

Deployment Options
Streamlit Cloud
Free and beginner-friendly deployment
Direct GitHub integration
Render / HuggingFace Spaces
Good alternatives for hosting ML applications
AWS / GCP / Azure
Suitable for scalable production deployments

**Dataset**

Dataset used for training:

Kaggle Dog Breed Identification Dataset

**Future Improvements**
Integrate Grad-CAM visualizations
Add Docker support
Add GPU inference support
Improve model accuracy
Add prediction history and logging
Suggested Repository Improvements


Author
Gracy Yadav
Deep Learning and Data Science Enthusiast

