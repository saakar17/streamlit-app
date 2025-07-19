
Chest X-ray Classification Web App using Streamlit

<img width="1440" height="781" alt="Screenshot 2025-07-19 at 10 23 21 AM" src="https://github.com/user-attachments/assets/4e356186-c2ff-42d8-9317-bb2ade894834" />


Project Title:

Streamlit Chest X-ray Classifier App

Project Level:
Beginner-Level Deep Learning & Web App Deployment

1.	Overview
This project is a web application developed using Streamlit.
It allows users to upload chest X-ray images and receive classification predictions using a pre-trained EfficientNetB0 model.
This tool can assist in automatic detection of chest diseases from X-ray images for educational and experimental purposes.

2.	Features
-	User-friendly web interface for image uploads
-	Real-time disease prediction from chest X-rays
-	Supports multi-class classification (e.g., Pneumonia, Cardiomegaly, Effusion, Normal, etc.)
-	Streamlit-based deployment, easily extendable to cloud platforms
  
3.	Dataset Used
The model was trained using the NIH Chest X-ray Dataset.
Dataset Link:
https://www.kaggle.com/datasets/saakaragrawal/preprocessed-and-splitted-nih-chest-xray-dataest

4.	Project Structure
-	app.py: Main Streamlit application
-	model/chest_xray_model.pth: Pre-trained model weights
-	requirements.txt: List of required Python libraries
-	README.md: Project documentation

5. Model Information
-	Architecture: EfficientNetB0
-	Input Image Size: 224x224 pixels
-	Type: Multi-class classification
-	Trained On: Balanced subset of NIH chest X-ray dataset
  
6. How It Works
1.	User uploads a chest X-ray image (format: JPG or PNG).
2.	The app preprocesses the image (resize, normalization).
3.	The EfficientNetB0 model predicts the disease class.
4.	The prediction is displayed on the screen.
   
7. Installation Guide
Prerequisites:
-	Python 3.8+
-	Pip (Python package manager)
Install Dependencies:
pip install -r requirements.txt
Run the Application: streamlit run app.py

 Contributor
- Saakar Agrawal
  

This project is licensed under the MIT License.
