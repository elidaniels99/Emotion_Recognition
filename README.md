# Image_Sentiment_Classification

Background:
Emotion recognition is a critical component of human-machine interaction in various applications, including virtual assistants, healthcare, entertainment, and customer service. Accurate detection of human emotions from visual data, such as facial expressions, is essential for enhancing the quality and effectiveness of these interactions.

Objective:
The objective of this project is to develop a robust emotion recognition system that can accurately classify human emotions. The system should be capable of detecting a range of emotions, such as happiness, sadness, anger, surprise, fear, disgust, and neutrality.

Tasks:

Data Preprocessing: Load and preprocess the training and testing data. Convert pixel values to a suitable numerical format, normalize them, and reshape them to the original image dimensions.
Label Encoding: Encode the emotion labels as integers or one-hot vectors, depending on the model requirements.
Model Selection: Choose an appropriate deep learning model architecture for emotion recognition. You can start with a simple CNN-based model and explore more advanced architectures like VGG-16 for fine-tuning.
Model Training: Train the selected model on the training data while monitoring key metrics such as loss and accuracy. Utilize techniques like data augmentation and dropout to enhance model performance.
Model Evaluation: Evaluate the trained model's performance on the test dataset. Measure accuracy, precision, recall, and F1-score to assess the model's ability to correctly classify emotions.
Fine-Tuning: If the initial model performance is not satisfactory, fine-tune hyperparameters, model architecture, or use transfer learning techniques to improve results.
Deployment: Once a satisfactory model is achieved, prepare it for deployment in real-world applications, where it can interact with users and recognize their emotions from facial expressions.
Deliverables:

A well-documented Python codebase that includes data preprocessing, model training, evaluation, and fine-tuning.
A trained emotion recognition model that can accurately classify emotions in real-time.
A report summarizing the project, including data analysis, model selection, training process, and evaluation results.
Recommendations for potential applications of the emotion recognition system in human-machine interaction scenarios.
Success Criteria:

The success of this project will be measured by achieving a high level of accuracy in classifying emotions from facial images, making the system suitable for practical use in various human-machine interaction applications.

Streamlit App:
1. Image Emotion Recognition:

Upload an image to detect and display emotions expressed in human faces. The app uses deep learning to recognize emotions like happiness, sadness, suprise, anger, disgust, neutral.

2. Live Webcam Emotion Detection:

Experience real-time emotion detection through your webcam. Toggle the webcam feed, and see your own emotions overlaid on the live video
