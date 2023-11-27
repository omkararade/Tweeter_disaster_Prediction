# Tweeter_Disaster_prediction

 Disaster Tweet Classification Using Bernoulli Naive Bayes
This repository contains the code and data for a machine learning project that classifies tweets as either disaster-related or not using the Bernoulli Naive Bayes algorithm.

Prerequisites
To run the code, you will need to have the following installed:

Python 3.6 or higher
scikit-learn
pandas
nltk
Data
The data for this project was obtained from the Kaggle dataset "Natural Language Processing with Disaster Tweets". The dataset contains over 5,000 tweets, each labeled as either disaster or non-disaster.

Code
The code for this project is organized into the following files:

data_preprocessing.py: Preprocesses the text data and splits it into training and testing sets.
model_training.py: Trains the Bernoulli Naive Bayes model and evaluates its performance.
tweet_classification.py: Classifies new tweets using the trained model.
Usage
To run the code, follow these steps:

Clone the repository to your computer.
Install the required dependencies using pip:
Bash
pip install -r requirements.txt
Use code with caution.
Preprocess the data:
Bash
python data_preprocessing.py
Use code with caution.
Train the model:
Bash
python model_training.py
Use code with caution.
Classify a new tweet:
Bash
python tweet_classification.py <tweet_text>
Use code with caution. Learn more
Results
The Bernoulli Naive Bayes model achieved an accuracy of 82% on the test dataset. This suggests that the model can be used to identify disaster tweets with a high degree of accuracy.

Conclusion
This project demonstrates the effectiveness of machine learning for classifying disaster tweets. The Bernoulli Naive Bayes algorithm is a simple and effective method for this task. The model can be used to identify potential disaster situations and help emergency responders coordinate their efforts.

References
Kaggle dataset "Natural Language Processing with Disaster Tweets": https://www.kaggle.com/competitions/nlp-getting-started
Bernoulli Naive Bayes algorithm: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
