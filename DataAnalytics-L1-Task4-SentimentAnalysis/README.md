
# Sentiment Analysis – OIBSIP Level 1 Task 4

## Objective
Build a Machine Learning model to classify text data into Positive, Negative, and Neutral sentiments.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- NLTK
- TF-IDF Vectorizer
- Matplotlib
- Seaborn
- WordCloud

## Project Workflow
1. Loaded and inspected the sentiment dataset.
2. Checked the distribution of sentiment classes.
3. Preprocessed text data.
4. Converted text into numerical features using TF-IDF.
5. Split the dataset into 80% training and 20% testing data.
6. Trained two classification models:
   - Naive Bayes
   - Logistic Regression
7. Evaluated both models using accuracy, precision, recall, F1-score, and confusion matrices.
8. Visualized sentiment distribution and WordClouds.
9. Reviewed five misclassified examples.

## Dataset Sentiment Distribution

| Sentiment | Records |
|---|---:|
| Neutral | 27,479 |
| Positive | 21,043 |
| Negative | 11,377 |
| **Total** | **59,899** |

## Model Performance

| Model | Accuracy |
|---|---:|
| Naive Bayes | 63.22% |
| Logistic Regression | 65.93% |

## Conclusion
Logistic Regression performed better than Naive Bayes, achieving approximately 65.93% accuracy.

This project demonstrates how Natural Language Processing and Machine Learning can help analyze customer feedback, product reviews, and public opinion.

## Author
Manik Patil

## Internship
Oasis Infobyte – Data Analytics Internship
