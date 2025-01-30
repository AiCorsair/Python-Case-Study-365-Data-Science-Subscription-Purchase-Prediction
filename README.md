# Python Case Study: 365 Data Science Subscription Purchase Prediction


## Table of Contents

- [I. Introduction & Business Value](#I.-Introduction-&-Business-Value)
- [II. Exploring & Preprocessing Data](#II.-Exploring-&-Preprocessing-Data)
- [III. Comparing & Evaluating the Models](#III.-Comparing-&-Evaluating-the-Models)
- [IV. Model Improvement Steps](#IV.-Model-Improvement-Steps)


## I. Introduction & Business Value

In this project, we examined 365 Data Science student engagement metrics, such as the number of days spent on the platform, the total minutes of watched content, and the number of courses started. Using this data, we trained machine learning models like logistic regression and k-nearest neighbors to predict whether students would upgrade their free plan to a paid one.

This analysis is valuable not only for 365 Data Science but for any subscription-based company. Predicting potential customers helps with targeted advertising and exclusive offers, allowing companies to allocate budgets efficiently and increase revenue.

It is important to note that our predictions were based only on platform activity and country. A student’s decision to purchase also depends on factors like financial status, and workload such as whether they have time to start a new career path.

Our classification problem involved a heavily imbalanced dataset, as most students kept their free plan than upgraded. Fortunately, addressing this imbalance was not necessary to complete the project successfully.

This was expected since many users sign up just to explore the platform. Some engage with the free content in ways similar to those who eventually subscribe, making it difficult to distinguish them.


## II. Exploring & Preprocessing Data

- We imported the necessary libraries and loaded the customer data from a CSV file.

- We plotted KDEs for numerical features before and after removing outliers based on predefined thresholds. Only `135` rows were removed from over `17,000`, which is insignificant.

- The thresholds could have been more aggressive to further reduce skewness, but this risked removing too many minority-class data points, leading to inaccurate predictions.

- We calculated and displayed the Variance Inflation Factor (VIF) for numerical variables. After dropping 'practice_exams_started,' which had the highest VIF (`10.20`), we recalculated the VIF values and found the highest remaining value was `3.17`.

- We replaced missing values in 'student_country' with 'Unknown,' split the data into training and testing sets stratified by 'purchased,' and encoded 'student_country' using OrdinalEncoder.


## III. Comparing & Evaluating the Models

Here are the classification reports for each trained model: 

| Logistic Regression Model | Precision | Recall | F1-Score | Support |
|---------------------------|-----------|--------|----------|---------|
| Will Not Purchase         | 0.97      | 0.99   | 0.98     | 3201    |
| Will Purchase             | 0.83      | 0.66   | 0.74     | 325     |

| K-Nearest Neighbors Model | Precision | Recall | F1-Score |
|---------------------------|-----------|--------|----------|
| Will Not Purchase         | 0.97      | 0.99   | 0.98     |
| Will Purchase             | 0.84      | 0.71   | 0.77     |

| SVC Model         | Precision | Recall | F1-Score |
|-------------------|-----------|--------|----------|
| Will Not Purchase | 0.97      | 0.98   | 0.97     |
| Will Purchase     | 0.79      | 0.68   | 0.73     |

| Decision Tree Model | Precision | Recall | F1-Score |
|---------------------|-----------|--------|----------|
| Will Not Purchase   | 0.97      | 0.99   | 0.98     |
| Will Purchase       | 0.84      | 0.72   | 0.78     |

| Random Forest Model | Precision | Recall | F1-Score |
|---------------------|-----------|--------|----------|
| Will Not Purchase   | 0.97      | 0.99   | 0.98     |
| Will Purchase       | 0.84      | 0.74   | 0.79     |

Using the random forest model, here are explanations of the metrics:

- **Metrics for "Will Not Purchase" (0.97–0.99):** The model is highly accurate in identifying non-purchasers, with very few misclassifications.

- **Precision (0.84) for "Will Purchase":** When the model predicts a purchase, it is correct `84%` of the time, meaning `16%` are false positives (incorrectly classified as "Will Purchase").

- **Recall (0.74) for "Will Purchase":** Out of all actual purchases, the model correctly identifies `74%`, missing `26%` (false negatives).

- **F1-Score (0.79) for "Will Purchase":** A balance between precision and recall, indicating decent performance but with room for improvement in recall.

Here are the main takeaways:

- Random Forest and Decision Tree are the strongest choices for the "Will Purchase" class due to higher recall and balanced F1-scores. Random Forest slightly outperforms Decision Tree with better recall (`0.74` vs. `0.72`).

- Logistic Regression and SVC have the lowest recall, missing `32-34%` of true "Will Purchase" cases. While KNN improves recall compared to Logistic Regression and SVC, it still falls behind the top models.


## IV. Model Improvement Steps

- Techniques like over and undersampling can help deal with imbalanced datasets. Oversampling generates new data points from the minority (paying students) class, while undersampling reduces the majority (free-plan students) class.

- Both techniques have limitations. Oversampling creates more data points but doesn't add new information, as the new points are based on existing minority class data. Undersampling reduces the majority class size, potentially losing valuable information.

- To address these issues, hybrid techniques like Synthetic Minority Oversampling Technique (SMOTE) combine over and undersampling to balance the classes.
These are not the only methods to handle imbalanced datasets. Despite the challenges, we were able to train models on our dataset that produced reliable, insightful results.

- Other ways to improve model performance include reducing dimensionality or exploring additional independent variables, such as participation in the Q&A hub, number of logins, or visits to the pricing page.
