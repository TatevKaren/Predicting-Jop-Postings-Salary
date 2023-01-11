<br>
<p align="center">
   
   <a href="">
        <img src="https://img.shields.io/badge/Case%20Study-Machine%20Learning-orange"></a>
     <a href="">
        <img src="https://img.shields.io/badge/Case%20Study-Salary%20Prediction-orange"></a>
   <a href="">
        <img src="https://img.shields.io/badge/Product-%20Job%20Search%20Engine-yellow"></a>
  
  <a href="">
        <img src="https://img.shields.io/badge/-Success%20Metrics-ff69b4"></a>
  <a href="">
        <img src="https://img.shields.io/badge/-%20KFold%20Cross%20Validation-green"></a>
  <a href="">
        <img src="https://img.shields.io/badge/Programming-Python-blue"></a>
  
  <a href="">
        <img src="https://img.shields.io/badge/-Feature%20Imprtance%20-yellowgreen"></a>
  
  
  
</p>
<br>

# Case Study: Predicting Salaries of Job Postings in Search Engine


#### Predicting Salaries of Job Applications for Job Search Engine Indeed using Machine Learning with Python Implementation

**Why** In this case study we aim to create salary estimate feature for job search engine. This estimated salaries for new job postings can improve the job seekers experience by providing them salary information.

**How** We will use range of Machine Learning Models (Lasso Regression, Random Forest, GBM, XGBoost) to estimate the salaries for given job postings and compare to select the best performing model. Keep in mind that we don't have true salaries to evaluate the model in usual way.

- Defining Business and Data Science Goals
- Data Preprocessing (Descriptive Stat, Data Preprocessing, Encoding)
- Methodology (ML Model Comparison and Description)
- Feature Importance
- Step-by-step Training Process of Machine Learning Model 
- Evaluating Our Machine Learning Models

<br><br>
**Python Code** <a href="https://github.com/TatevKaren/Predicting-Jop-Postings-Salary/tree/main/Code"> here</a>

**Data** <a href="https://github.com/TatevKaren/Predicting-Jop-Postings-Salary/tree/main/Data"> here</a>

**Medium Blog Post** <a href="https://medium.com/@tatev-aslanyan/data-science-case-study-predicting-salaries-of-job-postings-e1cbb4e83054"> here</a>

<br><br>

![figure_boxplots](https://user-images.githubusercontent.com/76843403/209078751-5201e03b-d18c-4fa3-a866-6827908432da.png)


![Image 2022-12-21 at 10 25 PM](https://user-images.githubusercontent.com/76843403/209078775-01c94bb7-d1f2-4097-aa4c-22c62989ae2c.jpg)

<br>

## Training Machine Learning Model 
Here is the step-by-step approach for training Machine Learning Model

### Step 1: Collect and prepare the data
The first step in training a machine learning model is to collect and prepare the data that will be used to train the model. This may involve cleaning the data, handling missing values, and selecting a subset of the data for training.

### Step 2: Split the data into training and validation sets
It is common to split the data into two sets: a training set and a validation set. The model will be trained on the training set and evaluated on the validation set.

### Step 3: Choose a model and set the hyperparameters
Next, you will need to choose a machine learning model and set the hyperparameters. The hyperparameters are the parameters of the model that are not learned from the data during training, such as the learning rate or the regularization strength.

### Step 4: Train the model
The model is trained using the training data and the chosen optimization algorithm. The optimization algorithm updates the model parameters to minimize the loss function, which measures the difference between the predicted output and the true output.

### Step 5: Evaluate the model
After the model is trained, it is evaluated on the validation set to assess its performance. The evaluation metric will depend on the specific problem, such as accuracy for a classification task or mean squared error for a regression task.

### Step 6: Fine-tune the model
If the model is not performing well on the validation set, you may need to adjust the hyperparameters or try a different model. This process is known as model fine-tuning.

### Step 7: Test the final model
Once you are satisfied with the model's performance on the validation set, you can finalize the model and evaluate it on the test set. This will give you an estimate of the model's performance on unseen data.

<br><br>

![figure_FeatureImportance](https://user-images.githubusercontent.com/76843403/209078821-b0f961f6-74f8-4e63-bc08-8584cfb7b436.png)
