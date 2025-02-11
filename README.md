# Heart Disease Prediction using Machine Learning

## Overview
This project implements a **Heart Disease Prediction** model using **Logistic Regression** with the `sklearn` library. The model is trained on medical data to predict whether a person has heart disease based on various health indicators.

## Dataset
The dataset used for training the model contains medical records with features such as:
- Age
- Gender
- Blood Pressure
- Cholesterol Levels
- Heart Rate
- Other clinical parameters

## Technologies Used
- **Python**
- **Scikit-Learn (sklearn)**
- **Pandas**
- **NumPy**
- **Matplotlib & Seaborn** (for visualization)
- **Jupyter Notebook / Google Colab** (for experimentation)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/rudranarayan-01/heart-disease-prediction.git
   cd heart-disease-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is not available, install the required libraries manually:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

## Implementation Steps
1. **Load Dataset**: The dataset is loaded using `pandas`.
2. **Data Preprocessing**:
   - Handling missing values
   - Encoding categorical features
   - Feature scaling
3. **Exploratory Data Analysis (EDA)**:
   - Visualizing data distributions
   - Identifying correlations
4. **Model Training**:
   - Splitting data into training and testing sets
   - Training Logistic Regression model
   - Hyperparameter tuning
5. **Model Evaluation**:
   - Accuracy, Precision, Recall, F1-score, Confusion Matrix
   - ROC Curve and AUC score

## Usage
Run the model using:
```bash
python heart_disease_prediction.py
```

Alternatively, run it in Jupyter Notebook:
```python
!jupyter notebook
```
Open the notebook and execute the cells step by step.

## Model Performance
- Achieved **X% accuracy** on the test set.
- Evaluated using Precision, Recall, F1-score, and AUC-ROC curve.

## Results & Visualizations
- Confusion Matrix
- Feature Importance
- ROC Curve

## Future Enhancements
- Implementing other ML models like Random Forest, SVM, or Neural Networks for comparison.
- Deploying as a web application using Flask/Django.
- Using deep learning models for improved accuracy.

## Contributing
Feel free to fork this repository and contribute improvements! Submit a pull request with changes.

## License
This project is licensed under the MIT License.

---
### Contact
**Name** - Rudranarayan Sahu

For any queries, reach out at: [rudranarayansahu.tech@gmail.com]

**Website** - https://akash0101.pythonanywhere.com
