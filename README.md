Customer Churn Prediction using Machine Learning 🚀
Overview
This project predicts customer churn (whether a customer will leave the bank) using machine learning algorithms. It leverages Random Forest, Logistic Regression, SVM, and KNN for classification. Advanced feature engineering techniques are used to improve model performance.

🔧 Features
✔ Data Preprocessing (handling categorical data, feature scaling)
✔ Feature Engineering (new features like balance-to-salary ratio, product usage)
✔ Multiple ML Models (Random Forest, Logistic Regression, SVM, KNN)
✔ Performance Evaluation (confusion matrix, classification report, accuracy)
✔ Feature Importance Visualization

🛠 Setup Instructions
1️⃣ Install Dependencies
Make sure you have Python installed, then install required libraries:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn
2️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
3️⃣ Run the Script
bash
Copy
Edit
python churn_prediction.py
📂 Dataset
The dataset used is Churn_Modelling.csv, which contains the following features:

CreditScore: Customer's credit score

Gender: Male/Female

Age: Customer's age

Tenure: Number of years with the bank

Balance: Account balance

NumOfProducts: Number of products used

HasCrCard: Whether the customer has a credit card

IsActiveMember: Whether the customer is an active member

EstimatedSalary: Customer's estimated salary

Exited: Target variable (1 = churned, 0 = retained)

📊 Feature Engineering
To improve prediction accuracy, we added:

BalanceZero: Whether the balance is 0 (binary feature)

BalanceToSalaryRatio: Ratio of balance to estimated salary

ProductUsage: Interaction feature between NumOfProducts and IsActiveMember

TenureGroup: Grouped tenure values

AgeGroup: Binned age values

Male_Germany, Male_Spain: Interaction features

