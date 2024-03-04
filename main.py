import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("Files/proj_web/company_data.csv")
X = df.drop(columns=["class"])
y = df["class"]

# One-hot encoding on categorical variables, split data
X_encoded = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic R- with class weights
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test_scaled)

# Count the number of individuals predicted to be promoted and not promoted
promoted_count = sum(y_pred == ' >50K')
not_promoted_count = sum(y_pred == ' <=50K')
# print("predicted to be promoted:", promoted_count)
# print("predicted not to be promoted:", not_promoted_count)






# Retrieving coefficients from the logistic regression model
coefficients = model.coef_[0]

# Pair feature names with coefficients
feature_coefficients = dict(zip(X.columns, coefficients))

# Sort feature coefficients by their absolute values
sorted_feature_coefficients = sorted(feature_coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
coefficients_list = []

category, values = [], []
for feature, coefficient in sorted_feature_coefficients:
    category.append(feature); values.append(round(float(coefficient), 2))
    
sorted_feature_coefficients


from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# plt.figure(figsize=(10,6))

# for i, val in enumerate(values):
#     if val > 0:
#         color = (0, 1 - val/2.5, 0)
#     else:
#         color = (1 - abs(val/2.5), 0, 0)
#     plt.barh(category[i], val, color=color)

# plt.xlabel('Coefficient Value')
# plt.ylabel('Feature')
# plt.title('Coefficients of Features')
# plt.savefig("Figures/Coefficient_of_features.png")





# import matplotlib.pyplot as plt

# plt.barh("To be Promoted", promoted_count, color='skyblue')
# plt.barh("Not to be Promoted", not_promoted_count, color='pink')
# plt.xlabel('Employee count')
# # plt.ylabel('')
# plt.title('')
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig("Figures/Result.png")



val = [1,2,3,6,4,5,2,2,7,4]
test = [1,2,3,4,5,6,7,8,9,10]
index = np.arange(len(test))

plot = lambda data: plt.plot(index, data)

plot(val)

# plt.show()

plt.savefig("Files/proj_web/Figures/Test.png")

# from icecream import ic as ic

# ic("hello")

