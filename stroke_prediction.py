###########################################################
# Stroke Prediction
###########################################################

# https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

# 1) id: unique identifier
# 2) gender: "Male", "Female" or "Other"
# 3) age: age of the patient
# 4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
# 5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
# 6) ever_married: "No" or "Yes"
# 7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
# 8) Residence_type: "Rural" or "Urban"
# 9) avg_glucose_level: average glucose level in blood
# 10) bmi: body mass index
# 11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
# 12) stroke: 1 if the patient had a stroke or 0 if not
# *Note: "Unknown" in smoking_status means that the information is unavailable for this patient

# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

# Load dataset
df = pd.read_csv("StrokePrediction/healthcare-dataset-stroke-data.csv")

###############################
# Exploratory Data Analysis
###############################

df.head()
df.tail()
df.shape  # Dataframe shape is (5110, 12)

# Null Values
df.isnull().sum()  # There is 201 null value in bmi
df["bmi"].isnull().sum() / df.shape[0] * 100  # Ratio of null values in bmi is 3.93

plt.bar(["Null", "Not Null"], [df["bmi"].isnull().sum(), df["bmi"].notnull().sum()])

# Categorical Variables
cat_cols = [col for col in df.columns if df[col].dtypes == "O"]

# Numerical Variables
num_cols = [col for col in df.columns if df[col].dtypes != "O"]

# Numerical but Categorical Variables
num_but_cat = [col for col in df.columns if df[col].nunique() < 3 and df[col].dtypes != "O"]

# Adding num_but_car to cat_cols
cat_cols = cat_cols + num_but_cat

# Removing the num_but_cat in num_cols
num_cols = [col for col in num_cols if col not in num_but_cat]

# Categorical but Cardinal Variables
cat_but_car = [col for col in df.columns if df[col].nunique() > 6 and df[col].dtypes == "O"]
# There is no cardinal variables

# Analyzing Categorical Cols

for col in cat_cols:
    print(pd.DataFrame({col: df[col].value_counts(), "Ratio": (df[col].value_counts() / len(df)) * 100}))
    sns.countplot(x=df[col], data=df)
    plt.show(block=True)

# Analyzing Numerical Cols

quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
for col in num_cols:
    print(df[col].describe(quantiles).T)
    df[col].hist()
    plt.xlabel(col)
    plt.show(block=True)

# Check Outlier in Numerical Cols

for col in num_cols:
    q1 = df[col].quantile(0.10)
    q3 = df[col].quantile(0.90)
    interquantile = q3 - q1
    up_limit = q3 + 1.5 * interquantile
    low_limit = q1 - 1.5 * interquantile
    if df[(df[col] > up_limit) | (df[col] < low_limit)].any(axis=None):
        print(col, True)
    else:
        print(col, False)

# There is outliers in bmi

# Correlation

sns.heatmap(df[num_cols].corr(), annot=True, linewidths=0.5, cmap="Greens")
plt.show()
# There is no highly correlated columns

# Target and num_cols

for col in num_cols:
    print(df.groupby("stroke").agg({col: "mean"}), end="\n\n")
    
# Target and cat_cols

for col in cat_cols:
    print(pd.DataFrame({"target_mean": df.groupby(col)["stroke"].mean()}), end="\n\n")

##########################################
# Data Preprocessing & Feature Engineering
##########################################


# Removing null values
df.dropna(inplace=True)

# age => baby, children, young_adult, middle_age, old
df["new_age_cat"] = pd.cut(x=df["age"], bins=[-1, 3, 17, 31, 46, 83], labels=["baby", "children", "young_adult", "middle_age", "old"])

# avg_glucose_level => normal, prediabetes, diabet
df["new_avg_glucose_level_cat"] = pd.cut(x=df["avg_glucose_level"], bins=[54, 140, 190, 272], labels=["normal", "prediabetes", "diabet"])

# bmi => underweight, healthy, overweight, obese
df["new_bmi_cat"] = pd.cut(x=df["bmi"], bins=[10, 18.5, 24.9, 29.9, 34.9, 98], labels=["underweight", "normal", "overweight", "obese", "extremely_obese"])

# Label Encoder
binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"] and df[col].nunique() == 2]
# ['ever_married', 'Residence_type']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-Hot-Encoder
cat_cols = [col for col in cat_cols if col not in "stroke"]
ohe_cols = [col for col in df.columns if 2 < df[col].nunique() < 10]
# ['gender', 'work_type', 'smoking_status', 'hypertension', 'heart_disease']
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
df.head()
# Updating categorical, numerical columns

# Categorical Variables
cat_cols = [col for col in df.columns if df[col].dtypes not in ["float64", "int64", "int32"]]

# Numerical Variables
num_cols = [col for col in df.columns if df[col].dtypes in ["float64", "int64", "int32"]]

# Numerical but Categorical Variables
num_but_cat = [col for col in df.columns if df[col].nunique() < 3 and df[col].dtypes in ["float64", "int64", "int32"]]

# Adding num_but_car to cat_cols
cat_cols = cat_cols + num_but_cat

# Removing the num_but_cat in num_cols
num_cols = [col for col in num_cols if col not in num_but_cat]

# Removing target variable from cat_cols
cat_cols = [col for col in cat_cols if "stroke" not in col]

# Categorical but Cardinal Variables
cat_but_car = [col for col in df.columns if df[col].nunique() > 6 and df[col].dtypes not in ["float64", "int64"]]
# There is no cardinal variables

# Replacing outliers with low or up limit

q1 = df["bmi"].quantile(0.10)
q3 = df["bmi"].quantile(0.90)
interquantile = q3 - q1
up_limit = q3 + 1.5 * interquantile
low_limit = q1 - 1.5 * interquantile
df.loc[(df["bmi"] < low_limit), "bmi"] = low_limit
df.loc[(df["bmi"] > up_limit), "bmi"] = up_limit

# Checking for outliers again

for col in num_cols:
    q1 = df[col].quantile(0.10)
    q3 = df[col].quantile(0.90)
    interquantile = q3 - q1
    up_limit = q3 + 1.5 * interquantile
    low_limit = q1 - 1.5 * interquantile
    if df[(df[col] > up_limit) | (df[col] < low_limit)].any(axis=None):
        print(col, True)
    else:
        print(col, False)

# There is no outliers

# Scaling with Robust Scaler

rs = RobustScaler()
df["age"] = rs.fit_transform(df[["age"]])
df["bmi"] = rs.fit_transform(df[["bmi"]])
df["avg_glucose_level"] = rs.fit_transform(df[["avg_glucose_level"]])
df.head()

#####################
# Model Building
#####################

y = df["stroke"]
X = df.drop(["stroke", "id"], axis=1)

# Logistic Regression: 0.9574255041304127
lr = LogisticRegression()
cv = cross_val_score(lr, X, y, cv=5)
print(cv.mean())

# KNN: 0.9547766006257383
knn = KNeighborsClassifier()
cv = cross_val_score(knn, X, y, cv=5)
print(cv.mean())

# RandomForest: 0.9566106325687036
rf = RandomForestClassifier()
cv = cross_val_score(rf, X, y, cv=5)
print(cv.mean())

# SVC: 0.9574252965198238
svc = SVC()
cv = cross_val_score(svc, X, y, cv=5)
print(cv.mean())

# DecisionTree: 0.9146458889989224
dtc = DecisionTreeClassifier()
cv = cross_val_score(dtc, X, y, cv=5)
print(cv.mean())

# GradientBoosting: 0.9549808894452851
gb = GradientBoostingClassifier()
cv = cross_val_score(gb, X, y, cv=5)
print(cv.mean())

# AdaBoost: 0.9566104249581144
ab = AdaBoostClassifier()
cv = cross_val_score(ab, X, y, cv=5)
print(cv.mean())

# XGBoost: 0.9484623321727902
xgb = XGBClassifier()
cv = cross_val_score(xgb, X, y, cv=5)
print(cv.mean())

# LightGBM: 0.9511099900139307
lgbm = LGBMClassifier()
cv = cross_val_score(lgbm, X, y, cv=5)
print(cv.mean())

#############################
# Hyperparameter Optimization
#############################

# RandomForest: 0.9574252965198238
rf_params = {"n_estimators": [100, 110, 130],
             "min_samples_split": [2, 5, 12, 20],
             "max_depth": [3, 7, 11, None],
             "max_features": ["auto", "sqrt", 2, 5]
             }

rf_best_grid = GridSearchCV(rf, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_final = rf.set_params(**rf_best_grid.best_params_).fit(X, y)

cv = cross_val_score(rf_final, X, y, cv=5)
print(cv.mean())

# KNN: 0.9574252965198238
knn_params = {"n_neighbors": range(6, 17)}

knn_best_grid = GridSearchCV(knn, knn_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
knn_final = knn.set_params(**knn_best_grid.best_params_).fit(X, y)

cv = cross_val_score(knn_final, X, y, cv=5)
print(cv.mean())

# XGBoost: 0.9576289625076037

xgb_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 10],
                  "n_estimators": [100, 150, 200]}

xgb_best_grid = GridSearchCV(xgb, xgb_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
xgb_final = xgb.set_params(**xgb_best_grid.best_params_).fit(X, y)

cv = cross_val_score(xgb_final, X, y, cv=5)
print(cv.mean())

# LightGBM: 0.9570179645442636

lgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 400, 500]}

lgbm_best_grid = GridSearchCV(lgbm, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
lgbm_final = lgbm.set_params(**lgbm_best_grid.best_params_).fit(X, y)

cv = cross_val_score(lgbm_final, X, y, cv=5)
print(cv.mean())

#############################
# Result
#############################

# XGBoost can use with %95.76 score

###############################
# Feature Importance
###############################

importances = pd.DataFrame(data={"Variable": X.columns, "Importance": xgb_final.feature_importances_})
importances = importances.sort_values(by="Importance", ascending=False)
sns.barplot(x=importances["Importance"], y=importances["Variable"])
plt.title("XGBoost Feature Importances")
plt.xlabel("Variables")
plt.ylabel("Importance")
plt.show(block=True)



