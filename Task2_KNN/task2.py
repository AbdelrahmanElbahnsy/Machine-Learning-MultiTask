import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

print("\n===== KNN (Heart Disease) =====")

df = pd.read_csv("Task2_KNN/heart.csv")
df = df.dropna()

# عمود التصنيف هنا اسمه HeartDisease
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# تحويل البيانات النصية لأرقام (One-Hot Encoding) زي كود صاحبك
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# التقييس (Scaling) ضروري جداً للـ KNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))