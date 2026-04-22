import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

print("===== Decision Tree (Iris Dataset) =====")

# قراءة الملف من المسار الصح بناءً على مكانك في الـ Terminal
df = pd.read_csv("Task1_DecisionTree/Iris.csv")

# مسح القيم الفارغة
df = df.dropna()

# حذف عمود الـ Id لأنه مش "Feature" بيساعد في التوقع
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

X = df.drop("Species", axis=1)
y = df["Species"]

# تقسيم الداتا لـ 80% تدريب و 20% اختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب الموديل
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# التوقع والتقييم
y_pred = model.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))