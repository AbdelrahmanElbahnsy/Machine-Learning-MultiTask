import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer

print("\n===== Naive Bayes (SMS Spam) =====")

# قراءة الداتا بـ encoding محدد عشان ملفات الـ spam غالباً بترميز مختلف
df = pd.read_csv("Task3_NaiveBayes/spam.csv", encoding='latin-1')
df = df[['v1', 'v2']].rename(columns={'v1': 'Survived', 'v2': 'Text'}) # سميتها Survived عشان تمشي مع كود صاحبك

df = df.dropna()

# تحويل النصوص لمصفوفة أرقام عشان GaussianNB يشتغل
cv = CountVectorizer()
X = cv.fit_transform(df['Text']).toarray()
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))