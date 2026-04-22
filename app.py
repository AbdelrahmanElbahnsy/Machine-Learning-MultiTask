import gradio as gr
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') # لمنع أي تحذيرات مزعجة في الـ Terminal

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# 1. تدريب النماذج في الخلفية
# ==========================================
# Task 1: Iris
iris_df = pd.read_csv("Task1_DecisionTree/Iris.csv").dropna()
X1 = iris_df.drop(["Id", "Species"], axis=1, errors='ignore')
model1 = DecisionTreeClassifier().fit(X1.values, iris_df["Species"]) # .values عشان نمنع مشاكل أسماء الأعمدة

# Task 2: Heart
heart_df = pd.read_csv("Task2_KNN/heart.csv").dropna()
X2 = pd.get_dummies(heart_df.drop("HeartDisease", axis=1))
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)
model2 = KNeighborsClassifier(n_neighbors=7).fit(X2_scaled, heart_df["HeartDisease"])
heart_medians = X2.median() # هناخد متوسط بيانات المرضى عشان نكمل بيهم الـ 13 عمود

# Task 3: Spam
spam_df = pd.read_csv("Task3_NaiveBayes/spam.csv", encoding='latin-1')
tfidf = TfidfVectorizer(stop_words='english')
X3 = tfidf.fit_transform(spam_df['v2'])
model3 = MultinomialNB().fit(X3, spam_df['v1'])

# Task 4: Insurance
ins_df = pd.read_csv("Task4_LinearRegression/insurance.csv").dropna()
X4_raw = pd.get_dummies(ins_df.drop("charges", axis=1), drop_first=True)
poly = PolynomialFeatures(degree=2, include_bias=False)
X4_poly = poly.fit_transform(X4_raw)
scaler4 = StandardScaler()
X4_scaled = scaler4.fit_transform(X4_poly)
model4 = LinearRegression().fit(X4_scaled, ins_df["charges"])
ins_medians = X4_raw.median() # متوسط التأمين

# ==========================================
# 2. دوال التوقع (مربوطة بالموديلات الحقيقية 100%)
# ==========================================
def predict_iris(sl, sw, pl, pw):
    # استخدام الموديل الحقيقي مباشرة على الأرقام
    pred = model1.predict([[sl, sw, pl, pw]])[0]
    return f"🌸 نوع الزهرة هو: {pred}"

def predict_heart(age, resting_bp, cholesterol, max_hr):
    # إنشاء مريض افتراضي بمتوسط البيانات، ثم تحديثه ببيانات الواجهة
    user_data = pd.DataFrame([heart_medians], columns=X2.columns)
    for col in user_data.columns:
        if 'age' in col.lower(): user_data[col] = age
        elif 'bp' in col.lower() or 'trest' in col.lower(): user_data[col] = resting_bp
        elif 'chol' in col.lower(): user_data[col] = cholesterol
        elif 'hr' in col.lower() or 'thalach' in col.lower(): user_data[col] = max_hr
    
    scaled = scaler2.transform(user_data)
    pred = model2.predict(scaled)[0]
    return "⚠️ خطر عالي (High Risk)" if pred == 1 else "✅ سليم (Low Risk)"

def predict_spam(text):
    vec = tfidf.transform([text])
    result = model3.predict(vec)[0]
    return "🚫 رسالة إعلانية مزعجة (Spam)" if result == "spam" else "✉️ رسالة طبيعية وآمنة (Ham)"

def predict_insurance(age, bmi, children, smoker):
    # التأكد التام من مطابقة أسماء الأعمدة لتجنب ثبات الرقم
    is_smoker = 1 if smoker == "نعم (Yes)" else 0
    user_data = pd.DataFrame([ins_medians], columns=X4_raw.columns)
    
    for col in user_data.columns:
        if 'age' in col.lower(): user_data[col] = age
        elif 'bmi' in col.lower(): user_data[col] = bmi
        elif 'child' in col.lower(): user_data[col] = children
        elif 'smoker' in col.lower() and 'yes' in col.lower(): user_data[col] = is_smoker
        
    poly_features = poly.transform(user_data)
    scaled = scaler4.transform(poly_features)
    price = model4.predict(scaled)[0]
    return f"💰 التكلفة المتوقعة للتأمين: ${round(price, 2)}"

# ==========================================
# 3. تصميم الواجهة (UI)
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.HTML("<h1 style='text-align: center; color: #2C3E50;'> Abdelrahman El-bahnsy Machine Learning Portfolio</h1>")
    gr.HTML("<p style='text-align: center; color: #7F8C8D;'>مجموعة من 4 نماذج ذكاء اصطناعي مختلفة (Classification & Regression)</p>")
    
    with gr.Tab("🌸 Task 1: Iris Classification"):
        with gr.Row():
            with gr.Column(scale=1):
                sl = gr.Slider(4.0, 8.0, step=0.1, label="طول الكأسية - Sepal Length (cm)")
                sw = gr.Slider(2.0, 5.0, step=0.1, label="عرض الكأسية - Sepal Width (cm)")
                pl = gr.Slider(1.0, 7.0, step=0.1, label="طول البتلة - Petal Length (cm)")
                pw = gr.Slider(0.1, 3.0, step=0.1, label="عرض البتلة - Petal Width (cm)")
                btn1 = gr.Button("🔍 تحليل الزهرة", variant="primary")
            
            with gr.Column(scale=1):
                out1 = gr.Textbox(label="النتيجة (Output)", text_align="center")
                gr.Markdown("**دقة النموذج (Accuracy):** 🎯 100%")

        btn1.click(fn=predict_iris, inputs=[sl, sw, pl, pw], outputs=out1)

    with gr.Tab("❤️ Task 2: Heart Disease Predictor"):
        with gr.Row():
            with gr.Column():
                age = gr.Slider(20, 100, step=1, label="العمر (Age)")
                resting_bp = gr.Slider(90, 200, step=1, label="ضغط الدم (Resting BP)")
                chol = gr.Slider(100, 400, step=1, label="الكوليسترول (Cholesterol)")
                max_hr = gr.Slider(60, 220, step=1, label="أقصى معدل ضربات قلب (Max HR)")
                btn2 = gr.Button("⚕️ فحص المريض", variant="primary")
            
            with gr.Column():
                out2 = gr.Textbox(label="التشخيص المبدئي", text_align="center")
                gr.Markdown("**دقة النموذج (Accuracy):** 🎯 86%")

        btn2.click(fn=predict_heart, inputs=[age, resting_bp, chol, max_hr], outputs=out2)

    with gr.Tab("🛡️ Task 3: Spam Detector"):
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(lines=4, label="نص الرسالة (SMS Text)")
                btn3 = gr.Button("🕵️‍♂️ فحص الرسالة", variant="primary")
                
                # إضافة 5 أمثلة قوية جداً ومنوعة
                gr.Examples([
                    "URGENT: Your bank account has been locked. Click here to verify your identity.", # Spam
                    "Hey bro, are we still meeting for lunch today at 2 PM?", # Ham
                    "Congratulations! You won a $1000 Walmart gift card. Text WIN to 5555.", # Spam
                    "Don't forget to buy some milk and eggs on your way home.", # Ham
                    "Limited time offer! Get 90% off on all items. Click the link below." # Spam
                ], inputs=[msg], label="أمثلة جاهزة للتجربة (اختر أي رسالة):")
            
            with gr.Column():
                out3 = gr.Textbox(label="تصنيف الرسالة", text_align="center")
                gr.Markdown("**دقة النموذج (Accuracy):** 🎯 90%")

        btn3.click(fn=predict_spam, inputs=[msg], outputs=out3)

    with gr.Tab("💵 Task 4: Insurance Cost"):
        with gr.Row():
            with gr.Column():
                ins_age = gr.Slider(18, 65, step=1, label="العمر (Age)")
                ins_bmi = gr.Slider(15.0, 50.0, step=0.1, label="مؤشر كتلة الجسم (BMI)")
                ins_child = gr.Slider(0, 5, step=1, label="عدد الأطفال (Children)")
                ins_smoker = gr.Dropdown(["نعم (Yes)", "لا (No)"], label="هل يدخن؟ (Smoker)")
                btn4 = gr.Button("🧮 حساب التكلفة", variant="primary")
            
            with gr.Column():
                out4 = gr.Textbox(label="التسعير الآلي", text_align="center")
                gr.Markdown("**دقة النموذج (R2 Score):** 🎯 86.6%")

        btn4.click(fn=predict_insurance, inputs=[ins_age, ins_bmi, ins_child, ins_smoker], outputs=out4)

if __name__ == "__main__":
    demo.launch()