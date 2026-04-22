import gradio as gr
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# 1. تدريب النماذج (نفس الأكواد القوية بتاعتك)
# ==========================================
iris_df = pd.read_csv("Task1_DecisionTree/Iris.csv").dropna()
X1 = iris_df.drop(["Id", "Species"], axis=1, errors='ignore')
model1 = DecisionTreeClassifier().fit(X1.values, iris_df["Species"])

heart_df = pd.read_csv("Task2_KNN/heart.csv").dropna()
X2 = pd.get_dummies(heart_df.drop("HeartDisease", axis=1))
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)
model2 = KNeighborsClassifier(n_neighbors=7).fit(X2_scaled, heart_df["HeartDisease"])
heart_medians = X2.median()

spam_df = pd.read_csv("Task3_NaiveBayes/spam.csv", encoding='latin-1')
tfidf = TfidfVectorizer(stop_words='english')
X3 = tfidf.fit_transform(spam_df['v2'])
model3 = MultinomialNB().fit(X3, spam_df['v1'])

ins_df = pd.read_csv("Task4_LinearRegression/insurance.csv").dropna()
X4_raw = pd.get_dummies(ins_df.drop("charges", axis=1), drop_first=True)
poly = PolynomialFeatures(degree=2, include_bias=False)
X4_poly = poly.fit_transform(X4_raw)
scaler4 = StandardScaler()
X4_scaled = scaler4.fit_transform(X4_poly)
model4 = LinearRegression().fit(X4_scaled, ins_df["charges"])
ins_medians = X4_raw.median()

# ==========================================
# 2. دوال التوقع
# ==========================================
def predict_iris(sl, sw, pl, pw):
    pred = model1.predict([[sl, sw, pl, pw]])[0]
    return f"🌸 نوع الزهرة: {pred}"

def predict_heart(age, resting_bp, cholesterol, max_hr):
    user_data = pd.DataFrame([heart_medians], columns=X2.columns)
    for col in user_data.columns:
        if 'age' in col.lower(): user_data[col] = age
        elif 'bp' in col.lower() or 'trest' in col.lower(): user_data[col] = resting_bp
        elif 'chol' in col.lower(): user_data[col] = cholesterol
        elif 'hr' in col.lower() or 'thalach' in col.lower(): user_data[col] = max_hr
    scaled = scaler2.transform(user_data)
    pred = model2.predict(scaled)[0]
    return "⚠️ خطر عالي (High Risk) - يُرجى استشارة طبيب" if pred == 1 else "✅ سليم (Low Risk) - حافظ على صحتك"

def predict_spam(text):
    if not text.strip(): return "يرجى إدخال نص الرسالة أولاً!"
    vec = tfidf.transform([text])
    result = model3.predict(vec)[0]
    return "🚫 رسالة إعلانية مزعجة (Spam)" if result == "spam" else "✉️ رسالة طبيعية وآمنة (Ham)"

def predict_insurance(age, bmi, children, smoker):
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
    return f"💰 التكلفة المتوقعة للتأمين: ${round(price, 2):,}"

# ==========================================
# 3. HTML / CSS / JS (تم الاختراق من الجذور 🔥)
# ==========================================
custom_css = """
/* استيراد خط احترافي */
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700;900&display=swap');

/* 🚀 إجبار Gradio على استخدام ألواننا الأساسية من الجذور */
:root, .dark {
    --body-background-fill: #0B1121 !important; /* لون الخلفية العام */
    --background-fill-primary: #151E32 !important; /* لون المربعات */
    --background-fill-secondary: #1E293B !important; /* لون الجداول */
    --border-color-primary: #334155 !important;
    --text-color: #FFFFFF !important; /* إجبار كل النصوص على الأبيض الناصع */
    --text-color-subdued: #94A3B8 !important; /* رمادي فاتح للنصوص الفرعية */
    --color-accent: #0EA5E9 !important; /* اللون الأزرق السماوي المميز */
    --font: 'Cairo', sans-serif !important;
}

* { font-family: 'Cairo', sans-serif !important; }

/* ========================================= */
/* 🚀 إصلاح التابات اللي فوق (التنقل) */
/* ========================================= */
.tab-nav {
    border-bottom: 2px solid var(--border-color-primary) !important;
}
.tab-nav button {
    color: var(--text-color-subdued) !important;
    font-size: 1.2em !important;
    font-weight: 800 !important;
}
.tab-nav button.selected {
    color: var(--color-accent) !important;
    border-bottom: 3px solid var(--color-accent) !important;
}

/* ========================================= */
/* 🚀 إصلاح جداول الأمثلة نهائياً (الأرقام هتظهر غصب عنها) */
/* ========================================= */
table, .gr-examples table {
    background-color: var(--background-fill-secondary) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
table th, .gr-examples th {
    background-color: #020617 !important; /* أسود كحلي لعناوين الجدول */
    color: var(--color-accent) !important; /* أزرق فاتح */
    font-weight: 900 !important;
    font-size: 1.1em !important;
    padding: 12px !important;
}
table td, .gr-examples td, .gr-examples td *, table td * {
    background-color: transparent !important;
    color: #FFFFFF !important; /* الأرقام هنا بقت أبيض ناصع 100% */
    font-weight: 700 !important;
    font-size: 1.1em !important;
    padding: 10px !important;
    text-align: center !important;
}
table tr:hover, .gr-examples tr:hover td {
    background-color: var(--color-accent) !important;
    color: #FFFFFF !important;
    cursor: pointer !important;
}

/* ========================================= */
/* 🚀 تنسيق المدخلات، الدليل، والأزرار */
/* ========================================= */
input[type="number"], input[type="text"], textarea, .slider {
    background-color: var(--body-background-fill) !important;
    color: var(--color-accent) !important; /* الأرقام اللي بتكتبها زرقا */
    font-weight: bold !important;
    font-size: 1.1em !important;
}

.guide-box {
    background: linear-gradient(to right, #082f49, #0B1121) !important;
    border-right: 4px solid var(--color-accent) !important;
    border-radius: 8px !important;
    padding: 15px !important;
    margin-bottom: 20px !important;
}
.guide-box p, .guide-box ul, .guide-box li { color: #FFFFFF !important; }
.guide-box code { 
    background: #000000 !important; 
    color: #FDE047 !important; /* أصفر فاقع يلفت الانتباه */
    padding: 2px 8px !important; 
    border-radius: 4px !important; 
    border: 1px solid #FDE047 !important;
    font-weight: 900 !important;
}

.output-text textarea {
    color: #10B981 !important; /* لون أخضر فسفوري للنتيجة */
    font-size: 1.4em !important;
    font-weight: 900 !important;
    text-align: center !important;
}

button.primary {
    background: linear-gradient(90deg, #0EA5E9, #2563EB) !important;
    color: white !important;
    font-size: 1.2em !important;
    font-weight: 900 !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3) !important;
    transition: 0.3s !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(14, 165, 233, 0.6) !important;
}
"""

# سكربت جافاسكربت لإجبار المتصفح على الوضع الليلي
custom_js = """
function() {
    console.log("%c🚀 Portfolio Initialized!", "color: #0EA5E9; font-size: 20px; font-weight: bold;");
    // إجبار واجهة Gradio إنها تقرأ الـ Theme بتاعنا ومتبوظش الألوان
    document.body.classList.add('dark'); 
}
"""

# ==========================================
# 4. بناء الواجهة 
# ==========================================
with gr.Blocks(css=custom_css, js=custom_js) as demo:
    
    gr.HTML("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='font-size: 3.5em; margin: 0; color: #0EA5E9; text-shadow: 0 0 10px rgba(14, 165, 233, 0.5);'>🚀 Ahmed's ML Portal</h1>
        <p style='font-size: 1.2em; color: #CBD5E1;'>لوحة تحكم احترافية لنماذج الذكاء الاصطناعي مع دليل الاستخدام التفاعلي</p>
    </div>
    """)
    
    with gr.Tabs(elem_classes="tabs"):
        
        # --- Task 1 ---
        with gr.Tab("🌸 1. Iris Classifier"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="guide-box">
                        <h4 style="color:#0EA5E9; margin-top:0;">💡 دليل الاستخدام (الكتالوج):</h4>
                        <p>اضغط على الأمثلة تحت عشان تطلع أنواع مختلفة:</p>
                        <ul>
                            <li><b>Setosa:</b> الطول <code>5.1</code>, العرض <code>3.5</code>, طول البتلة <code>1.4</code></li>
                            <li><b>Versicolor:</b> الطول <code>5.9</code>, العرض <code>2.8</code>, طول البتلة <code>4.2</code></li>
                            <li><b>Virginica:</b> الطول <code>6.7</code>, العرض <code>3.3</code>, طول البتلة <code>5.7</code></li>
                        </ul>
                    </div>
                    """)
                    sl = gr.Slider(4.0, 8.0, step=0.1, label="طول الكأسية (Sepal Length)")
                    sw = gr.Slider(2.0, 5.0, step=0.1, label="عرض الكأسية (Sepal Width)")
                    pl = gr.Slider(1.0, 7.0, step=0.1, label="طول البتلة (Petal Length)")
                    pw = gr.Slider(0.1, 3.0, step=0.1, label="عرض البتلة (Petal Width)")
                    
                    gr.Examples([[5.1, 3.5, 1.4, 0.2], [5.9, 2.8, 4.2, 1.3], [6.7, 3.3, 5.7, 2.1]], inputs=[sl, sw, pl, pw], label="أمثلة سريعة (اضغط للتجربة):")
                    btn1 = gr.Button("🚀 تحليل الزهرة", variant="primary")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 النتائج والتحليل")
                    out1 = gr.Textbox(label="النتيجة النهائية", elem_classes="output-text")
                    gr.HTML("<hr><p style='color:#10B981; font-weight:bold; font-size:16px;'>الخوارزمية: Decision Tree | الدقة: 100%</p>")
            btn1.click(fn=predict_iris, inputs=[sl, sw, pl, pw], outputs=out1)

        # --- Task 2 ---
        with gr.Tab("❤️ 2. Cardiology AI"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="guide-box">
                        <h4 style="color:#0EA5E9; margin-top:0;">💡 دليل الاستخدام الطبي:</h4>
                        <ul>
                            <li><b>خطر عالي:</b> عمر <code>65</code>، ضغط <code>160</code>، كوليسترول <code>280</code>، نبض <code>120</code></li>
                            <li><b>سليم:</b> عمر <code>30</code>، ضغط <code>110</code>، كوليسترول <code>180</code>، نبض <code>180</code></li>
                        </ul>
                    </div>
                    """)
                    age = gr.Slider(20, 100, step=1, label="العمر")
                    resting_bp = gr.Slider(90, 200, step=1, label="ضغط الدم (Resting BP)")
                    chol = gr.Slider(100, 400, step=1, label="الكوليسترول")
                    max_hr = gr.Slider(60, 220, step=1, label="نبض القلب الأقصى")
                    
                    gr.Examples([[65, 160, 280, 120], [30, 110, 180, 180]], inputs=[age, resting_bp, chol, max_hr], label="أمثلة سريعة (اضغط للتجربة):")
                    btn2 = gr.Button("🩺 فحص المريض", variant="primary")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 التقرير الطبي")
                    out2 = gr.Textbox(label="التشخيص", elem_classes="output-text")
                    gr.HTML("<hr><p style='color:#10B981; font-weight:bold; font-size:16px;'>الخوارزمية: KNN | الدقة: 86%</p>")
            btn2.click(fn=predict_heart, inputs=[age, resting_bp, chol, max_hr], outputs=out2)

        # --- Task 3 ---
        with gr.Tab("🛡️ 3. Spam Firewall"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="guide-box">
                        <h4 style="color:#0EA5E9; margin-top:0;">💡 دليل الفلترة:</h4>
                        <p>اضغط على واحدة من الأمثلة اللي تحت وجربها. الرسائل المزعجة دايماً فيها كلمات زي <code>Win, Free, Cash, Prize</code>.</p>
                    </div>
                    """)
                    msg = gr.Textbox(lines=4, label="نص الرسالة")
                    gr.Examples([
                        "URGENT! You have won a $100,000 Prize! Call now.",
                        "Hey Ahmed, are we meeting at 5 PM today?",
                    ], inputs=[msg], label="أمثلة سريعة (اضغط للتجربة):")
                    btn3 = gr.Button("🔍 فحص أمني", variant="primary")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 حالة الأمان")
                    out3 = gr.Textbox(label="تصنيف الجدار الناري", elem_classes="output-text")
                    gr.HTML("<hr><p style='color:#10B981; font-weight:bold; font-size:16px;'>الخوارزمية: Naive Bayes | الدقة: 90%</p>")
            btn3.click(fn=predict_spam, inputs=[msg], outputs=out3)

        # --- Task 4 ---
        with gr.Tab("💵 4. Insurance Oracle"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div class="guide-box">
                        <h4 style="color:#0EA5E9; margin-top:0;">💡 دليل التسعير المالي:</h4>
                        <p>جرب شخص عمره <code>25</code> مش مدخن <code>لا (No)</code>، وبعدين جرب شخص <code>50</code> سنة ومدخن <code>نعم (Yes)</code> ولاحظ الفرق الرهيب!</p>
                    </div>
                    """)
                    ins_age = gr.Slider(18, 65, step=1, label="العمر")
                    ins_bmi = gr.Slider(15.0, 50.0, step=0.1, label="مؤشر كتلة الجسم (BMI)")
                    ins_child = gr.Slider(0, 5, step=1, label="عدد الأطفال")
                    ins_smoker = gr.Dropdown(["نعم (Yes)", "لا (No)"], value="لا (No)", label="هل يدخن؟")
                    
                    gr.Examples([[25, 22.0, 0, "لا (No)"], [50, 35.5, 2, "نعم (Yes)"]], inputs=[ins_age, ins_bmi, ins_child, ins_smoker], label="أمثلة سريعة (اضغط للتجربة):")
                    btn4 = gr.Button("💸 حساب التكلفة", variant="primary")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 العرض المالي")
                    out4 = gr.Textbox(label="التكلفة", elem_classes="output-text")
                    gr.HTML("<hr><p style='color:#10B981; font-weight:bold; font-size:16px;'>الخوارزمية: Polynomial Regression | R2 Score: 86.6%</p>")
            btn4.click(fn=predict_insurance, inputs=[ins_age, ins_bmi, ins_child, ins_smoker], outputs=out4)

if __name__ == "__main__":
    demo.launch()