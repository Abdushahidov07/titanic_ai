import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
import plotly.express as px

st.set_page_config(page_title="📡 Telco Customer Churn Predictor", layout="wide")
st.title("📡 Telco Customer Churn Predictor - Обучение и предсказание")
st.write("## Работа с датасетом Telco")

# 1. Загружаем датасет
df = pd.read_excel("https://raw.githubusercontent.com/datasciencedojo/datasets/master/Telco_customer_churn.xlsx")

st.subheader("🔍 10 случайных строк")
st.dataframe(df.sample(10), use_container_width=True)

# 2. Предобработка
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df.dropna(subset=["Total Charges"], inplace=True)
df.drop(["customerID"], axis=1, inplace=True)

# Целевая переменная
y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)
X = df.drop("Churn", axis=1)

# 3. Визуализация
st.subheader("📊 Визуализация данных")
col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(df, x="Churn", color="gender", barmode="group", title="Отток по полу")
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = px.histogram(df, x="Churn", color="Contract", barmode="group", title="Отток по контракту")
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.violin(df, x="Churn", y="tenure", color="Churn", box=True, points="all", title="Отток и стаж клиента")
st.plotly_chart(fig3, use_container_width=True)

# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# 5. Кодирование категориальных признаков
cat_cols = X.select_dtypes(include="object").columns.tolist()
encoder = ce.TargetEncoder(cols=cat_cols)
X_train_enc = encoder.fit_transform(X_train, y_train)
X_test_enc = encoder.transform(X_test)

# 6. Подбор гиперпараметров
st.subheader("⚙️ Подбор гиперпараметров для RandomForest")
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid.fit(X_train_enc, y_train)

best_model = grid.best_estimator_
st.write(f"**Лучшие параметры:** {grid.best_params_}")

# 7. Метрики
acc_train = accuracy_score(y_train, best_model.predict(X_train_enc))
acc_test = accuracy_score(y_test, best_model.predict(X_test_enc))
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_enc)[:, 1])

st.write(f"**Train Accuracy:** {acc_train:.2f}")
st.write(f"**Test Accuracy:** {acc_test:.2f}")
st.write(f"**ROC-AUC:** {roc_auc:.2f}")

# ROC-кривая
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test_enc)[:, 1])
roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
roc_fig = px.line(roc_df, x="FPR", y="TPR", title=f"ROC-кривая (AUC={roc_auc:.2f})")
roc_fig.add_shape(type="line", line=dict(dash="dash", color="red"), x0=0, x1=1, y0=0, y1=1)
st.plotly_chart(roc_fig, use_container_width=True)

# 8. Ввод данных
st.subheader("🔮 Ввод данных для предсказания")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Пол", df["gender"].unique())
        senior = st.selectbox("Senior Citizen", df["SeniorCitizen"].unique())
        partner = st.selectbox("Partner", df["Partner"].unique())
        tenure = st.slider("Стаж (месяцы)", 0, int(df["tenure"].max()), int(df["tenure"].median()))
    with col2:
        phone = st.selectbox("PhoneService", df["PhoneService"].unique())
        multiple = st.selectbox("MultipleLines", df["MultipleLines"].unique())
        internet = st.selectbox("InternetService", df["InternetService"].unique())
        contract = st.selectbox("Contract", df["Contract"].unique())
    with col3:
        monthly = st.slider("MonthlyCharges", float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max()), float(df["MonthlyCharges"].median()))
        total = st.slider("TotalCharges", float(df["TotalCharges"].min()), float(df["TotalCharges"].max()), float(df["TotalCharges"].median()))
        payment = st.selectbox("PaymentMethod", df["PaymentMethod"].unique())
        dependents = st.selectbox("Dependents", df["Dependents"].unique())
    submit_button = st.form_submit_button("Предсказать")

if submit_button:
    user_input = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "Contract": contract,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    user_encoded = encoder.transform(user_input)
    user_encoded = user_encoded[X_train_enc.columns]

    prediction = best_model.predict(user_encoded)[0]
    proba = best_model.predict_proba(user_encoded)[0]

    result = "⚠️ Клиент уйдет" if prediction == 1 else "✅ Клиент останется"
    st.markdown(f"### Результат: **{result}**")

    proba_df = pd.DataFrame({"Класс": ["Останется", "Уйдет"], "Вероятность": proba})
    st.dataframe(proba_df.set_index("Класс"), use_container_width=True)
