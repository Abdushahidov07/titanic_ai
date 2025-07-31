import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce
import plotly.express as px
import numpy as np

# ==============================
# 🛳 Настройка страницы
# ==============================
st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="wide")
st.title('🚢 Titanic Survival Predictor - Обучение и предсказание')
st.write('## Работа с датасетом Titanic')

# ==============================
# 📥 Загрузка данных
# ==============================
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

st.subheader("🔍 10 случайных строк")
st.dataframe(df.sample(10), use_container_width=True)

# ==============================
# 🧹 Обработка данных
# ==============================
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Cabin'] = df['Cabin'].notna().astype(int)  # 1 если есть каюта, 0 если нет
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# ==============================
# 📊 Визуализация данных
# ==============================
st.subheader("📊 Визуализация данных")
col1, col2 = st.columns(2)

with col1:
    fig1 = px.histogram(df, x='Survived', color="Sex", barmode="group", title="Выживание по полу")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(df, x='Pclass', color="Survived", barmode="group", title="Выживание по классу")
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.violin(df, x="Survived", y="Age", color="Survived", box=True, points="all",
                 title="Возраст пассажиров и выживание")
st.plotly_chart(fig3, use_container_width=True)

# ==============================
# 🎯 Подготовка данных
# ==============================
X = df.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Кодирование категориальных признаков
encoder = ce.TargetEncoder(cols=['Sex', 'Embarked'])
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

# ==============================
# 🔍 Подбор гиперпараметров (GridSearch)
# ==============================
st.subheader("⚙️ Подбор гиперпараметров для RandomForest")

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_encoded, y_train)

best_model = grid.best_estimator_
st.write(f"**Лучшие параметры:** {grid.best_params_}")

# ==============================
# 📈 Точность модели и AUC
# ==============================
acc_train = accuracy_score(y_train, best_model.predict(X_train_encoded))
acc_test = accuracy_score(y_test, best_model.predict(X_test_encoded))
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_encoded)[:, 1])

st.write(f"**Train Accuracy:** {acc_train:.2f}")
st.write(f"**Test Accuracy:** {acc_test:.2f}")
st.write(f"**ROC-AUC:** {roc_auc:.2f}")

# ==============================
# 🎛 Форма для предсказания
# ==============================
with st.sidebar.form("prediction_form"):
    st.subheader("🔮 Предсказание выживания")
    pclass_input = st.selectbox("Класс билета", sorted(df['Pclass'].unique()))
    sex_input = st.selectbox("Пол", df['Sex'].unique())
    age_input = st.slider("Возраст", 0, 80, int(df['Age'].median()))
    sibsp_input = st.slider("SibSp (Братья/Сёстры)", 0, int(df['SibSp'].max()), 0)
    parch_input = st.slider("Parch (Родители/Дети)", 0, int(df['Parch'].max()), 0)
    fare_input = st.slider("Стоимость билета", float(df['Fare'].min()), float(df['Fare'].max()), float(df['Fare'].median()))
    embarked_input = st.selectbox("Порт посадки", df['Embarked'].unique())
    cabin_input = st.selectbox("Каюта указана?", [0, 1])

    submit_button = st.form_submit_button("Предсказать")

if submit_button:
    user_input = pd.DataFrame([{
        'Pclass': pclass_input,
        'Sex': sex_input,
        'Age': age_input,
        'SibSp': sibsp_input,
        'Parch': parch_input,
        'Fare': fare_input,
        'Embarked': embarked_input,
        'Cabin': cabin_input
    }])

    user_encoded = encoder.transform(user_input)
    user_encoded = user_encoded[X_train_encoded.columns]

    prediction = best_model.predict(user_encoded)[0]
    proba = best_model.predict_proba(user_encoded)[0]

    result = "✅ Выжил" if prediction == 1 else "❌ Не выжил"
    st.sidebar.markdown(f"### Результат: **{result}**")

    proba_df = pd.DataFrame({'Класс': ['Не выжил', 'Выжил'], 'Вероятность': proba})
    st.sidebar.dataframe(proba_df.set_index("Класс"), use_container_width=True)
