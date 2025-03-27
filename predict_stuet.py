import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# === 데이터 로딩 ===
df = pd.read_excel("pass_data.xlsx")
df.columns = df.columns.str.replace('\n', '').str.strip()

# === 전처리 ===
df['합격여부'] = df['최종결과'].map({'합격': 1, '불합격': 0})
df = df.dropna(subset=['합격여부'])

features = ['전과목성적', '국영수사_과', '지역', '대학계열', '유형', '전형']
target = '합격여부'

X = df[features]
y = df[target]

categorical_cols = ['지역', '대학계열', '유형', '전형']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# === 예측 함수 정의 ===
def predict_prob(total_score, core_score, region, track, admission):
    temp = df.copy()
    if region:
        temp = temp[temp['지역'] == region]
    if track:
        temp = temp[temp['대학계열'] == track]
    if admission:
        temp = temp[temp['전형'] == admission]

    temp['전과목성적'] = total_score
    temp['국영수사_과'] = core_score

    score_column = '전과목성적' if total_score > 0 else '국영수사_과'
    base_score = total_score if total_score > 0 else core_score
    temp = temp[
        (temp[score_column] >= base_score - 0.2) &
        (temp[score_column] <= base_score + 0.2)
    ]
    if temp.empty:
        temp = df.copy()

    input_X = temp[features]
    input_X_processed = preprocessor.transform(input_X)
    probas = model.predict_proba(input_X_processed)[:, 1]
    temp['합격확률'] = probas

    return temp[['대학명', '학과(부)', '유형', '전형', '합격확률']]

# === Streamlit 앱 ===
st.title("새로운 대학 합격 예측 프로그램")
st.write("성적과 조건을 입력하면 합격 가능성이 있는 대학을 추천해줍니다.")

score_type = st.radio("성적 기준 선택", ["전과목성적", "국영수사_과"])
if score_type == "전과목성적":
    total_score = st.number_input("전과목성적", 0.0, 9.0, step=0.1)
    core_score = 0
else:
    core_score = st.number_input("국영수사_과", 0.0, 9.0, step=0.1)
    total_score = 0

region = st.selectbox("지역 선택", [""] + sorted(df['지역'].dropna().unique().tolist()))
track = st.selectbox("대학계열 선택", [""] + sorted(df['대학계열'].dropna().unique().tolist()))
admission = st.selectbox("전형 선택", [""] + sorted(df['전형'].dropna().unique().tolist()))

if st.button("합격 가능성 예측"):
    result = predict_prob(total_score, core_score, region, track, admission)
    result = result.sort_values(by="합격확률", ascending=False).head(10)

    st.subheader("추천 대학(상위 10개)")
    for _, row in result.iterrows():
        pct = row['합격확률'] * 100
        st.markdown(f"""
        <div style='margin-bottom: 8px;'>
            <strong>{row['대학명']} - {row['학과(부)']}</strong><br>
            <div style='width: 100%; background: #eee; border-radius: 4px;'>
                <div style='width: {pct:.1f}%; 
                            background: linear-gradient(to right, red, yellow, green); 
                            padding: 6px 0; 
                            text-align: center; 
                            color: white; 
                            font-weight: bold; 
                            border-radius: 4px;'>
                    {pct:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)