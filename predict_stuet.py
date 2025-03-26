import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# === 1. 데이터 로딩 및 전처리 ===
# 엑셀 파일 읽기 (파일은 코드와 동일한 폴더 내에 있어야 함)
df = pd.read_excel("pass_data.xlsx")

# 열 이름 정리 (개행 문자 제거)
df.columns = df.columns.str.replace('\n', '').str.strip()

# 합격 여부 인코딩 (합격: 1, 불합격: 0) 및 결측치 제거
df['합격여부'] = df['최종결과'].map({'합격': 1, '불합격': 0})
df = df.dropna(subset=['합격여부'])

# 모델 학습에 사용할 특징 및 타깃 정의
features = ['전과목성적', '국영수사_과', '지역', '대학계열', '유형', '전형']
print("DEBUG - features:", features)
print("DEBUG - df.columns:", df.columns.tolist())
print("Final features list before selection:", features)
X = df[features]
y = df['합격여부']

# 범주형 변수에 대해 One-Hot Encoding 적용 (지역, 대학계열, 유형)
categorical_cols = ['지역', '대학계열', '유형', '전형']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# 전처리 및 학습 데이터 분리
X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# LogisticRegression로 모델 학습
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# === 2. 예측 및 추천 함수 정의 ===
def predict_pass_probabilities(total_score, core_score, region_pref, track_pref, admission_type_pref):
    """
    학생의 성적과 선호 조건을 반영하여 후보 대학(학과, 유형, 전형 정보 포함)에 대한 합격 확률을 예측합니다.
    - total_score, core_score: 학생의 성적 입력
    - region_pref, track_pref, admission_type_pref:
      사용자가 원하는 지역, 대학계열, 유형 선호 (빈 문자열이면 필터링하지 않음)
    """
    # 후보 데이터 필터링: 사용자의 선호 조건과 일치하는 항목만 선택
    filtered_df = df.copy()
    if region_pref:
        filtered_df = filtered_df[filtered_df['지역'] == region_pref]
    if track_pref:
        filtered_df = filtered_df[filtered_df['대학계열'] == track_pref]
    if admission_type_pref:
        filtered_df = filtered_df[filtered_df['전형'] == admission_type_pref]
    
    # 만약 필터링 결과가 없다면 전체 데이터로 예측
    if filtered_df.empty:
        filtered_df = df.copy()
    
    # 학생의 성적 정보(전과목성적, 국영수사_과)를 후보 데이터에 적용
    filtered_df = filtered_df.copy()
    filtered_df['전과목성적'] = total_score
    filtered_df['국영수사_과'] = core_score
    
    # 모델 예측
    input_X = filtered_df[features]
    input_X_processed = preprocessor.transform(input_X)
    probas = model.predict_proba(input_X_processed)[:, 1]
    
    filtered_df['합격확률'] = probas
    # 추천 결과에 대학, 학과, 유형, 전형 정보를 포함하여 반환
    return filtered_df[['대학명', '학과(부)', '유형', '전형', '합격확률']]

def recommend_universities(predictions, thresholds=[0.3, 0.5, 0.7], margin=0.1, top_n=3):
    """
    예측 결과에서 합격확률이 각 threshold(예: 30%, 50%, 70%) ± margin 범위에 해당하는 항목들을
    상위 top_n개씩 추천합니다.
    """
    recommendations = {}
    for threshold in thresholds:
        lower = threshold - margin
        upper = threshold + margin
        filtered = predictions[(predictions['합격확률'] >= lower) & (predictions['합격확률'] <= upper)]
        top_matches = filtered.sort_values(by='합격확률', ascending=False).head(top_n)
        recommendations[f"{int(threshold*100)}%대 추천"] = top_matches.reset_index(drop=True)
    return recommendations

# === 3. Streamlit 웹앱 구성 ===
st.title("대학 합격 가능성 추천 프로그램")
st.write("학생의 성적과 선호 전형/유형/계열을 기반으로, 합격 가능성이 30%, 50%, 70%인 대학과 학과를 추천합니다.")

# 사이드바에 사용자 입력 폼 구성
st.sidebar.header("학생 정보 입력")
score_type = st.sidebar.radio("성적 기준 선택", ["전과목성적", "국영수사_과"])

if score_type == "전과목성적":
    total_score = st.sidebar.number_input("전과목성적", value=1.5, step=0.1)
    core_score = 0  # or dummy
else:
    core_score = st.sidebar.number_input("국영수사_과", value=1.4, step=0.1)
    total_score = 0  # or dummy

st.sidebar.markdown("### 선호 조건 (필요시 선택)")
region_options = [""] + sorted(df['지역'].dropna().unique().tolist())
region_pref = st.sidebar.selectbox("지역 선호", options=region_options)

track_options = ["", "자연", "인문", "예체능"]
track_pref = st.sidebar.selectbox("대학계열 선호", options=track_options)

admission_type_options = ["", "일반", "기회균형", "농어촌", "기타"]
admission_type_pref = st.sidebar.selectbox("전형 선호", options=admission_type_options)

if st.sidebar.button("합격 가능성 예측 및 추천"):
    # 학생 입력을 바탕으로 후보 대학에 대한 합격 확률 예측
    predictions = predict_pass_probabilities(
        total_score=total_score,
        core_score=core_score,
        region_pref=region_pref,
        track_pref=track_pref,
        admission_type_pref=admission_type_pref
    )
    
    # 예측 결과에서 각 확률대별 추천 (여기서는 ±10% 오차 범위로 설정)
    recs = recommend_universities(predictions, thresholds=[0.3, 0.5, 0.7], margin=0.1, top_n=3)
    
    st.subheader("예측된 후보 데이터 (상위 10개)")
    st.write(predictions.sort_values(by="합격확률", ascending=False).head(10))
    
    # 각 확률대별 추천 결과 출력
    for key, rec_df in recs.items():
        st.subheader(key)
        if rec_df.empty:
            st.write("해당 조건에 맞는 추천 결과가 없습니다.")
        else:
            st.write(rec_df)