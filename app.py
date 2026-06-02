import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# ── 1. 페이지 설정 ──
st.set_page_config(page_title="영통구 집값 예측", page_icon="🏠")
st.title("🏠 영통구 아파트 가격 예측")
st.write("집 조건을 넣으면 예상 거래가격을 알려줘요")

# ── 2. 데이터 불러오고 모델 학습 (앱 켜질 때 1번) ──
@st.cache_data
def load_and_train():
    df = pd.read_csv("아파트_매매_영통구_정리.csv", encoding="utf-8-sig")
    X = df[["전용면적(㎡)", "건축년도", "층"]]
    y = df["거래금액(만원)"]
    model = LinearRegression().fit(X, y)
    return df, model

df, model = load_and_train()

# ── 3. 입력폼 ──
st.subheader("집 조건 입력")
area = st.number_input("전용면적 (㎡)", min_value=10.0, max_value=300.0, value=84.5)
year = st.number_input("건축년도", min_value=1980, max_value=2026, value=2015)
floor = st.number_input("층", min_value=1, max_value=50, value=10)

# ── 4. 예측 버튼 ──
if st.button("예상 가격 보기"):
    new_house = pd.DataFrame({
        "전용면적(㎡)": [area],
        "건축년도": [year],
        "층": [floor]
    })
    pred = model.predict(new_house)[0]
    eok = pred / 10000

    st.subheader("예상 거래가격")
    st.metric(label="예측값", value=f"{pred:,.0f} 만원", delta=f"약 {eok:.1f}억")

    # ── 5. 동네 정보 곁들임 (1일차 분석 연결) ──
    st.subheader("영통구 참고 정보")
    avg_price = df["거래금액(만원)"].mean()
    st.write(f"영통구 평균 거래가: {avg_price:,.0f} 만원")
    if pred > avg_price:
        st.write("이 집은 영통구 평균보다 비싼 편이에요")
    else:
        st.write("이 집은 영통구 평균보다 저렴한 편이에요")