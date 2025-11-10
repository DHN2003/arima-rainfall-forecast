"""
======================================================
ỨNG DỤNG WEB DỰ BÁO LƯỢNG MƯA BẰNG ARIMA
------------------------------------------------------
Tác giả: Đỗ Hải Nam - Lớp K4518-CNT1
Mục đích:
    - Cho phép người dùng nhập chuỗi lượng mưa thủ công hoặc tải file CSV.
    - Tự động nhận cột có tên liên quan đến 'Rainfall'.
    - Giới hạn số dòng đọc để tránh lag với file lớn.
    - Dự báo n giá trị tiếp theo bằng mô hình ARIMA(p, d, q).
    - Cho phép tải kết quả dự báo dưới dạng file CSV.
======================================================
"""

import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def load_manual_data(text_input: str) -> pd.Series:
    """Xử lý dữ liệu lượng mưa nhập thủ công."""
    values = [float(x.strip()) for x in text_input.split(",") if x.strip()]
    if len(values) < 4:
        raise ValueError("Cần ít nhất 4 giá trị để mô hình hoạt động.")
    return pd.Series(values)

def find_rainfall_column(columns):
    """Tìm cột có tên gần giống 'Rainfall'."""
    possible_names = ["rainfall", "rain", "rain (mm)", "rain_mm", "rainfall_mm"]
    for col in columns:
        name_lower = col.strip().lower()
        if any(keyword in name_lower for keyword in possible_names):
            return col
    return None

def load_csv_data(uploaded_csv, max_limit: int) -> pd.Series:
    """Đọc dữ liệu lượng mưa từ file CSV, tự động nhận cột liên quan đến 'Rainfall'."""
    try:
        dataframe = pd.read_csv(uploaded_csv, nrows=max_limit)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as csv_err:
        raise ValueError(f"Lỗi khi đọc file CSV: {csv_err}") from csv_err

    colname = find_rainfall_column(dataframe.columns)
    if not colname:
        raise ValueError("Không tìm thấy cột nào chứa dữ liệu lượng mưa (Rainfall).")

    return dataframe[colname].dropna().reset_index(drop=True)

def train_arima_model(data_series: pd.Series, p_val: int, d_val: int,
                      q_val: int, step_count: int):
    """Huấn luyện mô hình ARIMA và tạo dự báo."""
    model = ARIMA(data_series, order=(p_val, d_val, q_val))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=step_count)
    return model_fit, forecast

def plot_forecast(data_series: pd.Series, forecast_values: pd.Series,
                  step_count: int):
    """Vẽ biểu đồ thể hiện dữ liệu đầu vào và dự báo."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(data_series) + 1), data_series,
            marker="o", label="Dữ liệu đầu vào", color="blue")
    ax.plot(range(len(data_series) + 1, len(data_series) + step_count + 1),
            forecast_values, marker="o", linestyle="--",
            color="red", label="Giá trị dự báo")
    ax.set_xlabel("Thời điểm (ngày hoặc bước thời gian)")
    ax.set_ylabel("Lượng mưa (mm)")
    ax.set_title("Kết quả dự báo lượng mưa bằng mô hình ARIMA")
    ax.legend()
    st.pyplot(fig)

# ==========================
# GIAO DIỆN STREAMLIT
# ==========================
st.set_page_config(page_title="Thử nghiệm mô hình ARIMA - Dự báo lượng mưa",
                   layout="centered")
st.title("🌧️ Thử nghiệm mô hình ARIMA dự báo lượng mưa")

st.markdown("""
Ứng dụng cho phép bạn **thử nghiệm mô hình ARIMA(p, d, q)**  
bằng cách **nhập chuỗi lượng mưa thủ công hoặc tải lên file CSV**,  
tự động nhận cột có tên liên quan đến *Rainfall* và giới hạn số dòng để tránh lag.
""")

# -------------------------------
# 1️⃣ Nhập dữ liệu
# -------------------------------
st.subheader("1️⃣ Nhập hoặc tải dữ liệu lượng mưa")

input_mode = st.radio("Chọn cách nhập dữ liệu:", ["Nhập thủ công", "Tải file CSV"])
rainfall_series = None

try:
    if input_mode == "Nhập thủ công":
        rainfall_text = st.text_area(
            "Nhập các giá trị lượng mưa gần đây (mm, cách nhau bởi dấu phẩy):",
            "3.2, 5.1, 0.0, 1.4, 2.8",
        )
        rainfall_series = load_manual_data(rainfall_text)
    else:
        uploaded_file = st.file_uploader("Tải lên file CSV chứa dữ liệu lượng mưa:",
                                         type=["csv"])
        max_limit_rows = st.slider("Giới hạn số dòng đọc từ file (để tránh lag):",
                                   100, 10000, 1000)
        if not uploaded_file:
            st.stop()

        with st.spinner("⏳ Đang đọc file CSV, vui lòng đợi..."):
            rainfall_series = load_csv_data(uploaded_file, max_limit_rows)

        st.success(f"✅ Đã tải {len(rainfall_series)} giá trị lượng mưa từ file.")
        st.line_chart(rainfall_series)

except ValueError as input_err:
    st.error(str(input_err))
    st.stop()

# -------------------------------
# 2️⃣ Cấu hình mô hình
# -------------------------------
st.subheader("2️⃣ Cấu hình mô hình ARIMA")

col1, col2, col3 = st.columns(3)
ar_p = col1.number_input("Bậc AR (p)", 0, 5, 1)
diff_d = col2.number_input("Bậc sai phân (d)", 0, 2, 1)
ma_q = col3.number_input("Bậc MA (q)", 0, 5, 1)
forecast_count = st.slider("Số điểm cần dự báo (n)", 1, 30, 10)

# -------------------------------
# 3️⃣ Kết quả dự báo
# -------------------------------
st.subheader("3️⃣ Kết quả dự báo")

if st.button("🚀 Thực hiện dự báo"):
    try:
        with st.spinner("⏳ Đang huấn luyện mô hình và dự báo..."): 
            fitted_model, forecast_result = train_arima_model(
                rainfall_series, ar_p, diff_d, ma_q, forecast_count
            )

        st.success("✅ Dự báo hoàn tất!")
        plot_forecast(rainfall_series, forecast_result, forecast_count)

        result_df = pd.DataFrame({
            "Bước thời gian": range(len(rainfall_series) + 1,
                                    len(rainfall_series) + forecast_count + 1),
            "Giá trị dự báo (mm)": forecast_result,
        })
        st.dataframe(result_df.set_index("Bước thời gian"))

        # 💾 Tải kết quả dự báo xuống
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="💾 Tải xuống kết quả dự báo (CSV)",
            data=csv_buffer.getvalue(),
            file_name="du_bao_luong_mua.csv",
            mime="text/csv",
        )

        # -------------------------------
        # 4️⃣ Đánh giá mô hình
        # -------------------------------
        st.subheader("📊 Đánh giá mô hình (nếu có dữ liệu thực tế)")
        if len(rainfall_series) > forecast_count:
            prediction = fitted_model.predict(
                start=len(rainfall_series) - forecast_count,
                end=len(rainfall_series) - 1,
            )
            mse_value = mean_squared_error(rainfall_series[-forecast_count:],
                                           prediction)
            st.metric("Sai số trung bình bình phương (MSE)", f"{mse_value:.3f}")
        else:
            st.info("Cần dữ liệu dài hơn để đánh giá sai số.")

    except (ValueError, RuntimeError) as forecast_err:
        st.error(f"⚠️ Lỗi khi thực hiện dự báo: {forecast_err}")

st.info("📘 Bạn có thể thử các giá trị (p,d,q) khác nhau để xem mô hình phản ứng ra sao với dữ liệu lượng mưa.")

#streamlit run dubaoluongmua.py#