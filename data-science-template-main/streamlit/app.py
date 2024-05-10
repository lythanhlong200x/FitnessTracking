import json
import joblib
import pandas as pd
import requests
import streamlit as st
from build_data import perform_clustering, preprocess_data
from count_rep import count_reps_and_evaluate
from read_file import read_data_from_files, resample_and_save_data
from run_model import run_model_and_save_predictions
import streamlit.components.v1 as components

# Set page title and layout
st.set_page_config(page_title="Fitness Tracking Data Processing App 💪", layout="wide")


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

    # Page title


st.markdown(
    """
<style>
.centered-title {
    text-align: center;
}
    </style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    '<h1 class="centered-title">Fitness Tracking Data Processing App</h1>',
    unsafe_allow_html=True,
)
lottie_coding = load_lottiefile(
    "./data-science-template-main/streamlit/images/fitness.json"
)
# st_lottie(lottie_coding)

header_image = "./data-science-template-main/streamlit/images/exercises.jpg"
# header_image = "../streamlit/images/exercises.jpg"

# Đường dẫn đến hình ảnh header
st.image(header_image, use_column_width=True)


st.write("## 🗈 Note: Lấy dữ liệu từ ứng dụng Phyphox ")
st.markdown(
    "[![Foo](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRP66YjvR6u01MAM-jSK_0F4Jy_Fn3u7AZZtGtKSIp1&s)](http://google.com.au/)"
)

with st.expander("✨ Hướng dẫn cách lấy dữ liệu gốc"):
    """
    🌟🌟 Bạn có thể dùng điện thoại của mình, hoặc kết nối đồng hồ thông minh có khả năng thu thập 
    data về Gia tốc và Con quay hồi chuyển thông qua app Phyphox. Hãy thực hiện theo các bước sau đây:

    1. **Tải Xuống Phyphox**: Truy cập [trang web Phyphox](https://phyphox.org/) hoặc tải ứng dụng Phyphox từ Appstore hoặc Google Play.
    
    2. **Mở Phyphox**: Khởi động ứng dụng Phyphox trên điện thoại thông minh của bạn.
    
    3. **Chọn thí nghiệm**: Chọn thí nghiệm bạn muốn thực hiện từ danh sách các thực nghiệm có sẵn, ở đây là Con quay hồi chuyển và Gia tốc (không g)
    
    4. **Tiến hành thí nghiệm**: Tuân theo các hướng dẫn được cung cấp trong ứng dụng để thu thập dữ liệu gia tốc kế và gyro tốc kế.
    
    5. **Xuất Dữ Liệu**: Sau khi thu thập dữ liệu hoàn tất, chọn dấu 3 chấm ở góc trên phải màn hình -> chọn xuất dữ liệu -> chọn CSV (Comma, decimal Point).
    
    6. **Định dạng tệp tin**: Vui lòng đặt tên file theo cú pháp "Tên - Chú thích - Acc (với Gia tốc)/Gyr (với Con quay)" VD: Hoang-1-acc.
    
    6. **Tải Lên Dữ Liệu**: Tải tệp CSV đã xuất lên ứng dụng này để xử lý, bao gồm mỗi set gồm 2 file acceleration & gyroscope.
"""



uploaded_files = st.file_uploader(
    "Upload CSV files", type=["csv"], accept_multiple_files=True
)


# Main Streamlit function
def main():

    # Allow user to upload data file

    if uploaded_files:
        acc_df, gyr_df = read_data_from_files(uploaded_files)
        st.write("Processing uploaded files...")
        new_data, data_resample = view_data(acc_df, gyr_df)
        page_selection = st.sidebar.selectbox(
            "Select Options", ["View Data", "View Reps Count"]
        )

        if page_selection == "View Data":
            st.write("Resampling and saving processed data...")
            st.subheader("View Data After Preprocessed")
            st.subheader("Accelerometer Data:")
            st.dataframe(acc_df.head())
            st.subheader("Gyroscope Data:")
            st.dataframe(gyr_df.head())
            st.write("Data Resampled:")
            st.write(data_resample)
            st.write("Cluster File Content:")
            st.write(new_data)
        elif page_selection == "View Reps Count":
            view_reps_count(new_data, data_resample)


def view_data(acc_df, gyr_df):

    # Resample and save processed data
    data_resample = resample_and_save_data(acc_df, gyr_df)
    data_freq = preprocess_data(data_resample)
    data_resample = resample_and_save_data(acc_df, gyr_df)
    data_freq = preprocess_data(data_resample)
    new_data = perform_clustering(data_freq)
    return new_data,data_resample


def view_reps_count(new_data, data_resample):

    st.subheader("View Reps Count Section")
    loaded_model = joblib.load(
        "./data-science-template-main/streamlit/custom_random_forest_model.pkl"
    )
    df_train = pd.read_pickle(
        "./data-science-template-main/streamlit/TrainingData.pkl"
    )
    data_reps = run_model_and_save_predictions(
        df_train, data_resample, new_data, loaded_model
    )

    # Perform reps counting and evaluation
    st.write("Counting repetitions and evaluating results...")
    summary_by_exercise = count_reps_and_evaluate(data_reps)
    st.subheader("Summary by Exercise:")
    st.dataframe(summary_by_exercise)

    st.success("Processing complete!")


if __name__ == "__main__":
    try:
        main()
    
    except Exception as e:
        st.error("Đã xảy ra lỗi trong quá trình chạy ứng dụng. Vui lòng kiểm tra lại định dạng file của bạn.")
        # st.write(e)
  
 
