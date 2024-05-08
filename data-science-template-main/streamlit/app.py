# import streamlit as st
# import pandas as pd
# import joblib
# from datetime import datetime
# from datetime import timedelta
# from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
# from TemporalAbstraction import NumericalAbstraction
# from FrequencyAbstraction import FourierTransformation
# from sklearn.cluster import KMeans
# from io import StringIO
# from build_data import preprocess_data, perform_clustering, perform_clustering
# from run_model import run_model_and_save_predictions
# from read_file import read_data_from_files, resample_and_save_data
# from count_rep import count_reps_and_evaluate
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import seaborn as sns
# import itertools
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.metrics import classification_report
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score


# # Main Streamlit function
# def main():
#     st.title("Data Processing App")

#     # Allow user to upload data file
#     uploaded_files = st.file_uploader(
#         "Upload CSV files", type=["csv"], accept_multiple_files=True
#     )

#     if uploaded_files:
#         st.write("Processing uploaded files...")

#         # Process data from uploaded files
#         acc_df, gyr_df = read_data_from_files(uploaded_files)

#         # Display processed data
#         st.write("Accelerometer Data:")
#         st.write(acc_df.head())

#         st.write("Gyroscope Data:")
#         st.write(gyr_df.head())

#         st.write("Resampling and saving processed data...")

#         # Resample and save processed data

#         data_resample = resample_and_save_data(acc_df, gyr_df)
#         data_freq = preprocess_data(data_resample)
#         new_data = perform_clustering(data_freq)
#         loaded_model = joblib.load(
#             "D:\\AI\FitnessTracking\\FitnessTracking\\data-science-template-main\\streamlit\\custom_random_forest_model.pkl"
#         )
#         df_train = pd.read_pickle(
#             "D:\\AI\\FitnessTracking\\FitnessTracking\\data-science-template-main\\streamlit\\trainingRealdata.data_features_real.pkl"
#         )
#         train_accuracy, pred_test_y, data_reps = run_model_and_save_predictions(
#             df_train, data_resample, new_data, loaded_model
#         )
#         st.write("Train accuracy:", train_accuracy)
#         # st.write("Predicted labels:", pred_test_y)
#         summary_by_exercise = count_reps_and_evaluate(data_reps)
#         st.write("Total sets: ", summary_by_exercise)
#         st.success("Processing complete!")


# if __name__ == "__main__":
#     main()


import streamlit as st
import pandas as pd
import joblib
import json
import requests
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
from io import StringIO
from streamlit_lottie import st_lottie
from build_data import preprocess_data, perform_clustering
from run_model import run_model_and_save_predictions
from read_file import read_data_from_files, resample_and_save_data
from count_rep import count_reps_and_evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set page title and layout
st.set_page_config(page_title="Fitness Tracking Data Processing App", layout="wide")


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


st.write("## ** Note: Lấy dữ liệu từ ứng dụng Phyphox **")
st.markdown(
    "[![Foo](data:image/png;base64,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)](http://google.com.au/)"
)

with st.expander("Hướng dẫn cách lấy dữ liệu gốc"):
    """
    Bạn có thể dùng điện thoại của mình, hoặc kết nối đồng hồ thông minh có khả năng thu thập 
    data về Gia tốc và Con quay hồi chuyển thông qua app Phyphox. Hãy thực hiện theo các bước sau đây:

    1. **Tải Xuống Phyphox**: Truy cập [trang web Phyphox](https://phyphox.org/) hoặc tải ứng dụng Phyphox từ Appstore hoặc Google Play.
    
    2. **Mở Phyphox**: Khởi động ứng dụng Phyphox trên điện thoại thông minh của bạn.
    
    3. **Chọn thí nghiệm**: Chọn thí nghiệm bạn muốn thực hiện từ danh sách các thực nghiệm có sẵn, ở đây là Con quay hồi chuyển và Gia tốc (không g)
    
    4. **Tiến hành thí nghiệm**: Tuân theo các hướng dẫn được cung cấp trong ứng dụng để thu thập dữ liệu gia tốc kế và gyro tốc kế.
    
    5. **Xuất Dữ Liệu**: Sau khi thu thập dữ liệu hoàn tất, chọn dấu 3 chấm ở góc trên phải màn hình -> chọn xuất dữ liệu -> chọn CSV (Comma, decimal Point).
    
    6. **Định dạng tệp tin**: Vui lòng đặt tên file theo cú pháp "Tên - Chú thích - Acc (với Gia tốc)/Gyr (với Con quay)" VD: Hoang-1-acc.
    
    6. **Tải Lên Dữ Liệu**: Tải tệp CSV đã xuất lên ứng dụng này để xử lý.
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

        # Process data from uploaded files

        # tab_selection = st.sidebar.radio(
        #     "Select Section", ["View Data", "View Reps Count"]
        # )
        new_data, data_resample = view_data(acc_df, gyr_df)
        page_selection = st.sidebar.selectbox(
            "Select Options", ["View Data", "View Reps Count"]
        )

        if page_selection == "View Data":
            st.write("Resampling and saving processed data...")
            st.subheader("View Data Section")
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

    return new_data, data_resample


def view_reps_count(new_data, data_resample):

    st.subheader("View Reps Count Section")
    loaded_model = joblib.load(
        "./data-science-template-main/streamlit/custom_random_forest_model.pkl"
    )
    df_train = pd.read_pickle(
        "./data-science-template-main/streamlit/trainingRealdata.data_features_real.pkl"
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
    main()
