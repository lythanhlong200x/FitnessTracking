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
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
from io import StringIO
from build_data import preprocess_data, perform_clustering
from run_model import run_model_and_save_predictions
from read_file import read_data_from_files, resample_and_save_data
from count_rep import count_reps_and_evaluate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set page title and layout
st.set_page_config(page_title="Fitness Tracking Data Processing App", layout="wide")


# Main Streamlit function
def main():
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

    header_image = "D:\\AI\\FitnessTracking\\FitnessTracking\\data-science-template-main\\streamlit\\images\\exercises.jpg"
    # header_image = "../streamlit/images/exercises.jpg"

    # Đường dẫn đến hình ảnh header
    st.image(header_image, use_column_width=True)

    # Allow user to upload data file
    uploaded_files = st.file_uploader(
        "Upload CSV files", type=["csv"], accept_multiple_files=True
    )

    if uploaded_files:
        acc_df, gyr_df = read_data_from_files(uploaded_files)
        st.write("Processing uploaded files...")

        # Process data from uploaded files

        # tab_selection = st.sidebar.radio(
        #     "Select Section", ["View Data", "View Reps Count"]
        # )
        new_data, data_resample = view_data(acc_df, gyr_df)
        page_selection = st.sidebar.selectbox("Go to", ["View Data", "View Reps Count"])

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
        "D:\\AI\FitnessTracking\\FitnessTracking\\data-science-template-main\\streamlit\\custom_random_forest_model.pkl"
    )
    df_train = pd.read_pickle(
        "D:\\AI\\FitnessTracking\\FitnessTracking\\data-science-template-main\\streamlit\\trainingRealdata.data_features_real.pkl"
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
