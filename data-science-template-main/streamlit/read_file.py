from datetime import datetime, timedelta
from io import StringIO

import pandas as pd


# Function to read data from files
def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1
    time_create = datetime.now()
    delta_time = timedelta(days=7)

    for i, uploaded_file in enumerate(files):
        content = uploaded_file.getvalue().decode("utf-8")
        df = pd.read_csv(StringIO(content))
        if uploaded_file.name:
            participant = uploaded_file.name.split("-")[0]
        else:
            participant = "Customer"
        time_create -= (i % 2 - 1) * delta_time

        time_epoch = time_create.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        datetime_epoch = datetime.strptime(time_epoch, "%Y-%m-%d %H:%M:%S.%f")

        def add_epoch(time_in_seconds):
            timedelta = pd.to_timedelta(time_in_seconds, unit="s")
            return datetime_epoch + timedelta

        df["epoch (ms)"] = df["Time (s)"].apply(add_epoch)

        df["participant"] = participant

        if df.columns.str.contains("Acceleration").any():
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        if df.columns.str.contains("Gyroscope").any():  # Sửa dòng này
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])
    acc_df.set_index("epoch (ms)", inplace=True)
    gyr_df.set_index("epoch (ms)", inplace=True)

    del acc_df["Time (s)"]
    del gyr_df["Time (s)"]

    return acc_df, gyr_df


def resample_and_save_data(acc_df, gyr_df):
    # Merge accelerometer and gyroscope data
    data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
    column_mapping = [
        "acc_x",
        "acc_y",
        "acc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "participant",
        "set",
    ]
    columns_df = list(data_merged.columns)

    # Thực hiện đổi tên cột bắt đầu từ cột thứ hai
    for i, new_name in enumerate(column_mapping):
        if i < len(columns_df):
            columns_df[i] = new_name
    # Rename columns
    data_merged.columns = columns_df
    # data_merged.columns = column_mapping

    # Resample data
    sampling = {
        "acc_x": "mean",
        "acc_y": "mean",
        "acc_z": "mean",
        "gyr_x": "mean",
        "gyr_y": "mean",
        "gyr_z": "mean",
        "participant": "last",
        "set": "last",
    }
    days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
    data_resampled = pd.concat(
        [df.resample(rule="100ms").apply(sampling).dropna() for df in days]
    )

    # Save processed data
    data_resampled["set"] = data_resampled["set"].astype("int")
    # data_resampled.to_pickle("0205_data_real_processed.pkl")

    # Display the content of the pickle file
    # st.write("Data Resampled:")
    # st.write(data_resampled)
    return data_resampled
