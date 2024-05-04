import pandas as pd
from glob import glob
import os
from datetime import datetime
from datetime import timedelta

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
# single_file_acc = pd.read_csv(
#     "C:\\Users\\Suirad\\Documents\\Zalo Received Files\\2024-04-1519.02.25.csv"
# )
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------


# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------


# f = files[1]
# participant = f.split("-")[0].replace(data_path, "")
# label = f.split("-")[1]
# category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")
# df = pd.read_csv(f)
# df["participant"] = participant
# df["label"] = label
# df["category"] = category
# df.head()
# # --------------------------------------------------------------
# # Read all files
# # --------------------------------------------------------------


# # --------------------------------------------------------------
# # Working with datetimes
# # --------------------------------------------------------------
# acc_df.info()
# acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
# gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

# del acc_df["epoch (ms)"]
# del acc_df["time (01:00)"]
# del acc_df["elapsed (s)"]

# del gyr_df["epoch (ms)"]
# del gyr_df["time (01:00)"]
# del gyr_df["elapsed (s)"]

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
files = glob("../../data/raw/test/*.csv")
data_path = "../../data/raw/test\\"


def read_data_drom_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1
    time_create = datetime.now()
    delta_time = timedelta(days=7)  # Định nghĩa thời gian cần trừ

    for i, f in enumerate(files):
        participant = f.split("-")[0].replace(data_path, "")
        # label = f.split("-")[1]
        # category = f.split("-")[2]
        time_create -= (i % 2 - 1) * delta_time  # Trừ đi 7 ngày sau mỗi 2 file

        # Tính thời gian delta

        # # Chuyển đổi thời gian thành dạng datetime
        time_epoch = time_create.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        df = pd.read_csv(f)
        datetime_epoch = datetime.strptime(time_epoch, "%Y-%m-%d %H:%M:%S.%f")

        def add_epoch(time_in_seconds):
            timedelta = pd.to_timedelta(time_in_seconds, unit="s")
            return datetime_epoch + timedelta

        # Tạo một timedelta từ giá trị time_in_seconds

        # Cộng timedelta với datetime_epoch
        df["epoch (ms)"] = df["Time (s)"].apply(add_epoch)
        df["participant"] = participant
        # df["label"] = label
        # df["category"] = category
        if "-acc" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        if "-gyr" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    # Áp dụng hàm chuyển đổi cho cột 'Thoi_gian_s' và tạo ra cột mới 'Thoi_gian_ms'

    acc_df.set_index("epoch (ms)", inplace=True)
    gyr_df.set_index("epoch (ms)", inplace=True)

    del acc_df["Time (s)"]
    # del acc_df["Absolute acceleration (m/s^2)"]
    # del acc_df["elapsed (s)"]

    del gyr_df["Time (s)"]
    # del gyr_df["Absolute (rad/s)"]
    # del gyr_df["elapsed (s)"]
    return acc_df, gyr_df


acc_df, gyr_df = read_data_drom_files(files)
# acc_df[acc_df["participant"] == "B"]
# gyr_df[gyr_df["participant"] == "B"]
# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

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

# Lấy danh sách tên cột của DataFrame
columns_df = list(data_merged.columns)

# Thực hiện đổi tên cột bắt đầu từ cột thứ hai
for i, new_name in enumerate(column_mapping):
    if i < len(columns_df):
        columns_df[i] = new_name

# Cập nhật lại tên cột cho DataFrame
data_merged.columns = columns_df

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------
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
# data_merged.columns
# data_merged.index = pd.to_timedelta(data_merged.index, unit="ms")

data_merged.columns
data_merged[:10000].resample(rule="200ms").apply(sampling)
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat(
    [df.resample(rule="100ms").apply(sampling).dropna() for df in days]
)


# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz
data_resampled["set"] = data_resampled["set"].astype("int")

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
data_resampled.to_pickle("../../data/interim/squat_data_real_processed.pkl")
