import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/2604.data_with_predictions.pkl")
df = df[df["label"] != "rest"]

acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
gyr_r = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2
df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)
# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
bench_df = df[df["label"] == "bench"]
squat_df = df[df["label"] == "squat"]
row_df = df[df["label"] == "row"]
ohp_df = df[df["label"] == "ohp"]
dead_df = df[df["label"] == "dead"]

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------
# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------
fs = 1000 / 200
LowPass = LowPassFilter()
# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------
bench_set = pd.DataFrame()  # Khởi tạo DataFrame rỗng
squat_set = pd.DataFrame()  # Khởi tạo DataFrame rỗng
row_set = pd.DataFrame()  # Khởi tạo DataFrame rỗng
ohp_set = pd.DataFrame()  # Khởi tạo DataFrame rỗng
dead_set = pd.DataFrame()  # Khởi tạo DataFrame rỗng

unique_values_bench = bench_df["set"].unique()
unique_values_squat = squat_df["set"].unique()
unique_values_row = row_df["set"].unique()
unique_values_ohp = ohp_df["set"].unique()
unique_values_dead = dead_df["set"].unique()

# Kiểm tra và lọc DataFrame bench
if len(unique_values_bench) > 0:
    first_unique_value_bench = unique_values_bench[0]
    condition_bench = bench_df["set"] == first_unique_value_bench
    bench_set = bench_df[condition_bench]

# Kiểm tra và lọc DataFrame squat
if len(unique_values_squat) > 0:
    first_unique_value_squat = unique_values_squat[0]
    condition_squat = squat_df["set"] == first_unique_value_squat
    squat_set = squat_df[condition_squat]

# Kiểm tra và lọc DataFrame row
if len(unique_values_row) > 0:
    first_unique_value_row = unique_values_row[0]
    condition_row = row_df["set"] == first_unique_value_row
    row_set = row_df[condition_row]

# Kiểm tra và lọc DataFrame ohp
if len(unique_values_ohp) > 0:
    first_unique_value_ohp = unique_values_ohp[0]
    condition_ohp = ohp_df["set"] == first_unique_value_ohp
    ohp_set = ohp_df[condition_ohp]

# Kiểm tra và lọc DataFrame dead
if len(unique_values_dead) > 0:
    first_unique_value_dead = unique_values_dead[0]
    condition_dead = dead_df["set"] == first_unique_value_dead
    dead_set = dead_df[condition_dead]


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------
def count_reps(dataset, cutoff=0.4, order=10, column="acc_r"):
    data = LowPass.low_pass_filter(
        dataset, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order
    )
    indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)
    peaks = data.iloc[indexes]
    fig, ax = plt.subplots()
    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"], "o", color="red")
    ax.set_ylabel(f"{column}_lowpass")
    exercise = dataset["label"].iloc[0].title()
    plt.title(f"{exercise}:{len(peaks)}Reps")
    plt.show()
    return len(peaks)


# if not bench_set.empty:
#     count_reps(bench_set, cutoff=0.4)

# if not squat_set.empty:
#     count_reps(squat_set, cutoff=0.25)

# if not row_set.empty:
#     count_reps(row_set, cutoff=0.65, column="gyr_x")

# if not ohp_set.empty:
#     count_reps(ohp_set, cutoff=0.35)

# if not dead_set.empty:
#     count_reps(dead_set, cutoff=0.4)
# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------
rep_df = (
    df.groupby(["participant", "label", "set"]).size().reset_index(name="reps_pred")
)
rep_df["reps_pred"] = 0

for s in df["set"].unique():
    subset = df[df["set"] == s]
    column = "acc_r"
    if subset["label"].iloc[0] == "bench":
        cutoff = 0.35
    if subset["label"].iloc[0] == "squat":
        cutoff = 0.25
    if subset["label"].iloc[0] == "row":
        cutoff = 0.25
        column = "gyr_x"
    if subset["label"].iloc[0] == "ohp":
        cutoff = 0.25
    if subset["label"].iloc[0] == "dead":
        cutoff = 0.2
    reps = count_reps(subset, cutoff=cutoff, column=column)
    rep_df.loc[rep_df["set"] == s, "reps_pred"] = reps

print(rep_df)
# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

# rep_df.groupby(["label", "category"])[["reps_pred"]].mean().plot.bar()
summary_by_exercise = rep_df.groupby(["label"]).agg(
    {"set": "count", "reps_pred": "sum"}
)

# Đổi tên cột để phản ánh ý nghĩa và đảo ngược thứ tự cột
summary_by_exercise = summary_by_exercise.rename(
    columns={"set": "total_sets", "reps_pred": "total_reps"}
)

# Hiển thị tổng số sets tập và tổng số reps cho mỗi loại bài tập
print("Tổng số sets tập và tổng số reps cho mỗi loại bài tập:")
print(summary_by_exercise)
