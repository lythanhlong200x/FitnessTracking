import numpy as np
import pandas as pd
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema


# Define function to count reps and evaluate results
def count_reps_and_evaluate(df):
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
    # Configure LowPassFilter
    # --------------------------------------------------------------
    fs = 1000 / 200
    LowPass = LowPassFilter()

    # --------------------------------------------------------------
    # Create function to count repetitions
    # --------------------------------------------------------------
    def count_reps(dataset, cutoff=0.4, order=10, column="acc_r"):
        data = LowPass.low_pass_filter(
            dataset,
            col=column,
            sampling_frequency=fs,
            cutoff_frequency=cutoff,
            order=order,
        )
        indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)
        peaks = data.iloc[indexes]
        return len(peaks)

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

    # --------------------------------------------------------------
    # Evaluate the results
    # --------------------------------------------------------------

    # Summary by exercise
    summary_by_exercise = rep_df.groupby(["participant", "label"]).agg(
        {"set": "count", "reps_pred": "sum"}
    )
    summary_by_exercise = summary_by_exercise.rename(
        columns={
            "set": "Total Sets",
            "reps_pred": "Total Reps",
            "participant": "Name",
            "label": "Exercise",
        }
    )
    return summary_by_exercise
    # Display the total sets and reps for each exercise
    # print("Tổng số sets tập và tổng số reps cho mỗi loại bài tập:")
    # print(summary_by_exercise)
