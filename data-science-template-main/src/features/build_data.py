import pandas as pd
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
from TemporalAbstraction import NumericalAbstraction

# Load data
df = pd.read_pickle("../../data/interim/training_outliers.pkl")
predictor_columns = list(df.columns[:6])

# Dealing with missing values (imputation)
for col in predictor_columns:
    df[col] = df[col].interpolate()

# Butterworth lowpass filter
df_lowpass = df.copy()
LowPass = LowPassFilter()
fs = 1000 / 200
cutoff = 1.3
for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]

# Principal component analysis PCA
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

# Sum of squares attributes
df_squared = df_pca.copy()
df_squared["acc_r"] = (
    df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
) ** 0.5
df_squared["gyr_r"] = (
    df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2
) ** 0.5

# Temporal abstraction
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()
predictor_columns = predictor_columns + ["acc_r", "gyr_r"]
ws = int(1000 / 200)
for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, "std")

# Frequency features
df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()
fs = int(1000 / 200)
ws = int(2800 / 200)
df_freq_list = []
for s in df_freq["set"].unique():
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# Dealing with overlapping windows
df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]

# Clustering
df_cluster = df_freq.copy()
cluster_columns = ["acc_x", "acc_y", "acc_z"]
kmeans = KMeans(n_clusters=5, n_init=2, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)
df_cluster
# Export dataset
df_cluster.to_pickle("../../data/interim/TrainingData.pkl")
