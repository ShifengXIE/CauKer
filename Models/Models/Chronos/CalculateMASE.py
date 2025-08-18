import pandas as pd
from scipy import stats

file_naive = "/mnt/data_1/home_tsteam/chronos/scripts/evaluation/results/seasonal-naive-zero-shot.csv"
file_chronos = "/mnt/data_1/home_tsteam/chronos/evaluation/results/chronos-t5-small-zero-shot-CauKer1M.csv"
df_naive = pd.read_csv(file_naive)
df_chronos = pd.read_csv(file_chronos)
div = df_chronos["MASE"] / df_naive["MASE"]
df_naive["dataset"] = df_naive["dataset"].astype(str)
df_div = pd.DataFrame({"dataset": df_naive["dataset"], "division": div})
df_div = df_div.sort_values(by="division", ascending=False)
print(df_div)
mean_div = div.mean()
print("Mean of the division:", mean_div)
print("Geometric mean of the division:", stats.gmean(div))
