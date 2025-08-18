import logging
from pathlib import Path
from typing import Iterable, Tuple, Optional

import datasets
import numpy as np
import pandas as pd
import typer
import yaml
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import QuantileForecast
from tqdm.auto import tqdm
from gluonts.time_feature import get_seasonality

app = typer.Typer(pretty_exceptions_enable=False)


def to_gluonts_univariate(
    hf_dataset: datasets.Dataset,
) -> Tuple[list, str]:
    """
    将 HuggingFace Dataset 转为 GluonTS 风格的列表，并返回数据频率字符串
    """
    series_fields = [
        col
        for col in hf_dataset.features
        if isinstance(hf_dataset.features[col], datasets.Sequence)
    ]
    series_fields.remove("timestamp")
    dataset_freq = pd.DatetimeIndex(hf_dataset[0]["timestamp"]).to_period()[0].freqstr

    gts_dataset = []
    for entry in hf_dataset:
        for field in series_fields:
            gts_dataset.append(
                {
                    "start": pd.Period(entry["timestamp"][0], freq=dataset_freq),
                    "target": entry[field],
                }
            )
    return gts_dataset, dataset_freq


def load_and_split_dataset(backtest_config: dict) -> Tuple:
    """
    加载 HF 数据集并生成分割后的测试集
    """
    hf_repo = backtest_config["hf_repo"]
    dataset_name = backtest_config["name"]
    offset = backtest_config["offset"]
    prediction_length = backtest_config["prediction_length"]
    num_rolls = backtest_config["num_rolls"]

    trust_remote_code = hf_repo == "autogluon/chronos_datasets_extra"
    ds = datasets.load_dataset(
        hf_repo, dataset_name, split="train", trust_remote_code=trust_remote_code
    )
    ds.set_format("numpy")

    gts_dataset, dataset_freq = to_gluonts_univariate(ds)
    _, test_template = split(gts_dataset, offset=offset)
    test_data = test_template.generate_instances(
        prediction_length=prediction_length,
        windows=num_rolls,
    )
    return test_data, dataset_freq


# 替换这部分代码（第85-102行）
def generate_seasonal_naive_forecasts(
    test_entries: Iterable,
    prediction_length: int,
    quantiles: np.ndarray,
    dataset_freq: str,  # 保持这个新增参数
) -> list:
    """
    对每条序列做季节性朴素预测：重复历史最后一个周期的值。
    先填充 NaN，避免评估出错。
    返回 QuantileForecast 列表。
    """
    forecasts = []
    q_list = list(map(str, quantiles))

    # 获取正确的季节长度
    season_length = get_seasonality(dataset_freq)

    for ts in tqdm(test_entries, desc="Baseline forecasts"):
        # 原始历史序列
        history = ts["target"]
        # 填充 NaN：前向填充后向填充
        hist_series = pd.Series(history)
        if hist_series.isna().any():
            hist_series = hist_series.ffill().bfill()
        history_clean = hist_series.values

        # 使用季节长度提取最后一个周期
        if len(history_clean) < season_length:
            last_cycle = np.tile(
                history_clean, int(np.ceil(season_length / len(history_clean)))
            )[:season_length]
        else:
            last_cycle = history_clean[-season_length:]

        # 关键修复：根据prediction_length生成正确长度的预测
        if prediction_length <= season_length:
            # 预测长度不超过季节长度
            forecast_pattern = last_cycle[:prediction_length]
        else:
            # 预测长度大于季节长度，重复季节模式
            full_cycles = prediction_length // season_length
            remainder = prediction_length % season_length
            forecast_pattern = np.tile(last_cycle, full_cycles)
            if remainder > 0:
                forecast_pattern = np.concatenate(
                    [forecast_pattern, last_cycle[:remainder]]
                )

        # 构造分位数预测数组（使用正确长度的forecast_pattern）
        forecast_array = np.tile(forecast_pattern, (len(quantiles), 1))
        start = ts["start"] + len(history_clean)

        forecasts.append(
            QuantileForecast(
                forecast_arrays=forecast_array,
                forecast_keys=q_list,
                start_date=start,
            )
        )
    return forecasts


@app.command()
def main(
    config_path: Path,
    metrics_path: Path,
    batch_size: int = typer.Option(32, "--batch-size", help="(ignored)"),
    device: str = typer.Option("cuda", "--device", help="(ignored)"),
    num_samples: int = typer.Option(20, "--num-samples", help="(ignored)"),
    temperature: Optional[float] = typer.Option(
        None, "--temperature", help="(ignored)"
    ),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="(ignored)"),
    top_p: Optional[float] = typer.Option(None, "--top-p", help="(ignored)"),
):
    """
    使用季节性朴素基线评估多个数据集，输出 CSV。兼容原有 CLI 参数。
    """
    with open(config_path) as fp:
        backtest_configs = yaml.safe_load(fp)

    result_rows = []
    quantiles = np.arange(0.1, 1.0, 0.1)

    for config in backtest_configs:
        dataset_name = config["name"]
        prediction_length = config["prediction_length"]

        logger.info(f"Loading dataset {dataset_name} ...")
        test_data, dataset_freq = load_and_split_dataset(config)

        logger.info(f"Generating Seasonal-Naive forecasts for {dataset_name} ...")
        forecasts = generate_seasonal_naive_forecasts(
            test_entries=test_data.input,
            prediction_length=prediction_length,
            quantiles=quantiles,
            dataset_freq=dataset_freq,  # 传递频率参数
        )

        logger.info(f"Evaluating forecasts for {dataset_name} ...")
        metrics = (
            evaluate_forecasts(
                forecasts=forecasts,
                test_data=test_data,
                metrics=[
                    MASE(),
                    MeanWeightedSumQuantileLoss(quantiles),
                ],
                batch_size=5000,
            )
            .reset_index(drop=True)
            .to_dict(orient="records")
        )

        result_rows.append(
            {
                "dataset": dataset_name,
                "model": "seasonal_naive",
                **metrics[0],
            }
        )

    results_df = (
        pd.DataFrame(result_rows)
        .rename(
            {"MASE[0.5]": "MASE", "mean_weighted_sum_quantile_loss": "WQL"},
            axis="columns",
        )
        .sort_values(by="dataset")
    )
    results_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved evaluation metrics to {metrics_path}")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("SeasonalNaive Evaluation")
    logger.setLevel(logging.INFO)
    app()
