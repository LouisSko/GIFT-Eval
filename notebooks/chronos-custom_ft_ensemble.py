import sys

sys.path.append("../../")
from autogluon.timeseries import TimeSeriesDataFrame
from pathlib import Path
import os
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import json

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    short_datasets = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
    med_long_datasets = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
    PRETTY_NAMES = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }
    # all_datasets = list(set(short_datasets.split() + med_long_datasets.split()))
    all_datasets = list(set(short_datasets.split()))

    dataset_properties_map = json.load(open("dataset_properties.json"))

    from gift_eval.data import Dataset

    # Step 2: Create a list of (dataset_name, series_length) tuples
    dataset_lengths = [(d, Dataset(name=d, term="medium", to_univariate=False).sum_series_length) for d in all_datasets]

    # Step 3: Sort the list by series_length (ascending or descending)
    sorted_datasets = sorted(dataset_lengths, key=lambda x: x[1], reverse=False)  # ascending
    # sorted_datasets = sorted(dataset_lengths, key=lambda x: x[1], reverse=True)  # descending

    # Optional: extract just the sorted dataset names
    all_datasets = [d for d, _ in sorted_datasets]

    i = all_datasets.index("solar/10T")  # raises ValueError if not found
    all_datasets = all_datasets[i:]
    print(all_datasets)

    # all_datasets = ["bitbrains_rnd/5T"]
    from gluonts.ev.metrics import (
        MAE,
        MAPE,
        MASE,
        MSE,
        MSIS,
        ND,
        NRMSE,
        RMSE,
        SMAPE,
        MeanWeightedSumQuantileLoss,
    )

    # Instantiate the metrics
    metrics = [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    ]

    from src.pipeline.pipeline import ForecastingPipeline
    from src.predictors.chronos import Chronos
    from typing import List
    from gluonts.model.forecast import QuantileForecast
    from src.pipeline.pipeline import ForecastingPipeline
    from src.postprocessors.qr import PostprocessorFastQR
    from src.postprocessors.eqc import PostprocessorEQC
    from src.predictors.chronos import Chronos
    from typing import List, Dict
    from gluonts.model.forecast import QuantileForecast
    import math

    def _batched(iterable, n):
        batch = []
        for x in iterable:
            batch.append(x)
            if len(batch) == n:
                yield batch
                batch = []
        if batch:
            yield batch

    SIZE = "base"
    MODEL_NAME = f"Chronos-Bolt-{SIZE}-Ensemble"
    MODEL_NAME_FULL = f"Chronos-Bolt-{SIZE}-FT_Full"
    MODEL_NAME_LONGOUT = f"Chronos-Bolt-{SIZE}-FT_LongOut"
    MODEL_NAME_FULL_PP_QR = f"Chronos-Bolt-{SIZE}-FT_Full-PP_QuantReg"
    MODEL_NAME_FULL_PP_OS = f"Chronos-Bolt-{SIZE}-FT_Full-PP_Offset"
    MODEL_NAME_LONGOUT_PP_QR = f"Chronos-Bolt-{SIZE}-FT_LongOut-PP_QuantReg"
    MODEL_NAME_LONGOUT_PP_OS = f"Chronos-Bolt-{SIZE}-FT_LongOut-PP_Offset"
    MODEL_LIST = [
        MODEL_NAME,
        MODEL_NAME_FULL,
        MODEL_NAME_LONGOUT,
        MODEL_NAME_FULL_PP_QR,
        MODEL_NAME_FULL_PP_OS,
        MODEL_NAME_LONGOUT_PP_QR,
        MODEL_NAME_LONGOUT_PP_OS,
    ]  # , PP_QR_NAME, PP_EQC_NAME]

    CALIBRATION = True

    import numpy as np
    import torch
    from src.core.timeseries_evaluation import ForecastCollection

    def make_ensemble(results: Dict[str, ForecastCollection], model_keys: List[str], name="Ensemble", weights=None):
        """
        Build an ensemble ForecastCollection from multiple model keys.

        Args:
            results: dict-like mapping model_key -> ForecastCollection
            model_keys: list/tuple of model keys to ensemble
            name: results dict key to store the ensemble under
            weights: optional list/array of non-negative weights (same length as model_keys).
                    If None, uses equal weights.

        Returns:
            ForecastCollection placed into results[name]
        """
        assert len(model_keys) >= 2, "Provide at least two model keys to ensemble."
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            assert weights.shape == (len(model_keys),), "weights must match model_keys length"
            assert np.all(weights >= 0), "weights must be non-negative"
            if weights.sum() == 0:
                raise ValueError("At least one weight must be > 0")
            weights = weights / weights.sum()

        # Use the first model as the structural template
        ref_key = model_keys[0]
        ref_fc_coll = results[ref_key]

        # Sanity: intersect item_ids across all models to avoid missing series
        item_ids = set(ref_fc_coll.item_ids)
        for k in model_keys[1:]:
            item_ids &= set(results[k].item_ids)
        if len(item_ids) == 0:
            raise ValueError("No common item_ids across the selected models.")

        ts = {}
        for item_id in item_ids:
            # Collect per-model prediction arrays with consistent (T, H, Q) shape
            per_model_preds = []

            # Determine horizon order from the reference model and ensure all share it
            ref_ts = results[ref_key].get_time_series_forecast(item_id=item_id)
            horizon_keys = sorted(ref_ts.lead_time_forecasts.keys())

            for k in model_keys:
                fc_k = results[k].get_time_series_forecast(item_id=item_id)

                # Check horizons match
                hk_k = sorted(fc_k.lead_time_forecasts.keys())
                if hk_k != horizon_keys:
                    raise ValueError(f"Horizon mismatch for item_id {item_id} in model '{k}'.")

                # Stack predictions per horizon -> (H, T, Q), then -> (T, H, Q)
                arr = np.stack([fc_k.lead_time_forecasts[h].predictions for h in horizon_keys], axis=0).transpose(1, 0, 2)
                per_model_preds.append(arr)

            models_stack = np.stack(per_model_preds, axis=0)  # (M, T, H, Q)

            if weights is None:
                adj = models_stack.mean(axis=0)  # (T, H, Q)
            else:
                # Weighted average over model axis
                adj = np.average(models_stack, axis=0, weights=weights)  # (T, H, Q)

            # Write predictions back into a deep copy (use the last model as template if you prefer)
            ts_fc = ref_ts.model_copy(deep=True)
            for idx, h in enumerate(horizon_keys):
                fc_h = ts_fc.lead_time_forecasts[h]
                fc_h.predictions = torch.tensor(adj[:, idx, :])

            ts_fc.clear_cache_dir()
            ts[item_id] = ts_fc

        fc_collection = ForecastCollection(item_ids=ts)
        results[name] = fc_collection
        return fc_collection

    class ChronosWrapper:
        def __init__(self, prediction_length, freq):
            self.prediction_length = prediction_length
            lead_times = np.arange(1, self.prediction_length + 1)
            self.pipeline1 = ForecastingPipeline(
                model=Chronos,
                model_kwargs={
                    "name": MODEL_NAME_FULL,
                    "pretrained_model_name_or_path": f"amazon/chronos-bolt-{SIZE}",
                    "device_map": "cuda",
                    "lead_times": lead_times,
                    "finetuning_schedule": ["full"],
                    "finetuning_adjust_pretrained_prediction_length": False,
                    "finetuning_hp_search": False,
                },
                postprocessors=[PostprocessorFastQR, PostprocessorEQC],
                postprocessor_kwargs=[
                    {"n_jobs": 8, "name": MODEL_NAME_FULL_PP_QR},
                    {"n_jobs": 8, "name": MODEL_NAME_FULL_PP_OS},
                ],
                freq=freq,
                output_dir=Path("./results/gift-eval"),
            )
            self.pipeline2 = ForecastingPipeline(
                model=Chronos,
                model_kwargs={
                    "name": MODEL_NAME_LONGOUT,
                    "pretrained_model_name_or_path": f"amazon/chronos-bolt-{SIZE}",
                    "device_map": "cuda",
                    "lead_times": lead_times,
                    "finetuning_schedule": ["full"],
                    "finetuning_adjust_pretrained_prediction_length": True,
                    "finetuning_hp_search": False,
                },
                freq=freq,
                postprocessors=[PostprocessorFastQR, PostprocessorEQC],
                postprocessor_kwargs=[
                    {"n_jobs": 8, "name": MODEL_NAME_LONGOUT_PP_QR},
                    {"n_jobs": 8, "name": MODEL_NAME_LONGOUT_PP_OS},
                ],
                output_dir=Path("./results/gift-eval-2"),
            )

        def train(self, train_data_input: Dataset):
            data_train = TimeSeriesDataFrame(train_data_input)
            print(data_train.shape)
            del train_data_input
            for pipeline in [self.pipeline1, self.pipeline2]:
                val_window_step = min(self.prediction_length, pipeline.predictor.model_internal_prediction_length)
                max_val_windows = max(1, self.prediction_length // val_window_step)
                pipeline.train_predictor_model(data_train, val_window_step=val_window_step, max_val_windows=max_val_windows)

        def predict(self, test_data_input: Dataset, batch_size: int = 1024) -> Dict[str, List[QuantileForecast]]:

            forecasts: Dict[str, List[QuantileForecast]] = {m: [] for m in MODEL_LIST}
            batch_size = 512

            try:
                total_items = len(test_data_input)
                total_batches = math.ceil(total_items / batch_size)
            except TypeError:
                total_items = None
                total_batches = None

            for batch_idx, batch in enumerate(_batched(test_data_input, batch_size), start=1):
                if total_batches is not None:
                    print(f"[{batch_idx}/{total_batches}] Processing batch of size {len(batch)}...")
                else:
                    print(f"[{batch_idx}] Processing batch of size {len(batch)}...")

                result = {}
                for pipeline in [self.pipeline1, self.pipeline2]:
                    df_batch = TimeSeriesDataFrame(batch)
                    result.update(pipeline.generate_forecasts(data_test=df_batch))

                    if CALIBRATION and pipeline.postprocessors is not None:
                        calibration_preds = pipeline.auto_generate_calibration_forecasts(df_batch, max_calibration_samples=200)
                        pipeline.train_postprocessors(calibration_preds[pipeline.predictor.name])
                        result.update(pipeline.apply_postprocessors_to_forecasts(result))

                ensemble = make_ensemble(
                    result,
                    model_keys=[
                        MODEL_NAME_LONGOUT,
                        MODEL_NAME_FULL_PP_QR,
                        # add more keys here
                    ],
                    name=MODEL_NAME,
                    # weights=[0.5, 0.5, ...]  # optional
                )

                forecasts[MODEL_NAME].extend(result[MODEL_NAME].to_QuantileForecast())
                forecasts[MODEL_NAME_LONGOUT].extend(result[MODEL_NAME_LONGOUT].to_QuantileForecast())
                forecasts[MODEL_NAME_FULL_PP_QR].extend(result[MODEL_NAME_FULL_PP_QR].to_QuantileForecast())
                forecasts[MODEL_NAME_FULL_PP_OS].extend(result[MODEL_NAME_FULL_PP_OS].to_QuantileForecast())
                forecasts[MODEL_NAME_FULL].extend(result[MODEL_NAME_FULL].to_QuantileForecast())
                forecasts[MODEL_NAME_LONGOUT_PP_QR].extend(result[MODEL_NAME_LONGOUT_PP_QR].to_QuantileForecast())
                forecasts[MODEL_NAME_LONGOUT_PP_OS].extend(result[MODEL_NAME_LONGOUT_PP_OS].to_QuantileForecast())

            return forecasts

    import logging

    class WarningFilter(logging.Filter):
        def __init__(self, text_to_filter):
            super().__init__()
            self.text_to_filter = text_to_filter

        def filter(self, record):
            return self.text_to_filter not in record.getMessage()

    gts_logger = logging.getLogger("gluonts.model.forecast")
    gts_logger.addFilter(WarningFilter("The mean prediction is not stored in the forecast data"))

    from gluonts.model.evaluation import evaluate_forecasts
    from gluonts.model import Predictor
    from gluonts.dataset.split import TestData
    from typing import Optional, Union

    def evaluate_model(
        model: Predictor,
        *,
        test_data: TestData,
        metrics,
        axis: Optional[Union[int, tuple]] = None,
        batch_size: int = 100,
        mask_invalid_label: bool = True,
        allow_nan_forecast: bool = False,
        seasonality: Optional[int] = None,
    ):

        forecasts = model.predict(test_data.input)

        fc_evals = {
            name: evaluate_forecasts(
                forecasts=fc,
                test_data=test_data,
                metrics=metrics,
                axis=axis,
                batch_size=batch_size,
                mask_invalid_label=mask_invalid_label,
                allow_nan_forecast=allow_nan_forecast,
                seasonality=seasonality,
            )
            for name, fc in forecasts.items()
        }

        return fc_evals

    import csv
    import os

    from gluonts.time_feature import get_seasonality

    from gift_eval.data import Dataset

    csv_file_paths = {}
    # Iterate over all available datasets
    for m in MODEL_LIST:
        output_dir = f"../results/{m}"
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define the path for the CSV file
        csv_file_paths[m] = os.path.join(output_dir, "all_results.csv")

        pretty_names = {
            "saugeenday": "saugeen",
            "temperature_rain_with_missing": "temperature_rain",
            "kdd_cup_2018_with_missing": "kdd_cup_2018",
            "car_parts_with_missing": "car_parts",
        }

        with open(csv_file_paths[m], "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write the header
            writer.writerow(
                [
                    "dataset",
                    "model",
                    "eval_metrics/MSE[mean]",
                    "eval_metrics/MSE[0.5]",
                    "eval_metrics/MAE[0.5]",
                    "eval_metrics/MASE[0.5]",
                    "eval_metrics/MAPE[0.5]",
                    "eval_metrics/sMAPE[0.5]",
                    "eval_metrics/MSIS",
                    "eval_metrics/RMSE[mean]",
                    "eval_metrics/NRMSE[mean]",
                    "eval_metrics/ND[0.5]",
                    "eval_metrics/mean_weighted_sum_quantile_loss",
                    "domain",
                    "num_variates",
                ]
            )

    for ds_num, ds_name in enumerate(all_datasets):
        ds_key = ds_name.split("/")[0]
        print(f"Processing dataset: {ds_name} ({ds_num + 1} of {len(all_datasets)})")
        terms = ["medium", "long"]
        for term in terms:
            if (term == "medium" or term == "long") and ds_name not in med_long_datasets.split():
                continue

            if "/" in ds_name:
                ds_key = ds_name.split("/")[0]
                ds_freq = ds_name.split("/")[1]
                ds_key = ds_key.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
            else:
                ds_key = ds_name.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
                ds_freq = dataset_properties_map[ds_key]["frequency"]
            ds_config = f"{ds_key}/{ds_freq}/{term}"

            # Initialize the dataset
            to_univariate = False if Dataset(name=ds_name, term=term, to_univariate=False).target_dim == 1 else True
            dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
            season_length = get_seasonality(dataset.freq)
            print(f"Dataset size: {len(dataset.test_data)}")

            predictor = ChronosWrapper(dataset.prediction_length, dataset.freq)
            predictor.train(dataset.validation_dataset)

            # preds = predictor.generate_calibration_forecasts(dataset.test_data.input)
            # predictor.train_postprocessor(preds)
            # Measure the time taken for evaluation
            res_all = evaluate_model(
                predictor,
                test_data=dataset.test_data,
                metrics=metrics,
                batch_size=512,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=season_length,
            )

            for model_name, csv_file_path in csv_file_paths.items():
                # Append the results to the CSV file
                res = res_all[model_name]
                with open(csv_file_path, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        [
                            ds_config,
                            model_name,
                            res["MSE[mean]"][0],
                            res["MSE[0.5]"][0],
                            res["MAE[0.5]"][0],
                            res["MASE[0.5]"][0],
                            res["MAPE[0.5]"][0],
                            res["sMAPE[0.5]"][0],
                            res["MSIS"][0],
                            res["RMSE[mean]"][0],
                            res["NRMSE[mean]"][0],
                            res["ND[0.5]"][0],
                            res["mean_weighted_sum_quantile_loss"][0],
                            dataset_properties_map[ds_key]["domain"],
                            dataset_properties_map[ds_key]["num_variates"],
                        ]
                    )

                print(f"Results for {ds_name} have been written to {csv_file_path}")
            del predictor
            del dataset
            gc.collect()  # CPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # GPU memory cleanup
                torch.cuda.ipc_collect()


if __name__ == "__main__":
    main()
