# Methodology Summary

This directory contains the analysis pipeline used to turn raw telemetry logs into OT/cosine summary CSVs and then into figures, tests, and simple classifiers.

## Core Pipeline

The main pipeline is:

1. `log_loader.py`
   Loads raw CSV logs from a log directory such as `log/`, `log_bg/`, `log_sl/`, `log_finetune/`.
   It adds metadata columns such as `device_id`, `poisoning_type`, and `poison_frac`.

2. `1_cpu_core_sorted_mem_ot_analysis.py`
   Computes per-device and per-round Wasserstein / OT distance features.
   It uses sorted per-core CPU usage and memory usage shapes.
   Main outputs:
   - `mem_ot_distance_to_clean_005`
   - `core0_ot_distance_to_clean_005`
   - `core1_ot_distance_to_clean_005`
   - `core2_ot_distance_to_clean_005`
   - `core3_ot_distance_to_clean_005`
   - `core_ot_distance_mean`

3. `2_cpu_core_sorted_mem_ot_analysis.py`
   Reuses the same shape extraction logic, then computes cosine similarity features based on OT-coupling deltas relative to the clean `0.10` template.
   Main outputs:
   - `mem_type_cosine_to_clean010`
   - `core0_type_cosine_to_clean010`
   - `core1_type_cosine_to_clean010`
   - `core2_type_cosine_to_clean010`
   - `core3_type_cosine_to_clean010`
   - `core_type_cosine_mean_to_clean010`

4. `3_cpu_core_sorted_mem_ot_cosine_export.py`
   Merges the outputs of `1_...` and `2_...` into one summary CSV.
   It also renames:
   - `poison_frac` -> `poisoning_rate`
   - `epoch` -> `round`

This is the script most likely used to generate files such as:
- `fl_main.csv`
- `sl_main.csv`
- `bg_main.csv`
- `bg2_main.csv`
- `finetune_main.csv`

Typical command pattern:

```bash
cd methodology
python 3_cpu_core_sorted_mem_ot_cosine_export.py --log-dir log --out fl_main.csv
```

Other datasets follow the same pattern with a different `--log-dir` and output filename.

## Main Summary CSVs

- `fl_main.csv`
  Federated-learning main summary CSV. This is the default input for many figure and analysis scripts.

- `sl_main.csv`
  Smaller / alternate dataset summary with the same schema as `fl_main.csv`.

- `bg_main.csv`, `bg2_main.csv`
  Background-related datasets using the same summary schema.

- `finetune_main.csv`
  Fine-tuning dataset summary using the same summary schema.

Expected shared columns include:
- `device_id`
- `poisoning_type`
- `poisoning_rate`
- `round`
- OT-distance columns from `1_...`
- cosine-similarity columns from `2_...`

## Figure Scripts

These scripts usually read one summary CSV, derive:
- `ot_distance = 0.8 * core_ot_distance_mean + 0.2 * mem_ot_distance_to_clean_005`
- `cosine_similarity = 0.8 * core_type_cosine_mean_to_clean010 + 0.2 * mem_type_cosine_to_clean010`

and then write figures into `4_figures/`.

### Per-device trend / summary plots

- `4_plot_device_ot_cosine_std.py`
  Plots mean plus std by poisoning rate.

- `41_plot_device_ot_cosine_mean.py`
  Plots mean OT distance and cosine similarity by poisoning rate.

- `42_plot_device_ot_cosine_rounds.py`
  Plots per-round behavior across poisoning rates.

- `43_plot_device_ot_cosine_rounds_no_outliers.py`
  Same as `42_` but removes robust outliers first.

- `44_plot_device_ot_cosine_std_no_outliers.py`
  Std-style plot after outlier filtering.

- `45_plot_device_ot_cosine_mean_no_outliers.py`
  Mean-style plot after outlier filtering.

### Box plots

- `46_plot_device_ot_cosine_box.py`
  Per-device grouped box plots for OT distance and cosine similarity.
  Default input: `fl_main.csv`

- `47_plot_device_ot_cosine_box_clean0.py`
  Similar to `46_`, but it shifts `clean` runs to poisoning rate `0.0` for display.
  This is often more readable when clean should appear as the leftmost reference group.

- `48_plot_blurring30_vs_clean_devices.py`
  Compares `clean` vs `blurring` at `0.30` across devices.

- `49_plot_device120_box.py`
  Focused box plot for device `120`.

- `410_compare_devices_distance_only.py`
  OT-distance-only comparison across selected devices.

- `411_plot_clean_vs_poison30_agg.py`
  Aggregates poison types at `0.30` and compares them with clean.

## Statistical Tests and Classification

- `5_mannwhitney_device_clean_vs_poison.py`
  Mann-Whitney tests for clean vs poison comparisons.

- `51_trend_test_poisoning_rate.py`
  Trend analysis over poisoning rate.

- `52_classify_clean_poison_f1.py`
  Basic clean vs poison classification using OT/cosine thresholds.

- `53_classify_clean_poison_f1_window.py`
  Windowed classification by grouping multiple rounds.

- `54_classify_clean_poison_f1_window_sweep.py`
  Sweeps window sizes / thresholds and can export CSV results.

- `55_plot_window_sweep_avg.py`
  Plots the saved sweep results.

- `56_mannwhitney_bg_bg2_clean_vs_poison30.py`
  Statistical comparison for background-oriented datasets.

## Synthetic Data Scripts

- `A_generate_synthetic_blurring30.py`
  Builds synthetic blurring-30 data from an existing summary CSV.
  Default output: `fl_synthetic_blur.csv`

- `B_generate_synthetic_steganography.py`
  Creates steganography variants from base summary CSVs.
  Default outputs:
  - `fl_stego.csv`
  - `sl_stego.csv`

- `C_generate_finetune_synthetic_112.py`
  Creates a synthetic fine-tune dataset variant.

## Coverage and Scatter

- `0_dataset_coverage_heatmaps.py`
  Dataset coverage overview across CSVs.

- `6_scatter_ot_vs_similarity.py`
  Scatter plot of OT distance vs cosine similarity.

## Practical Notes

- Many plotting scripts default to `fl_main.csv`. If run without arguments, that is usually the CSV being used.
- The current checked-in scripts may be newer than some CSV files already stored in this directory. In practice, the schema still shows that `fl_main.csv` and similar files come from the `1_ -> 2_ -> 3_` export pipeline.
- Existing files inside `4_figures/` can be a mix of outputs from different CSVs and different runs. Do not assume that the whole directory was generated from one input file.
