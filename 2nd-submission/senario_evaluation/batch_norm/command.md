# Batch Norm Analysis Commands

Run from this directory:

```bash
cd /home/rasheed/kuchida/antivenom_ewsn/2nd-submission/senario_evaluation/batch_norm
python3 8_cpu_core_sorted_ot_analysis.py
python3 82_cpu_core_coupling_analysis.py
```

Optional: save outputs as CSV files:

```bash
python3 8_cpu_core_sorted_ot_analysis.py --out ot_distance_summary.csv
python3 82_cpu_core_coupling_analysis.py --out ot_coupling_summary.csv
```

## Data Used

- Reference run: `../baseclean/20260403_163816.csv`
- Comparison run 1: `./clean/20260403_150737.csv`
- Comparison run 2: `./blurring/20260403_155933.csv`

## Simple Summary

- The analysis compares `baseclean` against local `clean` and local `blurring`.
- In OT distance, local `clean` is closer to `baseclean` than `blurring`.
- Mean OT distance to `baseclean`:
  - `clean`: `0.000813`
  - `blurring`: `0.001873`
- In coupling-pattern similarity, local `clean` is the template, so it scores `1.0`.
- Mean cosine similarity to the local `clean` template:
  - `clean`: `1.000000`
  - `blurring`: `0.568832`

## Interpretation

- `blurring` changes the CPU-core temporal shape more than the local `clean` run does.
- The local `clean` run stays much closer to the `baseclean` reference in both distance and coupling pattern.
