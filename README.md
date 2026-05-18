# spectral_analysis

Spectral preanalysis scripts for the apple-bruise NIR HSI segmentation project
(companion repo to `msi`).  Each script in `scripts/preanalysis/` is an
independently runnable analysis that writes its outputs to
`outputs/preanalysis/<script_name>/`.

## Layout

```
spectral_analysis/
├── scripts/preanalysis/
│   ├── _common.py                          # shared utilities
│   ├── _adapters.py                        # ONLY place that imports from msi
│   ├── spectral_curves_with_separability.py  # A1
│   ├── bootstrap_band_stability.py           # A2
│   ├── error_distance_to_boundary.py         # A4
│   ├── single_band_ablation.py               # A5
│   ├── filter_tolerance_analysis.py          # A3
│   └── band_selection_spa_cars_mi.py         # SPA / CARS / MI feature selection
├── run_preanalysis.sh                      # driver
└── requirements.txt
```

## Expected layout on the GPU server

Clone this repo as a sibling of `msi`:
```
~/projects/
├── msi/
│   ├── data/dataset.py
│   ├── utils/metrics.py
│   ├── scripts/band_range_search.py
│   ├── configs/baseline.yaml
│   └── outputs/baseline_seed42/checkpoints/best_model.pth
└── spectral_analysis/                  <-- this repo
    ├── scripts/preanalysis/
    └── ...
```

`_adapters.py` injects `../msi` onto `sys.path`; override with `--msi-root`
on any script that needs it (A3/A4/A5).

## Dataset path

Default: `/root/autodl-tmp/datasets/full_HSI_dataset`, with subdirectories
`images/`, `masks/` (defect), `whole/` (apple region).  Override with
`--data-root`.  The same filename (e.g. `REFLECTANCE_2026-02-11_009.npy`)
must appear in all three subdirectories.

## Quick start

```bash
# install python deps (torch already installed in msi env)
pip install -r requirements.txt

# A1 (~5-15 min, no training)
python scripts/preanalysis/spectral_curves_with_separability.py

# A2 (a few seconds once A1 has run)
python scripts/preanalysis/bootstrap_band_stability.py

# A4 (10-30 min, needs trained checkpoint)
python scripts/preanalysis/error_distance_to_boundary.py \
    --checkpoint ../msi/outputs/baseline_seed42/checkpoints/best_model.pth

# A5 (~10 h on a single GPU, 51 bands x 80 epochs)
python scripts/preanalysis/single_band_ablation.py

# A3 (up to 8 h, 125 combos x 80 epochs)
python scripts/preanalysis/filter_tolerance_analysis.py --base-combo 77 82 90

# all at once (A0 is a no-op)
./run_preanalysis.sh --task all

# SPA / CARS / MI feature band selection (~1-5 min on a CPU)
# Only needs images/ and masks/ -- no whole/ required.
python scripts/preanalysis/band_selection_spa_cars_mi.py \
    --images-dir "C:/Users/10730/Desktop/hsi(20-110)/images" \
    --masks-dir  "C:/Users/10730/Desktop/hsi(20-110)/masks" \
    --n-select 10
```

## Notes on the design

* **A0 (per-apple grouped split) is dropped.**  The current dataset filenames
  (`REFLECTANCE_<date>_<seq>.npy`) do not encode apple identity and no
  mapping was provided.  When a mapping becomes available, re-introduce A0 by
  adding `--apple-map-csv` to `_common.list_stems` and using
  `sklearn.model_selection.GroupKFold`.
* **A2 (bootstrap stability) is image-level**, not per-apple, for the same
  reason.  The report explicitly notes this is an upper bound on stability.
* The combination-level bootstrap was downgraded to single-band Fisher rank
  bootstrap, per the spec's downgrade clause.
* `_adapters.py` is the **only** module that imports from `msi`.  If
  `train_and_eval`, `MSIDataset`, or the model class has a different
  signature in your project, edit `_adapters.py` -- nothing else.

## Outputs

Each script writes:
* a JSON results file
* one or more PNG figures (dpi=200)
* a `*_report.txt` plain-text summary
* a `<script_name>.log` containing the run trace

`run_preanalysis.sh` appends per-task rows to
`outputs/preanalysis/summary.md`.

## Resume

A3 and A5 train many independent combos and flush results to disk after
every combo.  Re-invoking the script picks up where it left off; pass
`--no-resume` to start fresh.
