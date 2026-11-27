# PainNAS

PainNAS searches convolutional neural networks for supervised pain assessment
from BioVid Part A physiological signals. It supports subject-disjoint neural
architecture search (NAS), leave-one-subject-out (LOSO) evaluation, and
resumable experiments. The default task distinguishes baseline (`T0`) from
high pain (`T4`) using GSR/EDA, ECG, and EMG.

Inputs have shape `[samples, 1152, 3]` and are rearranged for the model to
`[samples, 3, 1152, 1]`. Early fusion combines the signals at the input; late
fusion trains separate modality branches and learns normalized prediction
weights. Optuna searches architecture and training choices, starting with a
reference CNN.

Paper: [Efficient Architecture Search under Leave-One-Subject-Out Evaluation](https://arxiv.org/pdf/2609.21457).

## Citation

```bibtex
@article{hihn2026efficient,
  title={Efficient Architecture Search under Leave-One-Subject-Out Evaluation},
  author={Hihn, Heinke and Schwenker, Friedhelm},
  journal={arXiv preprint arXiv:2609.21457},
  year={2026}
}
```

## Requirements and setup

- Python with TensorFlow 2.20, NumPy, pandas, scikit-learn, and Optuna. The
  provided `requirements-colab.txt` pins TensorFlow; the other packages are
  unpinned. Install Matplotlib separately to generate result plots.
- A TensorFlow-compatible GPU is expected for full experiments. `--allow-cpu`
  permits CPU runs for debugging.
- Access to the preprocessed BioVid Part A dataset with its predefined Train and
  Test splits. The default loader expects 87 subjects, 1,152 time points per
  sample, and the three modalities above. Data is not included here.
- The `data_loaders` package from
  [FewShotPainAdaptation](https://github.com/hhihn/FewShotPainAdaptation),
  including `pain_ds_config` and `pain_meta_dataset`. It is **not included** in
  this repository or installed by `requirements-colab.txt`. The Colab staging
  step also requires its `dataset_staging` module. Use a compatible checkout and
  follow its data preparation instructions.

The source imports this repository as `painnas`, although this checkout has no
package installation metadata. One way to run locally is to put this checkout
in a directory named `painnas` and put both its parent and the directory
containing `data_loaders` on `PYTHONPATH`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-colab.txt matplotlib
export PYTHONPATH="/path/to/parent-of-painnas:/path/to/FewShotPainAdaptation:$PYTHONPATH"
python -m painnas --help
```

The TensorFlow pin targets the Colab environment; local GPU support depends on
the operating system and TensorFlow installation. Point `--data-dir` to the
dataset root expected by the external loader.

## Evaluation protocols

**Cross-fitted block NAS** (`cross-fitted`) partitions subjects into five
deterministic outer blocks by default. For each block, NAS excludes the entire
block and evaluates each remaining subject in one of three inner subject folds.
The default selection score is mean subject accuracy minus its standard error
(`--uncertainty-beta` controls the penalty). The strongest inner-fold
checkpoint of the winning trial initializes each target's model. A fresh
optimizer continues training on the other subjects, using source-subject Test
samples for early stopping; evaluation uses only the target's Test samples.
The default continuation budget is the rounded median inner best epoch.

**Nested LOSO** (`nested`) runs a separate target-exclusive search for each
subject, then trains a freshly initialized model on source subjects and tests
on the held-out target. This is more computationally expensive.

**Global search and LOSO** (`search`, `loso`, or `all`) selects an architecture
once using a subject-disjoint search split, then retrains it from scratch for
each target. The search uses all subject identities, so its subsequent LOSO
summary is exploratory and is not an unbiased nested-LOSO estimate. The search
uses 55 training, 14 validation, and 18 architecture-selection test subjects
with the default cohort and seed.

## Run

From an environment configured as above, start or resume cross-fitted NAS:

```bash
python -m painnas cross-fitted \
  --data-dir /path/to/BioVid-data \
  --output-dir /path/to/runs/run_001 \
  --resume
```

Add `--fusion-mode late` for late fusion. Cross-fitted and nested CLI commands
default to 10 trials and 20 search epochs; global search defaults to 50 trials
and 50 search epochs. Use `--n-trials`, `--search-max-epochs`, and
`--cross-fitted-continuation-epochs` to change the budget. Run
`python -m painnas cross-fitted --help` for all options. Configuration values
not exposed by the CLI, including `architecture_selection_mode`, can be set on
`PainNASConfig` when using the Python API or notebook.

Other workflows:

```bash
python -m painnas nested --data-dir DATA --output-dir RUN --resume
python -m painnas search --data-dir DATA --output-dir RUN --resume
python -m painnas loso --data-dir DATA --output-dir RUN --resume
# Alternatively, run search and LOSO together:
python -m painnas all --data-dir DATA --output-dir RUN --resume
```

`loso` reads `RUN/search/best_architecture.json` by default; use
`--architecture-json` to specify another selection. Fold ranges are one-based
and inclusive, allowing long jobs to be split across sessions:

```bash
python -m painnas cross-fitted --data-dir DATA --output-dir RUN --resume \
  --loso-start-index 1 --loso-stop-index 30
```

For a small smoke run on CPU, use `--allow-cpu --n-trials 1
--search-max-epochs 1 --outer-block-count 2 --inner-fold-count 2 --max-folds 1
--bootstrap-samples 100`. The real dataset and external loader are still
required. Resume checks configuration and architecture fingerprints; use a new
output directory when changing settings.

## Colab

[colab_entrypoint.ipynb](colab_entrypoint.ipynb) clones the
FewShotPainAdaptation repository and installs its `painnas` requirements, stages
BioVid from Google Drive, and runs the configured workflow. It currently
defaults to `cross_fitted_weight_sharing`, late fusion, and a reference run:
set `REFERENCE_RUN_DIR` to an existing cross-fitted run before using that mode.
Alternatively set `WORKFLOW_MODE` to `cross_fitted` or `global_search_loso`.
Review the repository branch, data paths, run name, and configuration cells
before running: the notebook uses the `painnas` code in its cloned checkout,
which may differ from this repository. The weight-sharing mode screens
candidates from a previous cross-fitted run before independently refitting a
shortlist.

[best_architecture_full_loso_colab.ipynb](best_architecture_full_loso_colab.ipynb)
provides a separate fixed-architecture LOSO workflow and also uses the external
checkout.

## Results

Cross-fitted runs write the following under `RUN/cross_fitted_loso/`:

- `blocks/block_NNN/search/study.sqlite3`: resumable Optuna study.
- `blocks/block_NNN/search/best_architecture.json` and
  `best_warmstart.weights.h5`: selected architecture and checkpoint.
- `folds/fold_NNN/result.json`: target-fold audit, metrics, and predictions.
- `fold_metrics.csv`, `predictions.csv`, `architecture_frequencies.csv`, and
  `summary.json`: aggregate results, including bootstrap confidence intervals.

Generate a PNG and optional PDF overview with:

```bash
python -m painnas.visualize_cross_fitted_results \
  --run-dir RUN/cross_fitted_loso --pdf
```

The plot distinguishes inner subject-macro validation accuracy used for
architecture selection from held-out LOSO test metrics.
