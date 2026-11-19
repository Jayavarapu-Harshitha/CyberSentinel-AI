# CyberSentinel AI — FSE 560

Reproducible binary intrusion detection on CIC-IDS2017, with an independent CPU MLP and validation-selected probability-fusion aggregator.

## Windows ARM64 setup (VS Code / PowerShell)

Use **native ARM64 Python 3.12**, if available. Check `python -c "import platform; print(platform.machine())"` (expected `ARM64` for native Python). Windows ARM64 wheel availability varies by package/version; installation must be tested on your machine. A Python x64 environment under Windows emulation is a fallback if a dependency lacks ARM64 support. The project deliberately does not require PyTorch, XGBoost, or LightGBM for its core workflow.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test,app]"
python -m pytest -q
```

If `py -3.12` is unavailable, use an installed supported Python 3.11–3.13 (`python -m venv .venv`) and check architecture. Optional tree libraries: `python -m pip install -e ".[trees]"` only if compatible wheels exist; if unavailable, training skips them and uses HistGradientBoosting as a CPU-safe boosted-tree comparison.

## Dataset

Place the original `MachineLearningCSV.zip` at `data/MachineLearningCSV.zip`. It is intentionally excluded from Git. The original ZIP is not modified.

## Run

```powershell
python scripts/audit.py --zip data/MachineLearningCSV.zip
python scripts/train.py --models mlp
python scripts/train.py
python -m streamlit run app.py
```

Run individual models by changing `--models`: `logistic_regression`, `random_forest`, `mlp`, `hist_gradient_boosting`, `xgboost`, `lightgbm`.

The audit streams through the whole ZIP, reports original class frequencies, and keeps at most 30,000 rows per binary class in a reproducible uniform reservoir sample. It removes exact feature-and-label duplicates before splitting. It does **not** assert that near-duplicates or capture-day leakage are eliminated. The sample changes class prevalence: precision and PR-AUC are **not** representative of deployment prevalence. Report this limitation.

All numeric imputers and scalers fit only on training data. The MLP uses 128–64–32 hidden units with ReLU/Adam, early stopping and internal training-only validation. Models share the same split; tree selection and fusion weights use validation only; final test is reserved for evaluation.

Outputs: `artifacts/audit.json`, `train.pkl`, `val.pkl`, `test.pkl`, trained `.joblib` models, `validation.json`, `comparison.csv`, `training_summary.json`. These are ignored by Git. Do not load untrusted `.joblib` files.

## Known limitations / future work

- First run is a **sampled-data baseline**, not full CIC-IDS2017 evaluation.
- Random stratified split does not prove generalization to unseen days or attack families.
- Exact deduplication alone cannot eliminate all correlated flows.
- Threshold 0.5 is fixed for initial comparison; calibrate and choose thresholds on validation only in follow-up experiments.
- Fusion latency is currently a single measured call, not a repeated benchmark; do not compare its latency directly with individual model medians.
- SHAP, Isolation Forest, file-based holdout, and initial ablations are implemented. Learning curves, full-dataset training, external generalization and deployment remain future work.
- The dashboard is a CSV inference prototype, not a production security monitoring system.

## Submission

Keep the full modular repository on GitHub. The professor's ZIP can contain `standalone_model.py`, `ai_aggregator.py`, the PDF, and supporting source/package files so the entry points remain runnable. Confirm the final packaging format with the instructor if strict two-file-only interpretation is used.

## Complete implementation commands (v0.2)

**You do not need Jupyter to run this project.** Install VS Code's **Python** extension; the **Jupyter** extension is optional for exploring notebooks. All training, evaluation, and UI components are ordinary `.py` programs.

```powershell
# From the project root, after environment activation and installation:
python scripts/audit.py --zip data/MachineLearningCSV.zip
python scripts/train.py --models logistic_regression
python scripts/train.py --models random_forest
python scripts/train.py --models xgboost
python scripts/train.py --models lightgbm
python scripts/train.py --models mlp
# IMPORTANT: The above separate commands overwrite comparison/summary outputs.
# Train ALL available models together for valid fusion and consolidated comparison:
python scripts/train.py
python scripts/advanced.py anomaly
python scripts/advanced.py explain
python scripts/advanced.py ablation
python scripts/advanced.py latency
python scripts/day_holdout.py --zip data/MachineLearningCSV.zip
python -m streamlit run app.py
# In another terminal (optional local API):
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

**All five proposal classifiers are implemented in separate files:** `logistic_regression.py`, `random_forest.py`, `xgboost.py`, `lightgbm.py`, and `mlp.py`. The sixth `hist_gradient_boosting.py` is a scikit-learn fallback, **not a substitute for reporting the required XGBoost/LightGBM experiments**. Native Windows ARM64 wheel compatibility for the optional libraries must be confirmed on your device. To attempt the two optional libraries: `python -m pip install -e ".[trees]"`. If either fails, do not claim it was evaluated; use a separate x64 Python environment under Windows emulation or a supported training machine. Check `python -c "import platform; print(platform.machine())"`.

**Additional components:** `scripts/advanced.py anomaly` fits Isolation Forest on training benign traffic; `explain` computes validation-set permutation importance; `ablation` compares standalone and fusion variants and a half-feature MLP; `latency` repeats timings for all trained models including the fusion; `scripts/day_holdout.py` evaluates an unseen capture file. `cybersentinel.explain.shap_explain` provides optional SHAP on supported tree models (install `shap` separately); SHAP is not part of the default execution because of dependency and performance costs.

**Cautions:** The default sampler balances binary classes and can underrepresent attack subtypes. Day holdout uses the sampled data and cannot guarantee detection of novel attack families. The first experimental comparison does not measure realistic live traffic prevalence. Heuristic priority scores are uncalibrated. Do not publish raw data, `.joblib` model artifacts from untrusted sources, or expose the development API publicly without authentication, limits, and validation. Final paper claims must use actual completed experiment outputs.

## GitHub

```powershell
git init
git add .
git commit -m "Initial reproducible CyberSentinel AI implementation"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

Create the empty repository first and replace the example URL. Verify `.gitignore` excludes `data/*.zip`, `artifacts/`, `.venv/`, and local credentials before pushing. Do not put raw CIC-IDS2017 ZIP into Git.

Optional SHAP export after installing a compatible `shap` wheel and training a supported tree model:

```powershell
python -m pip install shap
python scripts/shap_explain.py --model random_forest --rows 20
```

SHAP exports are signed feature contributions and are **not** causal explanations. The file uses the feature order of the trained model.

---

# Installation and Execution Guide

## System Requirements

- Python 3.12 recommended (project supports Python 3.11–3.13).
- Windows 10/11 x64, Windows 11 ARM64 with Ubuntu WSL2, or Linux.
- Internet connection for initial dependency installation.
- Saved model artifacts for inference.
- CIC-IDS2017 `MachineLearningCSV.zip` only when retraining or reproducing dataset-dependent evaluations.

**Important:** The included saved models were produced in Ubuntu WSL on ARM64. Loading them on another operating system or with different dependency versions is not guaranteed. For the most reproducible experience, use Ubuntu WSL2 with matching dependencies.

## Option A — Windows 11 ARM64 (Recommended: Ubuntu WSL2)

This is the environment used for project development and testing.

### A1. Install WSL2

Open PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu
```

Restart Windows if prompted. Open Ubuntu and complete its initial setup.

### A2. Open the project

Extract the project ZIP into your Ubuntu home directory. Open an Ubuntu terminal and navigate to the extracted folder:

```bash
cd ~/projects/cybersentinel-ai
```

Adjust this path to match your extraction location. If the ZIP extracts to a folder with a different name, use that folder name instead.

### A3. Create the environment

Install `uv` if it is not already available, following the official instructions at https://docs.astral.sh/uv/.

Then run:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

If the local package is not installed, also run:

```bash
uv pip install -e .
```

### A4. Launch the dashboard

```bash
export OPENBLAS_CORETYPE=ARMV8
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open http://localhost:8501 in your Windows browser.

### A5. Launch the API (optional)

Open a second Ubuntu terminal:

```bash
cd ~/projects/cybersentinel-ai
source .venv/bin/activate

export OPENBLAS_CORETYPE=ARMV8
export OPENBLAS_NUM_THREADS=1

python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Visit http://localhost:8000/health.

## Option B — Normal Windows 10/11 (Intel or AMD x64)

Open PowerShell in the extracted project folder.

### B1. Create a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### B2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The frozen requirements were generated on Linux ARM64 and may contain platform-specific packages or versions. If installation fails, use the project's declared dependencies instead:

```powershell
python -m pip install -e ".[app,trees,test]"
python -m pip install shap matplotlib
```

Package compatibility must be verified on the target computer.

### B3. Start the dashboard

```powershell
python -m streamlit run app.py
```

Open http://localhost:8501.

### B4. Start the API (optional)

In a second PowerShell terminal, from the project folder:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

### B5. If saved models cannot be loaded

The serialized models may be incompatible with Windows or different package versions.

To reproduce training, obtain the original CIC-IDS2017 dataset and place it at:

`data/MachineLearningCSV.zip`

Then run:

```powershell
python scripts/audit.py --zip data/MachineLearningCSV.zip
python scripts/train.py
```

Retraining may overwrite model and evaluation artifacts. Use a separate copy of the project to preserve the original experimental results.

## Option C — Native Ubuntu Linux

Use Python 3.12 and the same environment installation instructions as Option A.

For ARM64 Linux, apply the ARM-specific environment variables shown in Option A. On x64 Linux, these ARM-specific variables are generally unnecessary.

## Running the Standalone Model and AI Aggregator

The two assignment entry points are:

- `standalone_model.py`: trains the standalone MLP model.
- `ai_aggregator.py`: trains the model collection and evaluates the fusion aggregator.

These are training entry points, not the interactive dashboard. They require the prepared training data and may overwrite saved artifacts.

To reproduce training:

```bash
python scripts/audit.py --zip data/MachineLearningCSV.zip
python standalone_model.py
python ai_aggregator.py
```

To demonstrate predictions using the supplied trained artifacts, launch `app.py` instead. Retraining is not necessary for the dashboard demonstration.

## Testing the Application

1. Launch the Streamlit dashboard.
2. Download or select `sample_upload.csv`.
3. Upload the sample file.
4. Review classification results, model comparisons, and anomaly indicators.
5. Export the prediction results.
6. Optionally start FastAPI and check the `/health` endpoint.

To execute the automated tests:

```bash
python -m pytest -q
```

## Important Limitations

- The system expects CIC-IDS2017-compatible extracted network-flow CSV features, not raw PCAP files.
- The application is a research prototype, not a production intrusion prevention or monitoring system.
- The supplied dataset is excluded from the submission archive.
- Inference results depend on compatible saved model artifacts and package versions.
- Performance measurements reported in the paper are experimental results and should not be treated as production guarantees.
- Only load model artifacts from trusted sources.


