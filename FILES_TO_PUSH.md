# VeriDex Release - Files to Push Manifest

This file provides a comprehensive list of all files that have been staged in the `VeriDex_Release` directory and will be pushed to GitHub. Heavy binaries (>100MB) have been deliberately excluded and ignored via `.gitignore` to comply with GitHub restrictions.

## Root Configuration & Docs
- `README.md`: System documentation, architecture summary, and instructions.
- `.gitignore`: Git exclusion rules for `__pycache__`, `.env`, and heavy `.bin`/`.pth`/`.safetensors` files.
- `requirements.txt`: Python package dependencies.
- `FILES_TO_PUSH.md`: This manifest file.

## Web Application (`VeriDex_WebApp/`)
- `app.py`: FastAPI server handling multimodal inference logic and API routing.
- `.env.example`: Template for environment variables (Tavily API key sanitized).
- `static/index.html`: Glassmorphism frontend UI layout.
- `static/style.css`: UI stylesheets and animations.
- `static/script.js`: Frontend logic for interacting with the backend API.

## Models Configuration (`models/`)
**Note:** Model weight binaries are excluded. Only structures, lightweight heads, and tokenizers are pushed.
- **`fakeNewsModel/`**:
  - `config.json`, `special_tokens_map.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`
  - `fake_news_bert_detection.ipynb`
- **`stanceModel/`**:
  - `config.json`, `added_tokens.json`, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`
  - `spm.model`
  - `classifier_head.pt` (Lightweight head - safe to push)
  - `stance_detection_v4_full_run.ipynb`
- **`imageDetectionModel/`**:
  - `config.json`, `results.json`
  - `model_architecture.py`

## Execution & Evaluation Pipelines (`pipelines_and_evaluations/`)
- `evaluate_hybrid_pipeline.py`, `advanced_evaluation_pipeline.py`, `test_hybrid_system.py`
- `evaluate_baselines.py`, `evaluate_fake_news.py`, `evaluate_fake_news_v2.py`
- `evaluate_stance.py`, `evaluate_stance_v2.py`, `comprehensive_grid_search.py`
- `create_fake_dataset.py`, `kaggle_baseline_evaluation.py`
- `simulate_ablation_study.py`, `generate_*.py` (Visualization generation scripts)
- `run_both.py`, `run_both.ps1`, `run_veridex.bat`

## Additional Evaluation & Analytics
- **`Defense_Evaluation_Scripts/`**: Adversarial robustness scripts and metric logs (`.txt`, `.json`).
- **`Progressive_Evaluation/`**: Incremental testing scripts and datasets (`test_claims_dataset.csv`).
- **`Ablation_Visuals/`**: Output visualizations, ROC curves, radar charts, and confusion matrices.
- **`metrics/`**: JSON and CSV files containing benchmark evaluations.
- **`docs/`**: `Final_Project_Report_Draft.md`, `Proposed_Pipeline_Architecture.md`, `Sample_Test_Statements.md`, `dataset_comparison_insights.txt`.
