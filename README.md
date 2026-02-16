# MagMan-LLM: Large Language Models for VASP Convergence Prediction

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Overview

magman-llm is a research framework that leverages Large Language Models (LLMs) to predict the convergence behavior of VASP (Vienna Ab initio Simulation Package) self-consistent field (SCF) calculations. By analyzing partial OSZICAR output data from the early stages of electronic structure calculations, the system can forecast whether a calculation will ultimately converge, enabling early termination of non-convergent jobs and significant computational resource savings.

This work explores the application of LLMs to scientific computing workflows, demonstrating their capability to understand and predict complex physical processes in density functional theory (DFT) calculations without explicit training on domain-specific data.

## Key Contributions

- **LLM-based Convergence Prediction**: First application of large language models to predict VASP SCF convergence from partial OSZICAR data
- **Comprehensive Benchmark Dataset**: 907 real-world VASP calculations with ground-truth convergence labels
- **Multi-Model Evaluation**: Systematic comparison of 40+ state-of-the-art LLMs across different model families and sizes
- **Prompt Engineering Study**: Investigation of zero-shot and few-shot prompting strategies for scientific prediction tasks
- **Dynamic Workflow Integration**: Adaptive prediction system combining LLM intelligence with rule-based constraints
- **Reproducible Framework**: Open-source implementation supporting multiple LLM providers and evaluation protocols

## Methodology

### Problem Formulation

Given the first N electronic steps from a VASP OSZICAR file, predict whether the SCF calculation will converge within the maximum allowed iterations. The prediction task is formulated as a binary classification problem:

- **Input**: Partial OSZICAR data containing energy (E), energy change (dE), and charge density residual (rms) for N steps
- **Output**: Binary prediction (converge/diverge) with confidence score (0-9 scale)
- **Evaluation**: ROC-AUC, accuracy, precision, recall, and F1-score metrics

### LLM Prompting Strategies

We investigate four prompting approaches aligned with the manuscript and SI:

1. **Zero-shot**: `zero_shot.prompt.md`
2. **Few-shot_1**: `few_shot_1.prompt.md`
3. **Few-shot_2**: `few_shot_2.prompt.md` (includes the expert tie-breaker rule on long-term `rms(c)` trend)
4. **Few-shot_3**: `few_shot_3.prompt.md` (adds risk-averse instruction for workflow deployment)

### Dynamic Workflow System

For cases where initial LLM predictions have low confidence, the system implements an adaptive strategy:

1. Start with an initial step count (SI final deployment uses 100)
2. If confidence < threshold, incrementally increase steps
3. Re-query LLM with additional data
4. Continue until high confidence or maximum steps reached

This dynamic approach improves prediction reliability while minimizing computational overhead.

## Installation

### Requirements

- Python ≥ 3.11
- Dependencies: pandas, scikit-learn, matplotlib, seaborn, httpx

### Setup

```bash
# Clone the repository
git clone https://github.com/spdkit/magman-llm.git
cd magman-llm

# Install dependencies (uv-first workflow)
uv venv
uv pip install -r requirements.lock

# Or using pip directly
pip install -r requirements.lock
```

If you prefer Rye, `rye sync` remains compatible.

### API Configuration

Create a `.env` file with your LLM provider API keys:

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Supported providers:
- **ZhipuAI**: Chinese LLM provider (GLM-4 series)
- **SiliconFlow**: Multi-model aggregator (DeepSeek, Qwen, etc.)
- **OpenRouter**: International multi-model platform
- **DeepSeek**: Direct DeepSeek API access
- **LiteLLM**: Unified interface for 100+ providers

## Quick Start

### Basic Evaluation

```bash
# Evaluate a model on the benchmark dataset
python scripts/evaluate_llm_convergence.py \
    --provider zhipuai \
    --model glm-4.5-air \
    --dataset original_dataset/benchmark_dataset.csv \
    --steps 50 \
    --max-samples 100 \
    --output results.csv
```

### Prompt Template Usage

```bash
# Use Few-shot_2 prompt (best performance)
python scripts/evaluate_llm_convergence.py \
    --provider zhipuai \
    --model glm-4.5-air \
    --prompt-template prompt_templates/few_shot_2.prompt.md \
    --dataset original_dataset/benchmark_dataset.csv \
    --steps 50 \
    --output results.csv
```

### Results Analysis

```bash
# Generate ROC curves, confusion matrices, and performance metrics
python scripts/analyze_results.py results.csv
```

Output files:
- `roc_curve.png` - ROC curve with AUC score
- `confusion_matrix.png` - Classification confusion matrix
- `pr_curve.png` - Precision-recall curve
- `confidence_distribution.png` - Confidence score distribution
- `analysis_report.json` - Detailed performance metrics

### Dynamic Workflow Evaluation

```bash
# Run adaptive prediction with confidence-based step increments
# SI final deployment: FS-3 + start monitoring at 100 steps
python scripts/integrated_workflow.py \
    --provider zhipuai \
    --model glm-4.5-air \
    --prompt-template prompt_templates/few_shot_3.prompt.md \
    --dataset original_dataset/benchmark_dataset.csv \
    --initial-steps 100 \
    --step-increments "100,125,150,175,200" \
    --confidence-cutoff 7 \
    --output workflow_results.csv
```

## Benchmark Dataset

The benchmark dataset comprises 907 real-world VASP single-point energy calculations:

- **Total samples**: 907 calculations
- **Converged**: 681 (75.1%)
- **Non-converged**: 226 (24.9%)
- **Step range**: Up to 200 electronic iterations in this study
- **Systems**: Various magnetic materials and compounds

Dataset location: `original_dataset/benchmark_dataset.csv`

Each sample includes:
- OSZICAR file path
- Ground-truth convergence label
- Number of electronic steps
- System metadata

## Advanced Features

### Multi-Model ROC Comparison

```bash
# Compare multiple models on the same dataset
python scripts/multi_roc_of_models.py \
    --results-dir results_analysis/02_main/
```

### Multi-Prompt Comparison

```bash
# Compare different prompt strategies
python scripts/multi_roc_of_prompts.py \
    --results-dir results_analysis/02_main/
```

### Model Performance Visualization

```bash
# Generate violin plot showing performance distributions
python scripts/violin_of_models.py \
    --results-dir results_analysis/02_main/

# Generate Cleveland dot plot for model ranking
python scripts/cleveland_dot_plot_of_models.py \
    --results-dir results_analysis/02_main/
```

### Rule-Based Baseline

```bash
# Evaluate physics-based rule system for comparison
python scripts/rule_based_evaluator.py \
    --dataset original_dataset/benchmark_dataset.csv \
    --steps 50 \
    --output rule_results.csv
```

### Parallel Evaluation

For large-scale experiments, split the dataset and run parallel evaluations:

```bash
# 1. Split dataset into N parts
python scripts/split_dataset.py --steps 100 --parts 4

# 2. Run evaluations in parallel (separate terminals)
python scripts/evaluate_llm_convergence.py --dataset dataset_steps100_part0.csv ...
python scripts/evaluate_llm_convergence.py --dataset dataset_steps100_part1.csv ...
# ... (continue for all parts)

# 3. Merge results
python scripts/merge_results.py --steps 100
```

## Reproducibility

All experiments can be reproduced at the protocol level using the provided scripts and dataset:

1. **Model Evaluation**: Run `evaluate_llm_convergence.py` with fixed dataset, prompt version, and model ID
2. **Prompt Comparison**: Compare templates in `prompt_templates/`
3. **Workflow Testing**: Run `integrated_workflow.py` with explicit checkpoints and confidence threshold

Important note on LLM reproducibility:
- API-served LLMs are probabilistic and provider backends can change over time.
- This repository targets protocol-level reproducibility (same data split, prompts, workflow settings, and analysis scripts), rather than bitwise-identical outputs.

## Citation

If you use this work in your research, please cite the manuscript and this repository.

Manuscript title:
"Training-free monitoring of SCF convergence using large language models: A case study of magnetic iron oxides with VASP"

## License

This project is licensed under the GNU General Public License v3.0 (GPL-3.0-or-later). See [LICENSE](LICENSE) for details.
