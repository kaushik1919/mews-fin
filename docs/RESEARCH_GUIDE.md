# MEWS Research Enhancements

This guide summarizes the methodological additions introduced to position the Market Risk Early Warning System (MEWS) for academic research.

## Methodological Contributions

- **Regime-Adaptive Ensemble**: Learns probability weights per volatility regime (low/moderate/high) using validation AUC scores. Predictions optionally adapt to the detected regime at inference time.
- **Cross-Attention Fusion**: Optional transformer-driven attention module that fuses textual sentiment embeddings with tabular indicators, delivering higher-order interactions between modalities.
- **Extended Feature Fusions**: Multimodal pipeline can switch between concatenation and attention-based fusion without altering upstream preprocessing.

## Evaluation Framework

- **Baselines**: Utilities for GARCH/Value-at-Risk and LSTM sequence models enable benchmarking against classical and deep-learning references.
- **Crisis Windows**: Built-in datasets cover the Global Financial Crisis, COVID-19 shock, and recent Fed tightening period, with precision@K, AUC, Brier score, and calibration diagnostics.
- **Significance Testing**: Likelihood-ratio tests and paired comparisons quantify statistical improvements over baselines.

## Hypothesis & Ablation Testing

- **Sentiment Impact**: Hypothesis testing module compares fundamentals-only and sentiment-augmented models, reporting log-likelihood, LRT statistic, and $p$-values.
- **Graph Features**: Automated ablation study isolates the value of correlation-driven graph metrics.

## Ethics & Robustness

- **Bias Detection**: Kolmogorov-Smirnov based checks highlight sentiment skew across publishers or sectors.
- **Stress Tests**: Noise injection and delayed-news scenarios evaluate adversarial resilience of trained models.

## Research Reporting

- **Markdown & HTML Outputs**: `ResearchReportBuilder` synthesizes evaluation, hypothesis, and robustness results into citable artifacts under `outputs/research/`.
- **Pipeline Stage**: The core orchestrator runs a dedicated "Research Addendum" stage at the end of the full pipeline to generate artifacts automatically.

## Usage

1. Run the full pipeline:
   ```bash
   python main.py --full-pipeline
   ```
2. Review research artifacts:
   - Markdown: `outputs/research/research_overview.md`
   - HTML: `outputs/research/research_overview.html`

3. Iterate with advanced fusion by selecting the cross-attention strategy when constructing `MultiModalFeatureFusion`.

Refer to `src/research/` for detailed implementations and extendable hooks.
