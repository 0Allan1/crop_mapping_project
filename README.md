# Crop-Type Classification from Fused Optical-Radar Data

Traditional machine learning (scikit-learn) versus deep learning (TensorFlow/Keras) for classifying seven crop types, evaluated with both a random split and a spatially separated block split.


[Open in Colab](https://colab.research.google.com/github/YOUR-USERNAME/YOUR-REPO/blob/main/crop_mapping_project.ipynb)

## Problem

Crop-type maps support yield forecasting and food-security planning. This project asks how well standard classifiers identify crop types from fused optical and radar satellite features, and how much the evaluation protocol changes the answer. Two difficulties drive the design: strong class imbalance (the rarest class is 0.35% of the data) and spatial dependence between neighbouring pixels, which makes random validation over-optimistic.

## Dataset

UCI Machine Learning Repository, *Crop mapping using fused optical-radar data set* ([link](https://archive.ics.uci.edu/dataset/525/crop+mapping+using+fused+optical+radar+data+set)), licence CC BY 4.0.

- 325,834 samples, 174 continuous features, 7 classes (corn, peas, canola, soybeans, oats, wheat, broadleaf).
- RapidEye optical and UAVSAR radar imagery of farmland near Winnipeg, Canada, from 5 and 14 July 2012.
- **The raw data is not stored in this repository.** The notebook downloads it from UCI automatically. A zipped copy is in the Google Drive folder.

## Approach

- **Splits:** a stratified random 70/15/15 split (reference) and a within-class block split with 1% buffers (main protocol).
- **Preprocessing:** signed log transform and standardisation (fitted on training data only) for logistic regression and neural networks; class weights for imbalance.
- **Feature engineering:** date-difference features (+87) and a correlation filter (174 to 102 features).
- **Traditional models:** logistic regression and histogram gradient boosting, tuned with a random search.
- **Deep learning:** Sequential MLP variants and a two-branch Functional API model (separate radar and optical branches), using a `tf.data` pipeline, class weights, dropout, L2 and early stopping on validation loss or validation macro-F1.
- **Evaluation:** macro-F1 (main metric), accuracy, macro precision and recall, one-vs-rest ROC-AUC, per-class reports, confusion matrices, learning curves and class-level precision-recall.

## Key results (held-out test block)

| Model | Accuracy | Macro-F1 | ROC-AUC | Broadleaf precision |
|---|---|---|---|---|
| Tuned boosting | 0.9709 | 0.9127 | 0.9981 | 0.438 |
| MLP (macro-F1 stopping) | 0.9712 | 0.9231 | 0.9987 | 0.462 |
| Two-branch Functional model | 0.9722 | 0.9248 | 0.9982 | 0.469 |

- A random split overstated macro-F1 by 1.7 to 2.2 points compared with the block split.
- Differences between the three models are smaller than the neural networks' run-to-run variation (up to about 0.02 macro-F1 between runs).
- The test block exposed low precision for the rarest class (broadleaf) that ROC-AUC did not show.

All 21 experiments are in `experiment_log.csv` (settings and validation metrics); test results are in `test_results.csv`.

## How to run

1. Open the notebook in Google Colab (badge above) and set the runtime to **GPU**.
2. Choose *Runtime, Run all*. The notebook installs nothing extra and downloads the data itself.


To run locally, install the packages in `requirements.txt` (Python 3.10 or newer recommended) and run the notebook with Jupyter.

## Repository structure

```
crop_mapping_project.ipynb   main notebook (data, experiments, evaluation)
requirements.txt             Python dependencies
README.md
```

Saved models, TensorBoard logs, test_results.csv, experiment_log.csv, and the zipped dataset are in the Google Drive folder, not in this repository.

## Reproducibility notes

- Random seeds are fixed (42). The scikit-learn results were identical across two separate runs.
- Neural network results are not bit-identical across runs on GPU, and macro-F1 varied by up to about 0.02 between two executions. Report conclusions about the networks with this variability in mind.
- Model selection used the validation block only; the test block was scored once at the end.

## Limitations

One region, one season and two acquisition dates; no field identifiers, so the block split reduces but cannot remove spatial leakage; very few samples of the rarest class (161 in the test block).

## Acknowledgements

Dataset: UCI Machine Learning Repository (reused, not collected by me). Libraries: scikit-learn, TensorFlow/Keras, pandas, NumPy, Matplotlib, Seaborn. No pretrained models were used.

## Author

Allan TUMUSIME, African Leadership University.
