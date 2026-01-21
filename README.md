## Income Prediction — Adult Census Income

This repo contains a single Jupyter notebook that trains a **Naive Bayes** classifier to predict whether a person earns **\> \$50K/year** using the **UCI Adult (Census Income)** dataset.

### What’s inside

- **Notebook**: `IncomePrediction-NaiveBayes.ipynb`
- **Dataset archive**: `adult.zip` (unzips to `adult.data`, `adult.test`, and metadata files)

### Approach

- **Data loading**: reads `adult.data` (train) and `adult.test` (test), then concatenates for preprocessing.
- **Preprocessing**:
  - Label-encodes all categorical columns using `LabelEncoder`
  - Drops `fnlwgt` and `education` (keeps `education_num`)
  - Scales features with `MinMaxScaler`
- **Imbalance handling**: applies **SMOTE** to the scaled training split
- **Model**: `GaussianNB`
- **Evaluation**:
  - Accuracy / Precision / Recall / F1
  - Confusion matrix heatmap
  - ROC curve + ROC-AUC
  - 10-fold stratified cross-validation
  - Null-accuracy comparison

### Results

On the provided `adult.test` split:

- **Accuracy**: ~0.8199  
- **ROC-AUC**: ~0.8547  

10-fold CV (on full dataset):

- **Mean accuracy**: ~0.7991 (range ~0.7920–0.8061)

### Requirements

You’ll need Python + Jupyter, plus these libraries (as used by the notebook):

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`

Install:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### How to run

1. Open the notebook:
   - `IncomePrediction-NaiveBayes.ipynb`
2. Run the cells from top to bottom.

The notebook includes a cell that unzips the dataset:

```bash
unzip adult.zip
```

If `unzip` isn’t available on your machine, extract `adult.zip` using your OS tools, then rerun the notebook.
