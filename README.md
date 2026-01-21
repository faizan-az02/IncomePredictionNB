## Income Prediction — Adult Census Income

This repo contains a single Jupyter notebook that trains a **Naive Bayes** classifier to predict whether a person earns **\> \$50K/year** using the **UCI Adult Census Income** dataset.

### What’s inside

- **Notebook**: `IncomePrediction-NaiveBayes.ipynb`
- **Dataset archive**: `adult.zip`

### Approach

- **Data loading**: reads `adult.data`, for training and `adult.test`, for test, then concatenates for preprocessing.
- **Preprocessing**:
  - Label-encodes all categorical columns using `LabelEncoder`
  - Drops `fnlwgt` and `education`
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

10-fold CV:

- **Mean accuracy**: ~0.7991

### Requirements

You’ll need Python + Jupyter, plus these libraries:

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
