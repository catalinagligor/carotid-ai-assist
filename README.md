# Multimodal Analysis for Cardiovascular Risk

This project combines segmentation of carotid ultrasound images with cardiovascular risk classification based on clinical data.

## Datasets Used

The project uses two main data sources from Kaggle.

- The imaging data comes from the **Carotid Ultrasound Images** dataset: [Kaggle dataset](https://www.kaggle.com/datasets/orvile/carotid-ultrasound-images), which contains ultrasound images of the carotid artery and masks for segmentation.
- The classification data comes from the **Cardiovascular Disease dataset**: [Kaggle dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset), which contains 70,000 patient records, 11 features, and the target variable.

In the project, the imaging data is saved as NumPy files for train, validation, and test splits, while the clinical data is kept in separate CSV files for the same subsets. For the clinical part, both numerical and categorical variables are used.

## Trained Models

### U-Net and Attention U-Net

For image segmentation, I used U-Net, a classic and efficient architecture for medical images, and then compared it with Attention U-Net.

The Attention U-Net variant adds an attention mechanism that helps the model focus on relevant areas in the image and can reduce the influence of the background. Comparing the two models is useful to see whether attention brings a real improvement in segmenting carotid structures.

U-Net achieved:

- **Best Val Dice:** 0.966400
- **Last Train Loss:** 0.038777
- **Last Val Loss:** 0.041145

Attention U-Net achieved:

- **Best Val Dice:** 0.967272
- **Last Train Loss:** 0.034791
- **Last Val Loss:** 0.040028

Attention U-Net has a slightly better result on validation and a lower loss value at the end, suggesting a modest improvement over U-Net.  
In addition, the difference in parameters is small, so the gain comes more from the attention mechanism than from a large increase in model complexity.

### XGBoost

For cardiovascular risk classification, the main model is **XGBoost**.

It was chosen because it performs very well on tabular data and can leverage both preprocessed clinical features and TF-IDF representations extracted from text.

On validation, XGBoost achieves:

- **Accuracy:** 0.7334
- **F1-Score:** 0.7183
- **AUC-ROC:** 0.8014

### Logistic Regression

**Logistic Regression** is used as a baseline model, to compare the performance of a simple model with that of the XGBoost model.

On validation, the model achieves:

- **Accuracy:** 0.7310
- **F1-Score:** 0.7122
- **AUC-ROC:** 0.7935

## Project Structure

The logical structure of the project is as follows.

- `notebooks/` — contains the notebooks used for exploration, preprocessing, training, and evaluation:
  - `01-carotid-eda.ipynb`
  - `02-preprocessing.ipynb`
  - `03-training.ipynb`
  - `04-evaluation.ipynb`
- `app.py` — the application.
- `requirements.txt` — the Python dependencies required to run the project.

The training notebook loads the data, defines the models, performs training, and saves the checkpoints and final results.

## Generated Files

At the end of the run, the following are saved:

- `unet_best.pth` — the best U-Net model.
- `unet_final.pth` — the final U-Net model.
- `attention_unet_best.pth` — the best Attention U-Net model.
- `attention_unet_final.pth` — the final Attention U-Net model.
- `xgboost_model.pkl` — the main classification model.
- `logistic_regression.pkl` — the baseline model.
- `model_comparison.csv` — metrics comparison.
- `unet_training_curves.png` — training curves for segmentation.

## Remarks

The project treats the imaging part and the clinical part separately, which makes it easy to extend later toward a complete multimodal solution.  
A natural direction for development is integrating the results of the two components into a single final prediction model.
