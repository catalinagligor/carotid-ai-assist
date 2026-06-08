# Analiză multimodală pentru risc cardiovascular

Acest proiect combină segmentarea imaginilor ecografice carotidiene cu clasificarea riscului cardiovascular pe baza datelor clinice. 
## Dataseturi folosite

Proiectul folosește două surse principale de date din Kaggle.

- Datele imagistice provin din datasetul **Carotid Ultrasound Images**: [Kaggle dataset](https://www.kaggle.com/datasets/orvile/carotid-ultrasound-images), care conține imagini ecografice ale arterei carotide și măști pentru segmentare.
- Datele pentru clasificare provin din datasetul **Cardiovascular Disease dataset**: [Kaggle dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset), care conține 70.000 de înregistrări de pacienți, 11 caracteristici și variabila țintă.

În proiect, datele imagistice sunt salvate ca fișiere NumPy pentru train, validation și test, iar datele clinice sunt păstrate în fișiere CSV separate pentru aceleași subseturi. Pentru partea clinică sunt folosite variabile numerice si categorice.

## Modele antrenate

### U-Net și Attention U-Net
Pentru segmentarea imaginilor am folosit U-Net, o arhitectură clasică și eficientă pentru imagini medicale, apoi am comparat-o cu Attention U-Net. 

Varianta Attention U-Net adaugă un mecanism de atenție care ajută modelul să se concentreze pe zonele relevante din imagine și poate reduce influența fundalului. Compararea celor două modele este utilă pentru a vedea dacă atenția aduce o îmbunătățire reală în segmentarea structurilor carotidiene.

U-Net a obținut:

**Best Val Dice de 0.966400**

**Last Train Loss 0.038777**

**Last Val Loss 0.041145** 

Attention U-Net a obținut:

**Best Val Dice de 0.967272**

**Last Train Loss 0.034791**

**Last Val Loss 0.040028** 


Attention U-Net are un rezultat puțin mai bun pe validation și o valoare de loss mai mică la final, ceea ce sugerează o îmbunătățire modestă față de U-Net. 
În plus, diferența de parametri este mică, deci câștigul vine mai degrabă din mecanismul de atenție decât dintr-o creștere mare a complexității modelului.

### XGBoost

Pentru clasificarea riscului cardiovascular, modelul principal este **XGBoost**. 

Acesta a fost ales deoarece funcționează foarte bine pe date tabulare și poate valorifica atât atributele clinice preprocesate, cât și reprezentările TF-IDF extrase din text.

Pe validation, XGBoost obține Accuracy 0.7334, F1-Score 0.7183 și AUC-ROC 0.8014.

### Logistic Regression

**Logistic Regression** este folosită ca model baseline, pentru a compara performanța unui model simplu cu cea a modelului XGBoost.
Pe validation, modelul obține Accuracy 0.7310, F1-Score 0.7122 și AUC-ROC 0.7935.

## Structura proiectului

Structura logică a proiectului este următoarea.

- `notebooks/` — conține notebook-urile folosite pentru explorare, preprocesare, antrenare și evaluare:
  - `01-carotid-eda.ipynb`
  - `02-preprocessing.ipynb`
  - `03-training.ipynb`
  - `04-evaluation.ipynb`
- `app.py` — aplicația.
- `requirements.txt` — dependențele Python necesare pentru rulare.

Notebook-ul de training încarcă datele, definește modelele, realizează antrenarea și salvează checkpoint-urile și rezultatele finale.

## Fișiere generate

La finalul rulării sunt salvate următoarele:

- `unet_best.pth` — cel mai bun model U-Net.
- `unet_final.pth` — modelul U-Net final.
- `attention_unet_best.pth` — cel mai bun model Attention U-Net.
- `attention_unet_final.pth` — modelul Attention U-Net final.
- `xgboost_model.pkl` — modelul principal de clasificare.
- `logistic_regression.pkl` — modelul baseline.
- `model_comparison.csv` — comparația metricilor.
- `unet_training_curves.png` — curbele de antrenare pentru segmentare.

## Observații

Proiectul tratează separat partea imagistică și partea clinică, ceea ce îl face ușor de extins ulterior către o soluție multimodală completă.
O direcție firească de dezvoltare este integrarea rezultatelor celor două componente într-un singur model final de predicție.