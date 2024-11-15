**Tech stack:** scikit-learn, Matplotlib, Seaborn, Pandas, NumPy

**Dataset used:** Cancer Cell Dataset from UCI Machine Learning Repository (Asuncion and Newman, 2007), link [**here**](http://mlearn.ics.uci.edu/MLRepository.html).

**Description:** An SVM model from the scikit-learn library was trained on a real dataset of human cell characteristics to determine if a cell is benign or malignant, as confirmed by separate medical procedures. The dataset was cleaned and correlation values were visualised to select training features for the SVM. The selected features were normalized and split into train/test sets. The model was trained on the train set and validated on the test set for every value of the hyperparameter ‘kernel’. The best performing kernel value was chosen and the validation metrics were cross-checked.

**Resulting accuracy metrics:**
- Best kernel = 'poly'
- Jaccard score = 0.93
- f1 score = 0.95
- Classification report accuracy = 0.95
- Confusion matrix: TP = 67, TN = 128, FP = 6, FN = 4
