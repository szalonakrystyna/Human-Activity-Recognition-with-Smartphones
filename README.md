# Human-Activity-Recognition-with-Smartphones
# HAR Pipeline — analiza sygnałów IMU i klasyfikacja aktywności

Repozytorium zawiera kompletny pipeline do przetwarzania sygnałów IMU (akcelerometr + żyroskop) i klasyfikacji aktywności na bazie cech czasowych i częstotliwościowych. Skrypt wspiera dwa formaty danych:

- per-sample `train.csv` / `test.csv` (np. mirror Kaggle),
- oryginalną strukturę **UCI HAR** (`UCI HAR Dataset/.../inertial_signals`).

Pipeline obejmuje:
- automatyczne wczytywanie i segmentację sygnałów (sliding windows),
- preprocessing (bandpass + detrend),
- ekstrakcję cech (czasowe / częstotliwościowe),
- selekcję cech (VarianceThreshold),
- skalowanie (StandardScaler),
- PCA (wizualizacja),
- klasyfikatory: **RandomForest**, **XGBoost**, opcjonalnie **SVM** dla binary,
- generowanie wykresów + zapisywanie modeli (`*.joblib`).

Wyniki i artefakty

Skrypt generuje:

Wykresy
class_distribution.png
pca_scatter.png
feature_importances_rf.png
feature_importances_xgb.png
confusion_matrix_xgb.png
roc_compare.png (tylko dla binary)

Modele
rf_model.joblib
xgb_model.joblib
best_model.joblib (jeśli użyto SVM)
scaler.joblib
pca.joblib
variance_threshold.joblib
label_encoder.joblib
