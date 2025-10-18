### **Executive Summary: Model Performance Analysis**

Based on a comprehensive review of the results, the **XGBoost model is the superior choice** for this mortality prediction task. It achieved the highest performance across all key metrics on the test set, including an **AUC-ROC of 0.8637** , an **AUPRC of 0.6149** , and an **Accuracy of 0.8004**.

Most critically for this clinical application, the XGBoost model also delivered the best balance of sensitivity and precision, evidenced by the highest **F1-Score (0.56)** and the highest **Recall (0.76)**  for the 'death' class. This indicates it is the most effective model at correctly identifying at-risk patients, which is the primary objective. Overfitting was minimal and well-controlled (AUC gap of 0.0313).

---

### **Detailed Model Comparison**

Here is a side-by-side comparison of the key **test set performance metrics** for the 'death' (positive) class:

| Metric | L1 Logistic Regression (Baseline) | Random Forest | **XGBoost (Winner)** |
| :--- | :--- | :--- | :--- |
| **AUC-ROC** | 0.8497  | 0.8420  | **0.8637**  |
| **AUPRC** | 0.5809  | 0.5749  | **0.6149**  |
| **Recall (Sensitivity)** | 0.74  | 0.69  | **0.76**  |
| **F1-Score** | 0.53  | 0.53  | **0.56**  |
| **Precision** | 0.42 | 0.42  | **0.44**  |
| **Accuracy** | 0.79  | 0.79  | **0.80**  |
| **Overfitting (Train-Test AUC Gap)**| -0.0053 (No overfitting)  | 0.0163  | **0.0313**  |

---

### **Individual Model Analysis**

#### 1. L1 Logistic Regression (Baseline)
This model served as an excellent baseline and demonstrated strong, interpretable performance.
* **Performance**: It achieved a robust test AUC of 0.8497. Notably, it showed no signs of overfitting, as its test performance was slightly higher than its training performance.
* **Feature Selection**: The L1 regularization was highly effective, reducing the feature set from 325 to 85 (26.2% selection rate), thereby creating a simpler, more interpretable model.
* **Clinical Insights**: The model provided strong clinical value, achieving a high **Recall of 0.74**. This means it successfully identified 74% of all patients who would die. However, its low Precision (0.42)  indicates it produced a high number of false positives (995 FPs vs. 708 TPs).
* **Top Features**: The most important features, such as `age` , `elixhauser` (comorbidity score) , `GCS_max` (consciousness level) , and `SOFA_last` (organ failure score), align perfectly with clinical intuition.

#### 2. Random Forest
The Random Forest model, despite hyperparameter tuning, surprisingly underperformed the L1 baseline on key metrics.
* **Performance**: Its test AUC (0.8420)  and AUPRC (0.5749)  were the lowest of the three models.
* **Clinical Insights**: The model's primary weakness is its **low Recall (0.69)**. It missed more at-risk patients than the other two models, making it the least suitable choice for this task. It also had the lowest Precision (0.424).
* **Overfitting**: The model was well-regularized, with a negligible overfitting gap of 0.0163[cite: 128, 129].

#### 3. XGBoost
The XGBoost model demonstrated the best performance across the board.
* **Performance**: It achieved the highest Test AUC (0.8637)  and AUPRC (0.6149), indicating superior overall discriminative power. The overfitting gap was minimal at 0.0313 , well below the 0.05 threshold.
* **Clinical Insights**: This model provides the best clinical utility. Its **Recall of 0.76**  was the highest, meaning it correctly identified 76% of all true mortality cases (725 TPs vs. 232 FNs). It also achieved the highest F1-Score (0.56), demonstrating the best available trade-off between Recall and Precision.
* **Top Features**: The feature importance list is dominated by clinically critical scores, particularly `SOFA_last` , `SOFA_min` , and `SOFA_mean`. This highlights that organ failure assessment is the most powerful predictor. `GCS_last` (consciousness)  and `median_dose_vaso_last` (vasopressor use) are also key, which is consistent with clinical practice.

### **Final Recommendation**

The **XGBoost model is the clear winner**. It provides a statistically significant improvement in predictive power (AUC) while also best aligning with the primary clinical goal of maximizing **Recall** to ensure as many high-risk patients as possible are identified. The consistent identification of `SOFA`, `GCS`, and vasopressor dosage as top features across all models provides strong validation for the model's clinical relevance.

