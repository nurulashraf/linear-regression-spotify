

---

# **Regression Analysis Project**

This repository demonstrates three types of regression analysis using Python:
1. **Linear Regression**
2. **Multiple Regression (2 variables)**
3. **Multiple Regression (more than 2 variables)**

The project focuses on training and evaluating regression models with:
- **Metrics**: Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
- **Visualisation**: Scatter plots comparing actual vs predicted values
- **Summaries**: Text-based results to summarise the performance

---

## **Project Structure**

```
regression-analysis-project/
│
├── README.md                      # Project overview
├── requirements.txt               # Python dependencies
├── data/                          # Datasets used in the analysis
│   └── dataset.csv
│
├── src/                           # Source code for the project
│   ├── utils/                     # Shared utilities
│   │   ├── data_preprocessing.py  # Functions for loading and cleaning data
│   │   ├── evaluation.py          # MSE and RMSE calculation
│   │   └── visualisation.py       # Functions to create plots
│   │
│   ├── linear_regression/         # Code for Linear Regression analysis
│   │   ├── linear_model.py        # Model training script
│   │   └── main_linear.py         # Main script to run analysis
│   │
│   ├── multiple_regression_two/   # Code for Multiple Regression (2 variables)
│   │   ├── multiple_model_two.py  # Model training script
│   │   └── main_multiple_two.py   # Main script to run analysis
│   │
│   └── multiple_regression_many/  # Code for Multiple Regression (> 2 variables)
│       ├── multiple_model_many.py # Model training script
│       └── main_multiple_many.py  # Main script to run analysis
│
└── results/                       # Results for each regression type
    ├── linear_regression/
    │   ├── linear_metrics.json           # Evaluation metrics
    │   ├── linear_plot.png               # Actual vs Predicted scatter plot
    │   └── linear_summary.txt            # Text summary of results
    │
    ├── multiple_regression_two/
    │   ├── multiple_two_metrics.json
    │   ├── multiple_two_plot.png
    │   └── multiple_two_summary.txt
    │
    └── multiple_regression_many/
        ├── multiple_many_metrics.json
        ├── multiple_many_plot.png
        └── multiple_many_summary.txt
```

---

## **How to Use**

### **1. Install Dependencies**
Clone this repository and install the required Python libraries:
```bash
git clone https://github.com/yourusername/regression-analysis-project.git
cd regression-analysis-project
pip install -r requirements.txt
```

### **2. Add Your Dataset**
Place your dataset in the `data/` folder and name it `dataset.csv`. Ensure the dataset includes the appropriate features and target variable for analysis.

### **3. Run the Analysis**
Run the main scripts for each type of regression:
- **Linear Regression**:
  ```bash
  python src/linear_regression/main_linear.py
  ```
- **Multiple Regression (2 variables)**:
  ```bash
  python src/multiple_regression_two/main_multiple_two.py
  ```
- **Multiple Regression (> 2 variables)**:
  ```bash
  python src/multiple_regression_many/main_multiple_many.py
  ```

### **4. View Results**
Results will be saved in the `results/` directory:
- **`metrics.json`**: Contains MSE and RMSE values.
- **`plot.png`**: Scatter plot of actual vs predicted values.
- **`summary.txt`**: Text summary of the model's performance.

---

## **Example Outputs**

### **Metrics**
**Example: `results/linear_regression/metrics.json`**
```json
{
    "MSE": 25.3421,
    "RMSE": 5.0342
}
```

### **Plot**
A scatter plot comparing the actual vs predicted values:

![Actual vs Predicted](results/linear_regression/plot.png)

### **Summary**
**Example: `results/linear_regression/summary.txt`**
```
Linear Regression Results
Mean Squared Error (MSE): 25.3421
Root Mean Squared Error (RMSE): 5.0342
```

---

## **Directory Details**

### **1. `data/`**
This folder contains the dataset(s) used for training and testing the models.

### **2. `src/`**
- **`utils/`**: Contains helper functions for data preprocessing, evaluation, and plotting.
- **`linear_regression/`**: Scripts for linear regression analysis.
- **`multiple_regression_two/`**: Scripts for multiple regression analysis with 2 variables.
- **`multiple_regression_many/`**: Scripts for multiple regression analysis with more than 2 variables.

### **3. `results/`**
Each regression type has its own folder containing:
- `metrics.json`: Evaluation metrics (MSE and RMSE).
- `plot.png`: Scatter plot of actual vs predicted values.
- `summary.txt`: Text summary of the results.

---

## **Dependencies**
The following Python libraries are required:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## **About This Project**
This project demonstrates my skills in:
- Building and evaluating regression models
- Modularising code for reusability
- Presenting results in a clear and concise format

Feel free to explore the repository and contact me for further discussion!
