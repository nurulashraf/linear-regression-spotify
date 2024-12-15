

---

# **Spotify Tracks Regression Analysis Project**

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
lin-reg-spt/
│
├── README.md                      
├── LICENSE                      
├── requirements.txt            
├── data/                         
│   └── dataset.csv/                       
│ 
├── single_linear_regression/ 
│   ├── single_linear_regression.py
│   ├── results/
│       ├── actual_vs_predicted.png
│       ├── correlation_matrix.png 
│       ├── scatter_plot_valence_vs_danceability.png
│
├── multiple_linear_regression_2vars/
│   ├── multiple_linear_regression_2vars.py
│   ├── results/
│       ├── actual_vs_predicted_3d.png
│       ├── regression_plane_3d_plot.png
│
├── multiple_linear_regression_more_vars/
│   ├── multiple_linear_regression_more_vars.py
│   ├── results/
│       ├── actual_vs_predicted.png
│       ├── correlation_matrix.png
│       ├── scatter_plot_speechiness_vs_valence.png
│
└── docs/                      
    ├── single_linear_regression.md
    ├── multiple_linear_regression_2vars.md
    ├── multiple_linear_regression_more_vars.md

```

---

## Repository Structure 
- `data/`: Contains the dataset used for analysis.
- `single_linear_regression/`: Contains the script and results for single linear regression analysis.
- `multiple_linear_regression_2vars/`: Contains the script and results for multiple linear regression with 2 variables.
- `multiple_linear_regression_more_vars/`: Contains the script and results for multiple linear regression with more than 2 variables.
- `docs/`: Additional documentation explaining each analysis

## Setup Instructions 
1. Clone the repository:
   ```sh
   git clone https://github.com/nurulashraf/lin-reg-spt.git

2. Navigate to the project directory:
   ```sh
   cd lin-reg-spt
   
3. Install the required libraries:
   ```sh
   pip install -r requirements.txt

## Running the Analyses

### Single Linear Regression

Navigate to the `single_linear_regression` directory and run the script:
```sh
python multiple_linear_regression_2vars.py
```
### Multiple Linear Regression (2 variables)
Navigate to the `smultiple_linear_regression_2vars` directory and run the script:
```sh
python multiple_linear_regression_2vars.py
```
### Multiple Linear Regression (more than 2 variables)
Navigate to the `multiple_linear_regression_more_vars` directory and run the script:
```sh
python multiple_linear_regression_more_vars.py
```

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


## License
This project is licensed under the MIT License - see the LICENSE file for details.

Feel free to explore the repository and contact me for further discussion!
