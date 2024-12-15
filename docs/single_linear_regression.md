# Spotify Tracks Linear Regression Analysis

## Overview
This analysis explores the relationship between track valence and danceability using single linear regression on a Spotify tracks dataset.

## Dataset
- **Source**: Spotify tracks dataset
- **Features Analyzed**: 
  - Independent Variable (X): Valence
  - Dependent Variable (Y): Danceability

## Methodology

### Data Preparation
- Loaded dataset using pandas
- Extracted valence and danceability features
- Split data into training (80%) and testing (20%) sets

### Linear Regression Approach
- Used scikit-learn's LinearRegression model
- Trained on the relationship between valence and danceability
- Performed predictive modeling and evaluation

## Key Findings

### Correlation Analysis
- Pearson correlation coefficient calculated between valence and danceability
- Correlation strength and direction determined

### Model Performance
- **Mean Squared Error (MSE)**: Indicates average squared difference between predicted and actual danceability
- **Root Mean Squared Error (RMSE)**: Standard deviation of prediction errors

### Visualisations
1. **Correlation Heatmap**
   - Displayed correlations between numeric features
   - Used color intensity to represent correlation strength

2. **Scatter Plot (Valence vs Danceability)**
   - Visualized raw relationship between valence and danceability
   - Helped assess linear relationship potential

3. **Train-Test Split Visualization**
   - Showed distribution of training and testing data points
   - Ensured representative data splitting

4. **Model Prediction Visualization**
   - Compared predicted regression line with actual test data
   - Illustrated model's predictive performance

## Example Prediction
- **Input**: Valence = 0.821
- **Predicted Danceability**: (Model's predicted value)

## Limitations and Considerations
- Single feature (valence) used for prediction
- Linear model assumes linear relationship
- Performance may vary with different datasets

## Recommendations
- Consider multivariate regression for more complex predictions
- Explore non-linear modeling techniques
- Validate model with additional feature interactions

## Libraries Used
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning tools
- seaborn: Statistical data visualization
- matplotlib: Plotting and graphing

## Code Repository
[[GitHub Repository]](https://github.com/nurulashraf/lin-reg-spt)

## License
MIT 

## Contact
nurulashraf23@gmail.com
