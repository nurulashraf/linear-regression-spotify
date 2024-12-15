# Spotify Tracks Multiple Linear Regression Analysis

## Overview
This analysis explores the relationship between multiple musical features (Energy and Danceability) to predict track Valence using multiple linear regression on a Spotify tracks dataset.

## Dataset
- **Source**: Spotify tracks dataset
- **Features Analyzed**: 
  - Independent Variables (X): 
    1. Energy
    2. Danceability
  - Dependent Variable (Y): Valence

## Methodology

### Data Preparation
- Loaded Spotify tracks dataset using pandas
- Selected relevant features: Energy, Danceability, and Valence
- Split data into training (70%) and testing (30%) sets

### Multiple Linear Regression Approach
- Used scikit-learn's LinearRegression model
- Trained on the relationship between Energy, Danceability, and Valence
- Performed predictive modeling and evaluation

## Key Findings

### Visualization Techniques
1. **3D Matplotlib Visualizations**
   - Created two 3D plots to represent:
     a) Regression plane with actual and predicted data points
     b) Comparison of actual vs. predicted valence values

2. **Interactive Plotly 3D Visualization**
   - Provided an interactive 3D representation
   - Displayed actual and predicted data points
   - Showed regression plane for better understanding

### Model Performance
- **Mean Squared Error (MSE)**: Indicates average squared difference between predicted and actual valence
- **Root Mean Squared Error (RMSE)**: Standard deviation of prediction errors

## Example Prediction
- **Input**: 
  - Energy: 0.97
  - Danceability: 0.753
- **Predicted Valence**: (Model's predicted value)

## Visualisation Insights
- 3D plots help visualize the complex relationship between:
  - Energy
  - Danceability
  - Valence
- Regression plane shows the predictive model's surface

## Limitations and Considerations
- Uses only two features for prediction
- Linear model assumes linear relationships
- Performance may vary with different datasets or feature combinations

## Recommendations
- Explore additional feature interactions
- Consider non-linear modeling techniques
- Validate model with cross-validation
- Experiment with feature engineering

## Technical Details
- **Features Used**: 
  - X1: Energy
  - X2: Danceability
  - Y: Valence

- **Model Type**: Multiple Linear Regression
- **Train-Test Split**: 70% training, 30% testing
- **Random State**: 0 (for reproducibility)

## Libraries Used
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning tools
- matplotlib: Static plotting
- plotly: Interactive 3D visualization

## Code Repository
[[GitHub]](https://github.com/nurulashraf/lin-reg-spt)

## License
MIT

## Contact
nurulashraf23@gmail.com

## Future Work
- Investigate non-linear regression techniques
- Explore additional Spotify track features
- Develop more sophisticated predictive models
