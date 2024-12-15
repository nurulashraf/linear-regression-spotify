# Spotify Tracks Multi-Variable Linear Regression Analysis

## Overview
This analysis explores the relationship between multiple musical features to predict track Valence using multiple linear regression on a Spotify tracks dataset.

## Dataset
- **Source**: Spotify tracks dataset
- **Features Analyzed**: 
  - Independent Variables (X): 
    1. Speechiness
    2. Energy
    3. Danceability
  - Dependent Variable (Y): Valence

## Methodology

### Data Preparation
- Loaded Spotify tracks dataset using pandas
- Selected relevant features: Speechiness, Energy, Danceability, and Valence
- Split data into training (70%) and testing (30%) sets

### Multiple Linear Regression Approach
- Used scikit-learn's LinearRegression model
- Trained on the relationship between multiple features and Valence
- Performed predictive modeling and evaluation

## Key Findings

### Visualisations
1. **Actual vs Predicted Valence Plot**
   - Scatter plot comparing actual and predicted valence values
   - Helps assess model's prediction accuracy

2. **Correlation Heatmap**
   - Visualized correlations between numeric features
   - Provides insights into feature relationships

3. **Feature vs Target Scatter Plot**
   - Explored relationship between Speechiness and Valence
   - Helps understand individual feature impact

### Model Performance
- **Mean Squared Error (MSE)**: Indicates average squared difference between predicted and actual valence
- **Root Mean Squared Error (RMSE)**: Standard deviation of prediction errors

## Example Prediction
- **Input**: 
  - Speechiness: 0.103
  - Energy: 0.97
  - Danceability: 0.753
- **Predicted Valence**: (Model's predicted value)

## Visualisation Insights
- Correlation heatmap reveals relationships between different musical features
- Scatter plots help understand non-linear and linear relationships
- Actual vs Predicted plot shows model's prediction accuracy

## Limitations and Considerations
- Uses three features for prediction
- Linear model assumes linear relationships
- Performance may vary with different datasets or feature combinations

## Recommendations
- Explore additional feature interactions
- Consider non-linear modeling techniques
- Validate model with cross-validation
- Experiment with feature engineering

## Technical Details
- **Features Used**: 
  - X1: Speechiness
  - X2: Energy
  - X3: Danceability
  - Y: Valence

- **Model Type**: Multiple Linear Regression
- **Train-Test Split**: 70% training, 30% testing
- **Random State**: 0 (for reproducibility)

## Libraries Used
- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: Machine learning tools
- matplotlib: Static plotting
- seaborn: Statistical data visualization
- scipy: Statistical analysis

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
- Perform feature importance analysis
