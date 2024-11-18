import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def unified_model(df):
    # Preprocessing and filtering
    df = df[df['quarter'] != 'Unknown/NotCentralArea'].copy()  # Remove unknown quarters
    df['log_price'] = np.log1p(df['price'])  # Log-transform the price

    # Define numerical predictors
    numerical_columns = [
        'accommodates', 'bathrooms', 'bedrooms',
        'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy',
        'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
        'review_scores_location', 'review_scores_value'
    ]

    # Drop rows with missing values in numerical predictors
    df_filtered = df.dropna(subset=numerical_columns).copy()

    # Standardize numerical predictors
    scaler = StandardScaler()
    X_numerical_scaled = scaler.fit_transform(df_filtered[numerical_columns])
    X_numerical = pd.DataFrame(X_numerical_scaled, columns=numerical_columns, index=df_filtered.index)

    # Handle quarters as categorical variable
    cheapest_quarter = df_filtered.groupby('quarter')['price'].mean().idxmin()
    df_filtered['quarter'] = pd.Categorical(
        df_filtered['quarter'],
        categories=[cheapest_quarter] + [q for q in df_filtered['quarter'].unique() if q != cheapest_quarter],
        ordered=True
    )

    # One-hot encode quarters
    quarter_dummies = pd.get_dummies(df_filtered['quarter'], prefix='quarter', drop_first=True)
    quarter_dummies = quarter_dummies.astype(float)  # Ensure all dummies are numerical (float)

    # Combine numerical predictors and quarter dummies
    X_combined = pd.concat([X_numerical, quarter_dummies], axis=1)
    X_combined = sm.add_constant(X_combined)  # Add constant for intercept

    # Align dependent variable with the cleaned X_combined
    y = df_filtered.loc[X_combined.index, 'log_price']

    # Check and handle missing or infinite values in X_combined
    if X_combined.isnull().any().any():
        print("Warning: Missing values detected in independent variables. Dropping rows with missing values.")
        valid_index = X_combined.dropna().index
        X_combined = X_combined.loc[valid_index]
        y = y.loc[valid_index]  # Align dependent variable

    if np.isinf(X_combined).any().any():
        print("Warning: Infinite values detected in independent variables. Dropping rows with infinite values.")
        valid_index = X_combined[~np.isinf(X_combined).any(axis=1)].index
        X_combined = X_combined.loc[valid_index]
        y = y.loc[valid_index]  # Align dependent variable

    # Fit OLS regression model
    model = sm.OLS(y, X_combined).fit()

    # Display regression summary
    print("\nUnified Model Summary:")
    print(model.summary())

    # Compute predictions
    y_pred_log = model.predict(X_combined)
    residuals = y - y_pred_log

    # Residual analysis
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_log, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs Fitted Values (Unified Model)')
    plt.xlabel('Fitted Values (Log Price)')
    plt.ylabel('Residuals')
    plt.show()

    # Histogram of residuals
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Histogram of Residuals (Unified Model)')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()

    # Evaluate model performance on original price scale
    y_pred = np.expm1(y_pred_log)  # Convert predictions back to original price scale
    mse = mean_squared_error(df_filtered.loc[y.index, 'price'], y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df_filtered.loc[y.index, 'price'], y_pred)
    r2 = r2_score(df_filtered.loc[y.index, 'price'], y_pred)

    print("\nModel Performance Metrics (Unified Model):")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"R-squared: {r2:.2f}")
