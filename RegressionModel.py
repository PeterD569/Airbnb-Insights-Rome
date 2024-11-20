import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def unified_model(df):
    """
    Perform regression analysis on an Airbnb dataset to predict log-transformed prices (log_price)
    using numerical predictors, categorical variables (quarter, room_type with 'Private room' as baseline),
    and the presence of top 20 amenities.

    Parameters:
        df (pd.DataFrame): Input dataset with preprocessed price, numerical predictors,
                           categorical variables, and amenity indicators.

    Outputs:
        - Regression summary.
        - Residual plots.
        - Model performance metrics.
    """
    # Filter out rows with missing or non-central quarters
    df = df[df['quarter'].notnull() & (df['quarter'] != 'Unknown/NotCentralArea')].copy()

    # Log-transform the price
    df['log_price'] = np.log1p(df['price'])

    # Define numerical predictors
    numerical_columns = [
        'accommodates', 'bathrooms', 'bedrooms',
        'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy',
        'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication',
        'review_scores_location', 'review_scores_value'
    ]

    # Include top 20 amenities as predictors
    amenity_counts = df['amenities_list'].explode().value_counts()
    top_20_amenities = amenity_counts.head(20).index
    for amenity in top_20_amenities:
        df[f'amenity_{amenity}'] = df['amenities_list'].apply(lambda x: amenity in x).astype(float)
    top_20_amenity_columns = [f'amenity_{amenity}' for amenity in top_20_amenities]

    # Drop rows with missing values in numerical predictors
    all_predictors = numerical_columns + top_20_amenity_columns
    df_filtered = df.dropna(subset=all_predictors).copy()

    # Standardize numerical predictors
    scaler = StandardScaler()
    X_numerical_scaled = scaler.fit_transform(df_filtered[numerical_columns])
    X_numerical = pd.DataFrame(X_numerical_scaled, columns=numerical_columns, index=df_filtered.index)

    # Include binary amenity columns as-is (already numeric)
    X_amenities = df_filtered[top_20_amenity_columns]

    # Handle quarter as a categorical variable
    cheapest_quarter = df_filtered.groupby('quarter')['price'].mean().idxmin()
    df_filtered['quarter'] = pd.Categorical(
        df_filtered['quarter'],
        categories=[cheapest_quarter] + [q for q in df_filtered['quarter'].unique() if q != cheapest_quarter],
        ordered=True
    )
    quarter_dummies = pd.get_dummies(df_filtered['quarter'], prefix='quarter', drop_first=True)
    quarter_dummies = quarter_dummies.astype(float)  # Explicitly convert to float

    # Handle room_type as a categorical variable with 'Private room' as the baseline
    df_filtered['room_type'] = pd.Categorical(df_filtered['room_type'], categories=['Private room', 'Entire home/apt', 'Hotel room', 'Shared room'])
    room_type_dummies = pd.get_dummies(df_filtered['room_type'], prefix='room_type', drop_first=True)
    room_type_dummies = room_type_dummies.astype(float)  # Explicitly convert to float

    # Combine numerical predictors, amenity indicators, quarter dummies, and room type dummies
    X_combined = pd.concat([X_numerical, X_amenities, quarter_dummies, room_type_dummies], axis=1)
    X_combined = sm.add_constant(X_combined)  # Add constant for intercept

    # Align dependent variable with the cleaned X_combined
    y = df_filtered.loc[X_combined.index, 'log_price']

    # Ensure all predictors are numerical
    non_numeric_columns = X_combined.select_dtypes(exclude=np.number).columns
    if not non_numeric_columns.empty:
        print("Non-numerical columns detected:", non_numeric_columns)
        raise ValueError("All predictor columns must be numerical.")

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
