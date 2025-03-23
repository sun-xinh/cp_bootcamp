import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import joblib
import os

###############################
# CONFIGURATION
###############################
INPUT_FOLDER = "/PHShome/xs081/cp_bootcamp/result_gene_19/"
INPUT_DATA = f"{INPUT_FOLDER}cdk9_df.csv"
OUTPUT_DIR = f"{INPUT_FOLDER}ml_models"
TOP_N_FEATURES = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Hyperparameter grids
LASSO_ALPHAS = np.logspace(-4, 2, 50)
RIDGE_ALPHAS = np.logspace(-2, 4, 50)

###############################
# MAIN FUNCTIONS
###############################

def load_data(auc_column):
    """Load data and select top features"""
    # Load merged data
    df = pd.read_csv(INPUT_DATA)
    
    # Load correlations
    corr_file = f"corr_{auc_column.split()[0]}.csv"
    try:
        correlations = pd.read_csv(os.path.join(INPUT_FOLDER, corr_file), 
                                index_col=0, header=0).squeeze()
        
        # Select top features
        top_features = correlations.abs().nlargest(TOP_N_FEATURES).index.tolist()
        print(top_features)
        output_df = df[top_features + [auc_column, "Status"]]
        return output_df.dropna(), top_features
    except FileNotFoundError:
        print(f"Correlation file {corr_file} not found. Using all features.")
        # If no correlation file, select all numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != auc_column and col != "Status"]
        return df[feature_cols + [auc_column, "Status"]], feature_cols

def stratified_split(X, y, status):
    """Stratified train-test split maintaining Status percentages"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE,
        stratify=status,
        random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_type, alphas):
    """Train model with cross-validated alpha selection"""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso() if model_type == 'lasso' else Ridge())
    ])
    
    param_grid = {'model__alpha': alphas}
    
    searcher = GridSearchCV(pipeline, param_grid, 
                          cv=5, scoring='neg_mean_squared_error',
                          n_jobs=-1)
    searcher.fit(X_train, y_train)
    
    print(f"Best {model_type.capitalize()} alpha: {searcher.best_params_['model__alpha']:.6f}")
    
    return searcher.best_estimator_, searcher.best_params_

def evaluate_model(model, X, y, set_name):
    """Evaluate model performance"""
    preds = model.predict(X)
    return {
        'set': set_name,
        'r2': r2_score(y, preds),
        'mae': mean_absolute_error(y, preds),
        'mse': mean_squared_error(y, preds),
        'rmse': np.sqrt(mean_squared_error(y, preds))
    }

def print_prediction_equation(model, feature_names, auc_column):
    """Print the prediction equation for linear models"""
    if not hasattr(model.named_steps['model'], 'coef_'):
        print("Prediction equation only available for linear models")
        return
    
    coefs = model.named_steps['model'].coef_
    intercept = model.named_steps['model'].intercept_
    
    # Create the equation string
    equation = f"{auc_column} = {intercept:.4f}"
    
    # Count non-zero coefficients
    non_zero_count = sum(1 for coef in coefs if abs(coef) > 1e-10)
    
    # Print full equation or truncated version
    if non_zero_count <= 20:
        # Full equation if there are few non-zero terms
        for i, (name, coef) in enumerate(zip(feature_names, coefs)):
            if abs(coef) > 1e-10:  # Only include non-zero coefficients
                sign = "+" if coef > 0 else "-"
                equation += f" {sign} {abs(coef):.4f} × {name}"
    else:
        # Truncated equation with top 10 most important terms
        important_features = [(name, coef) for name, coef in zip(feature_names, coefs) if abs(coef) > 1e-10]
        important_features.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for name, coef in important_features[:10]:
            sign = "+" if coef > 0 else "-"
            equation += f" {sign} {abs(coef):.4f} × {name}"
        
        equation += f" + ... ({non_zero_count - 10} more terms)"
    
    print("\nPrediction Equation:")
    print(equation)
    
    # Save to text file
    with open(os.path.join(OUTPUT_DIR, f"{auc_column}_equation.txt"), 'w') as f:
        f.write(equation)
        
        # Also save the full coefficients for reference
        f.write("\n\nFull Coefficients:\n")
        for name, coef in zip(feature_names, coefs):
            if abs(coef) > 1e-10:
                f.write(f"{name}: {coef:.6f}\n")
    
    return equation

def create_pdf_plots(model, X_train, X_test, y_train, y_test, features, model_type, auc_column):
    """Create diagnostic plots and save to PDF"""
    pdf_filename = os.path.join(OUTPUT_DIR, f"{auc_column}_{model_type}_plots.pdf")
    
    with PdfPages(pdf_filename) as pdf:
        # Plot 1: Actual vs Predicted
        plt.figure(figsize=(10, 8))
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        plt.scatter(y_train, y_pred_train, alpha=0.5, label='Train')
        plt.scatter(y_test, y_pred_test, alpha=0.5, label='Test')
        
        # Plot the perfect prediction line
        min_val = min(min(y_train), min(y_test))
        max_val = max(max(y_train), max(y_test))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{auc_column} - Actual vs Predicted ({model_type.capitalize()})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        pdf.savefig()
        plt.close()
        
        # Plot 2: Feature Coefficients
        plt.figure(figsize=(12, 10))
        coefficients = model.named_steps['model'].coef_
        
        # Sort coefficients by absolute value for better visualization
        coeffs_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': coefficients
        })
        coeffs_df['AbsCoef'] = coeffs_df['Coefficient'].abs()
        coeffs_df = coeffs_df.sort_values('AbsCoef', ascending=False)
        
        # Plot top 20 features by importance
        top_n = min(20, len(coeffs_df))
        plt.barh(range(top_n), coeffs_df['Coefficient'].head(top_n))
        plt.yticks(range(top_n), coeffs_df['Feature'].head(top_n))
        plt.xlabel("Coefficient Value")
        plt.title(f"Top {top_n} Feature Coefficients - {model_type.capitalize()}")
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Plot 3: Residual Plot
        plt.figure(figsize=(10, 8))
        residuals_train = y_train - y_pred_train
        residuals_test = y_test - y_pred_test
        
        plt.scatter(y_pred_train, residuals_train, alpha=0.5, label='Train')
        plt.scatter(y_pred_test, residuals_test, alpha=0.5, label='Test')
        plt.axhline(y=0, color='r', linestyle='--')
        
        plt.xlabel("Predicted Value")
        plt.ylabel("Residual")
        plt.title(f"Residual Plot - {model_type.capitalize()}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        pdf.savefig()
        plt.close()
        
        # Plot 4: Residual Distribution
        plt.figure(figsize=(10, 8))
        
        sns.histplot(residuals_train, kde=True, label='Train', alpha=0.5)
        sns.histplot(residuals_test, kde=True, label='Test', alpha=0.5)
        
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.title(f"Residual Distribution - {model_type.capitalize()}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        pdf.savefig()
        plt.close()
        
        print(f"Plots saved to {pdf_filename}")

###############################
# MAIN EXECUTION
###############################

def main(auc_column):
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data and select features
    data, features = load_data(auc_column)
    X = data[features]
    y = data[auc_column]
    status = data["Status"]
    
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Stratified split
    X_train, X_test, y_train, y_test = stratified_split(X, y, status)
    
    # Train Lasso
    print("\nTraining Lasso model...")
    lasso_model, lasso_params = train_model(X_train, y_train, 'lasso', LASSO_ALPHAS)
    
    # Evaluate Lasso
    lasso_train_metrics = evaluate_model(lasso_model, X_train, y_train, 'train')
    lasso_test_metrics = evaluate_model(lasso_model, X_test, y_test, 'test')
    
    # Print Lasso metrics
    print(f"\nLasso Model Metrics:")
    print(f"Train R² = {lasso_train_metrics['r2']:.4f}, RMSE = {lasso_train_metrics['rmse']:.4f}")
    print(f"Test R² = {lasso_test_metrics['r2']:.4f}, RMSE = {lasso_test_metrics['rmse']:.4f}")
    
    # Print Lasso equation
    print_prediction_equation(lasso_model, features, auc_column)
    
    # Create Lasso plots
    create_pdf_plots(lasso_model, X_train, X_test, y_train, y_test, features, 'lasso', auc_column)
    
    # Train Ridge
    print("\nTraining Ridge model...")
    ridge_model, ridge_params = train_model(X_train, y_train, 'ridge', RIDGE_ALPHAS)
    
    # Evaluate Ridge
    ridge_train_metrics = evaluate_model(ridge_model, X_train, y_train, 'train')
    ridge_test_metrics = evaluate_model(ridge_model, X_test, y_test, 'test')
    
    # Print Ridge metrics
    print(f"\nRidge Model Metrics:")
    print(f"Train R² = {ridge_train_metrics['r2']:.4f}, RMSE = {ridge_train_metrics['rmse']:.4f}")
    print(f"Test R² = {ridge_test_metrics['r2']:.4f}, RMSE = {ridge_test_metrics['rmse']:.4f}")
    
    # Print Ridge equation
    print_prediction_equation(ridge_model, features, auc_column)
    
    # Create Ridge plots
    create_pdf_plots(ridge_model, X_train, X_test, y_train, y_test, features, 'ridge', auc_column)
    
    # Save results to CSV
    results = pd.DataFrame({
        'Metric': ['R²', 'MAE', 'MSE', 'RMSE'],
        'Lasso_Train': [lasso_train_metrics['r2'], lasso_train_metrics['mae'], 
                        lasso_train_metrics['mse'], lasso_train_metrics['rmse']],
        'Lasso_Test': [lasso_test_metrics['r2'], lasso_test_metrics['mae'], 
                       lasso_test_metrics['mse'], lasso_test_metrics['rmse']],
        'Ridge_Train': [ridge_train_metrics['r2'], ridge_train_metrics['mae'], 
                        ridge_train_metrics['mse'], ridge_train_metrics['rmse']],
        'Ridge_Test': [ridge_test_metrics['r2'], ridge_test_metrics['mae'], 
                       ridge_test_metrics['mse'], ridge_test_metrics['rmse']]
    })
    results.to_csv(os.path.join(OUTPUT_DIR, f"{auc_column}_metrics.csv"), index=False)
    
    # Save models
    joblib.dump(lasso_model, os.path.join(OUTPUT_DIR, f"{auc_column}_lasso.pkl"))
    joblib.dump(ridge_model, os.path.join(OUTPUT_DIR, f"{auc_column}_ridge.pkl"))
    
    # Save feature list
    pd.Series(features).to_csv(os.path.join(OUTPUT_DIR, f"{auc_column}_features.csv"), index=False)
    
    print(f"\nAll results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <AUC_COLUMN>")
        sys.exit(1)
    
    auc_column = sys.argv[1]
    main(auc_column)
