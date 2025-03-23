import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import joblib

###############################
# CONFIGURATION
###############################
INPUT_FOLDER = "/PHShome/xs081/cp_bootcamp/result_gene_19/"
INPUT_DATA = f"{INPUT_FOLDER}cdk9_df.csv"
OUTPUT_DIR = f"{INPUT_FOLDER}stratified_models1000"
TOP_N_FEATURES = 2000

# Custom bins with 4 classes (25% sensitive, 25% semi-sensitive, 25% semi-resistant, 25% resistant)
CUSTOM_BINS = [0, 0.25, 0.5, 0.75, 1.0]
CLASS_LABELS = ['sensitive', 'semi-sensitive', 'semi-resistant', 'resistant']

# Hyperparameter grids
DT_PARAMS = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

RF_PARAMS = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 0.8]
}

###############################
# MAIN FUNCTIONS
###############################

def load_data(auc_column):
    """Load and split data into sets with quartile-based bins, stratifying by class distribution"""
    try:
        # Load original data
        df = pd.read_csv(INPUT_DATA)
        print(f"Original data shape: {df.shape}")
        
        # Load correlations and select top features
        corr_file = f"corr_{auc_column.split()[0]}.csv"
        correlations = pd.read_csv(os.path.join(INPUT_FOLDER, corr_file), 
                                  index_col=0, header=0).squeeze()
        features = correlations.abs().nlargest(TOP_N_FEATURES).index.tolist()
        
        # Filter out rows with NaN in the AUC column or any feature columns
        columns_to_check = features + [auc_column, 'Status']
        df_clean = df.dropna(subset=columns_to_check)
        print(f"Data shape after removing NaNs: {df_clean.shape}")
        print(f"Removed {df.shape[0] - df_clean.shape[0]} rows with NaN values")
        
        # Separate hematologic and solid samples
        heme_df = df_clean[df_clean['Status'] == 'hematologic']
        solid_df = df_clean[df_clean['Status'] == 'solid']
        
        print(f"Solid samples: {solid_df.shape[0]}")
        print(f"Hematologic samples: {heme_df.shape[0]}")
        
        # Check if we have enough data after filtering
        if solid_df.shape[0] < 100:  # Arbitrary threshold
            print(f"Warning: Only {solid_df.shape[0]} solid samples after filtering. Results may be unreliable.")
        if heme_df.shape[0] < 20:  # Arbitrary threshold
            print(f"Warning: Only {heme_df.shape[0]} hematologic samples after filtering. Results may be unreliable.")
        
        # First, create quartile bins based on the entire solid dataset
        # This ensures that each class has approximately 25% of the solid data
        solid_quartiles = solid_df[auc_column].quantile([0.25, 0.5, 0.75])
        
        # Create bins based on these quartiles
        dynamic_bins = [
            solid_df[auc_column].min() - 0.001,  # Use slightly below min to ensure inclusion
            solid_quartiles[0.25],
            solid_quartiles[0.5],
            solid_quartiles[0.75],
            solid_df[auc_column].max() + 0.001   # Use slightly above max to ensure inclusion
        ]
        
        print(f"\nDynamic bins based on solid data quartiles:")
        for i, (lower, upper) in enumerate(zip(dynamic_bins[:-1], dynamic_bins[1:])):
            print(f"{CLASS_LABELS[i]}: {lower:.4f} to {upper:.4f}")
        
        # Create target variable for solid data
        solid_df['target'] = pd.cut(
            solid_df[auc_column],
            bins=dynamic_bins,
            labels=CLASS_LABELS,
            include_lowest=True
        )
        
        # Now split solid data into train/test, stratifying by the target variable
        # This ensures that train and test sets have similar class distributions
        test_size = heme_df.shape[0]
        
        solid_train, solid_test = train_test_split(
            solid_df,
            test_size=test_size,
            stratify=solid_df['target'],  # Stratify by target class
            random_state=42
        )
        
        # Create target variable for hematologic data using the same bins
        heme_df['target'] = pd.cut(
            heme_df[auc_column],
            bins=dynamic_bins,
            labels=CLASS_LABELS,
            include_lowest=True
        )
        
        # Check for any NaN in target
        for df_name, df_set in [("solid_train", solid_train), ("solid_test", solid_test), ("heme_df", heme_df)]:
            if df_set['target'].isna().any():
                print(f"Warning: Found {df_set['target'].isna().sum()} NaN values in {df_name} target")
                df_set = df_set.dropna(subset=['target'])
                print(f"Removed NaN values, new {df_name} shape: {df_set.shape}")
        
        # Final verification of data integrity
        for df_set in [solid_train[features], solid_test[features], heme_df[features]]:
            assert not df_set.isna().any().any(), "NaN values found in feature data"
        
        assert not solid_train['target'].isna().any(), "NaN values found in solid_train target"
        assert not solid_test['target'].isna().any(), "NaN values found in solid_test target"
        assert not heme_df['target'].isna().any(), "NaN values found in heme_df target"
        
        # Print class distribution
        print("\nClass distribution:")
        for df_name, df_set in [("Solid Train", solid_train), ("Solid Test", solid_test), ("Hematologic", heme_df)]:
            class_dist = df_set['target'].value_counts(normalize=True).sort_index() * 100
            print(f"{df_name}: {', '.join([f'{c}: {v:.1f}%' for c, v in class_dist.items()])}")
        
        return (
            solid_train[features], solid_train['target'],
            solid_test[features], solid_test['target'],
            heme_df[features], heme_df['target'],
            features,
            correlations
        )
        
    except Exception as e:
        print(f"Data loading error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def train_model(X_train, y_train, model_type, params):
    """Train model on solid training data"""
    try:
        model = DecisionTreeClassifier() if model_type == 'dt' else RandomForestClassifier()
        
        searcher = GridSearchCV(
            model, params,
            cv=5, scoring='f1_weighted',
            n_jobs=-1, verbose=1
        )
        
        print(f"\nTraining {model_type.upper()} classifier...")
        searcher.fit(X_train, y_train)
        
        print(f"Best parameters: {searcher.best_params_}")
        print(f"Best CV F1: {searcher.best_score_:.4f}")
        
        return searcher.best_estimator_
        
    except Exception as e:
        print(f"Training error: {str(e)}")
        sys.exit(1)

def save_confusion_matrices(model, X_solid_test, y_solid_test, 
                           X_heme_test, y_heme_test, features, auc_column):
    """Save confusion matrices for both test sets"""
    pdf_filename = os.path.join(OUTPUT_DIR, f"{auc_column}_test_results.pdf")
    
    with PdfPages(pdf_filename) as pdf:
        # Solid test confusion matrix
        plt.figure(figsize=(10, 8))  # Larger figure for 4 classes
        preds = model.predict(X_solid_test)
        cm = confusion_matrix(y_solid_test, preds, labels=CLASS_LABELS)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
        disp.plot(cmap='Blues')
        plt.title("Solid Test Set Confusion Matrix")
        pdf.savefig()
        plt.close()
        
        # Print classification report for solid test
        print("\nSolid Test Set Classification Report:")
        print(classification_report(y_solid_test, preds, labels=CLASS_LABELS))
        
        # Hematologic test confusion matrix
        plt.figure(figsize=(10, 8))  # Larger figure for 4 classes
        preds = model.predict(X_heme_test)
        cm = confusion_matrix(y_heme_test, preds, labels=CLASS_LABELS)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_LABELS)
        disp.plot(cmap='Reds')
        plt.title("Hematologic Test Set Confusion Matrix")
        pdf.savefig()
        plt.close()
        
        # Print classification report for hematologic test
        print("\nHematologic Test Set Classification Report:")
        print(classification_report(y_heme_test, preds, labels=CLASS_LABELS))

def plot_decision_tree(dt_model, features, auc_column):
    """Plot decision tree visualization"""
    plt.figure(figsize=(20, 15))
    plot_tree(
        dt_model,
        feature_names=features,
        class_names=CLASS_LABELS,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title(f"Decision Tree for {auc_column}", fontsize=16)
    plt.tight_layout()
    
    # Save the tree visualization
    tree_file = os.path.join(OUTPUT_DIR, f"{auc_column}_decision_tree.pdf")
    plt.savefig(tree_file, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Decision tree visualization saved to {tree_file}")

def plot_feature_importance(model, features, correlations, model_name, auc_column):
    """Plot top 20 feature importance with direction (positive/negative correlation)"""
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print(f"Warning: {model_name} does not have feature_importances_ attribute")
        return
    
    # Create dataframe with feature importances
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances,
        'Direction': [np.sign(correlations.get(feat, 0)) for feat in features]
    })
    
    # Get top 20 features by importance
    top_features = importance_df.nlargest(20, 'Importance')
    
    # Set colors based on direction (positive/negative correlation)
    colors = ['red' if d < 0 else 'green' for d in top_features['Direction']]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1], color=colors[::-1])
    plt.title(f"Top 20 Feature Importances ({model_name})", fontsize=14)
    plt.xlabel('Importance')
    
    # Add a legend for direction
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Positive correlation'),
        Patch(facecolor='red', label='Negative correlation')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Save the feature importance plot
    importance_file = os.path.join(OUTPUT_DIR, f"{auc_column}_{model_name}_feature_importance.pdf")
    plt.tight_layout()
    plt.savefig(importance_file, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {importance_file}")

###############################
# MAIN EXECUTION
###############################

def main(auc_column):
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load and split data
        print(f"\n{'='*50}\nProcessing: {auc_column}\n{'='*50}")
        (X_train, y_train, 
         X_solid_test, y_solid_test,
         X_heme_test, y_heme_test,
         features, correlations) = load_data(auc_column)
        
        print(f"\nData splits:")
        print(f"\nFeature: {TOP_N_FEATURES}")
        print(f"Solid training: {X_train.shape[0]} samples")
        print(f"Solid test: {X_solid_test.shape[0]} samples")
        print(f"Hematologic test: {X_heme_test.shape[0]} samples")
        
        # Train models
        dt_model = train_model(X_train, y_train, 'dt', DT_PARAMS)
        rf_model = train_model(X_train, y_train, 'rf', RF_PARAMS)
        
        # Save confusion matrices
        for name, model in [('DecisionTree', dt_model), ('RandomForest', rf_model)]:
            save_confusion_matrices(
                model, X_solid_test, y_solid_test,
                X_heme_test, y_heme_test,
                features, f"{auc_column}_{name}"
            )
        
        # Plot decision tree visualization
        plot_decision_tree(dt_model, features, f"{auc_column}_DecisionTree")
        
        # Plot feature importances with direction
        plot_feature_importance(dt_model, features, correlations, "DecisionTree", auc_column)
        plot_feature_importance(rf_model, features, correlations, "RandomForest", auc_column)
        
        # Save models
        joblib.dump(dt_model, os.path.join(OUTPUT_DIR, f"{auc_column}_dt_model.pkl"))
        joblib.dump(rf_model, os.path.join(OUTPUT_DIR, f"{auc_column}_rf_model.pkl"))
        
        print(f"\n{'='*50}\nAnalysis complete!\nResults saved to: {OUTPUT_DIR}\n{'='*50}")

    except Exception as e:
        print(f"\nMain execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <AUC_column_name>")
        sys.exit(1)
    
    main(sys.argv[1])
