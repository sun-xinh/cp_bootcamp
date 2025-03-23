import pandas as pd
import sys
import os

###############################
# CONFIGURATION
###############################

# Input paths
INPUT_TARGET = "/PHShome/xs081/cp_bootcamp/data/all_cdk.csv"
INPUT_FEATURES = "/PHShome/xs081/cp_bootcamp/data/protein.csv"
COL_CELL_LINE = "Unnamed: 0"  # Merging column

# Output configuration
OUTPUT_DIR = "/PHShome/xs081/cp_bootcamp/result_protein19"
MERGED_DATA_PATH = os.path.join(OUTPUT_DIR, "cdk9_df.csv")

AUC_COLUMNS = [
    "StrippedCellLineName",  # Will be excluded from correlations
    "OncotreeLineage",       # Will be excluded from correlations
    "Status",                # Will be excluded from correlations
    "SEL-120",              "JWZ-5-13",            
    "THZ1",                 "THAL-SNS-032",        
    "THZ531",               "ATUVECICLIB",         
    "ABEMACICLIB",          "BSJ-03-123",          
    "BI-1347",              "ATIRMOCICLIB",        
    "NVP-2",                "BSJ-4-116",           
    "DINACICLIB",           "HQ461",               
    "TAGTOCICLIB",          "RIBOCICLIB",          
    "PALBOCICLIB",          "AZD4573",             
    "EBVACICLIB"
]

###############################
# MAIN FUNCTIONS
###############################

def generate_merged_data():
    """Generate and save the merged dataset"""
    print("Loading target data...")
    target = pd.read_csv(INPUT_TARGET)
    
    print("Loading features...")
    features = pd.read_csv(INPUT_FEATURES)
    
    print("Merging datasets...")
    merged = target.merge(features, on=COL_CELL_LINE, how="inner")
    
    print(f"Saving merged data to {MERGED_DATA_PATH}")
    os.makedirs(os.path.dirname(MERGED_DATA_PATH), exist_ok=True)
    merged.to_csv(MERGED_DATA_PATH, index=False)
    return merged

def calculate_single_correlation(df, auc_column):
    """Calculate correlations for a single AUC column"""
    print(f"\nProcessing {auc_column}...")
    
    # Skip metadata columns
    if auc_column in ["StrippedCellLineName", "OncotreeLineage", "Status"]:
        print(f"Skipping metadata column: {auc_column}")
        return
    
    if auc_column not in df.columns:
        raise ValueError(f"Column {auc_column} not found in dataset")
    
    # Get feature columns (exclude metadata and AUC columns)
    feature_cols = [col for col in df.columns 
                   if col not in AUC_COLUMNS + [COL_CELL_LINE]]
    
    # Calculate correlations
    correlations = df[feature_cols].corrwith(df[auc_column])
    
    # Create safe filename
    base_name = auc_column.replace("/", "-").replace(" ", "_")
    output_path = os.path.join(OUTPUT_DIR, f"corr_{base_name}.csv")
    
    correlations.to_csv(output_path)
    print(f"Saved correlations to {output_path}")

###############################
# MAIN EXECUTION
###############################

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <AUC_COLUMN>")
        print("Available AUC columns:")
        print("\n".join([col for col in AUC_COLUMNS if col not in ["StrippedCellLineName", "OncotreeLineage", "Status"]]))
        sys.exit(1)
        
    target_column = sys.argv[1]
    
    # Generate or load merged data
    if not os.path.exists(MERGED_DATA_PATH):
        print("Generating merged dataset...")
        merged_df = generate_merged_data()
    else:
        print("Loading existing merged data...")
        merged_df = pd.read_csv(MERGED_DATA_PATH)
    
    # Calculate correlations
    calculate_single_correlation(merged_df, target_column)
