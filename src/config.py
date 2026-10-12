"""Central configuration for the optimization experiments."""

# ---- Dataset ----
DATASET_NAME = "breast_cancer"  
DATASET_DIR = "dataset"                
DATASET_FILE = "breast_cancer.csv"     

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

# Relevance-redundancy objective
REDUNDANCY_PENALTY = 0.10

# (candidate feature count, maximum selected features)
EXPERIMENTS = [
    (10, 5),
    (20, 10),
    (30, 15),
]

# ILP settings
ILP_TIME_LIMIT = 120  # seconds

RESULTS_DIR = "results"