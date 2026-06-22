"""Grid axes for the conformal EHR benchmark.

Single source of truth for the cell axes: framework.py maps these names to PyHealth
classes and run_grid.py expands them into results.csv.
"""

DATASETS = ["mimic3", "mimic4", "eicu"]
TASKS = ["los", "mortality", "readmission"]
MODELS = ["Transformer", "RNN", "RETAIN"]
METHODS = ["LABEL", "APS", "RAPS"]
MODES = ["marginal", "class-conditional"]
ALPHAS = [0.2, 0.1, 0.05, 0.01]
SEEDS = [0, 1, 2, 3, 4]
RATIOS = [0.6, 0.1, 0.1, 0.2]
SPLIT = "patient 0.6/0.1/0.1/0.2"

# dataset -> tables to load
TABLES = {
    "mimic3": ["diagnoses_icd", "procedures_icd", "prescriptions"],
    "mimic4": ["diagnoses_icd", "procedures_icd", "prescriptions"],
    "eicu": ["diagnosis", "medication", "physicalexam"],
}
READMISSION_WINDOW_DAYS = 30

# RAPS regularization (TorchCP penalty=0 default is degenerate = APS)
RAPS_K_REG = 5
RAPS_LAMBDA = 0.01

# task -> native output type
OUTPUT_TYPE = {"los": "multiclass", "mortality": "binary", "readmission": "binary"}

# task -> output field key
LABEL_KEY = {"los": "los", "mortality": "mortality", "readmission": "readmission"}

# task -> validation monitor
MONITOR = {
    "los": "f1_macro",
    "mortality": "roc_auc_weighted_ovr",
    "readmission": "roc_auc_weighted_ovr",
}

# task -> training metrics
METRICS = {
    "los": ["f1_macro", "accuracy"],
    "mortality": ["roc_auc_weighted_ovr", "f1_macro", "accuracy"],
    "readmission": ["roc_auc_weighted_ovr", "f1_macro", "accuracy"],
}
