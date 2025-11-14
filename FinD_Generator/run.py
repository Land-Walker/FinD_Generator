import sys
sys.path.append('\workspaces\AI_FinSys\FinD_Generator')

from src.data_collector import DataCollector
from src.data_loader import TimeGradDataModule

# 1. Collect or load data
collector = DataCollector()
dfs = collector.collect_all_data()  # returns dict of DataFrames

# ===========================================
# 2. Initialize DataModule
# ===========================================
dm = TimeGradDataModule(data_dict=dfs, device="cpu") # or "cuda" if available

# ===========================================
# 3. Preprocess, split, and transform data
# ===========================================
# This single method handles:
# - Building raw blocks (wavelet, returns, etc.)
# - Merging all data sources
# - Adding calendar features and regime labels
# - Splitting into train/val/test sets
# - Fitting scalers/PCA on the training set and transforming all sets
dm.preprocess_and_split()

# ===========================================
# 4. Build Datasets and Dataloaders
# ===========================================
dm.build_datasets()
train_loader = dm.train_dataloader()

# ===========================================
# 5. Inspect a sample batch
# ===========================================
print("\n🔍 Inspecting a sample batch from the train dataloader:")
sample_batch = next(iter(train_loader))
for key, tensor in sample_batch.items():
    print(f"  - {key}: {tensor.dtype}, {tensor.shape}")