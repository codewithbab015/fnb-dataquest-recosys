# Library
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import logging
from datetime import datetime

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from argparse import ArgumentParser

# Logging setup ──────────────────────────────
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "processing_summary.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# Load raw dataset from CSV in directory
def loading_raw_data(filepath: Path) -> pd.DataFrame:
    file = next(filepath.glob('*.csv'))
    logger.info(f"Loading data from: {file}")
    return pd.read_csv(file)


# Quick data overview and cleaning
def data_overview(data: pd.DataFrame) -> pd.DataFrame:
    data['item'] = data['item'].replace('NONE', np.nan)

    print(f"Total records: {len(data):,}")
    print("*" * 50)
    
    missing = data.isna().sum()
    print("Missing values per column:")
    for col, val in missing.items():
        print(f"  {col}: {val}")
        
    print("*" * 50)
    
    duplicates = data.duplicated().sum()
    print(f"Duplicated records: {duplicates:,}")
    print("*" * 50)
    
    print("Data types:")
    for col, dtype in data.dtypes.items():
        print(f"  {col}: {dtype}")
        
    print("*" * 50)
    return data

    
# Pre-cleaning of dataset
def data_precleaning(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(columns='item_descrip')
    data['user_id'] = data['idcol'].astype(str)
    data = data.drop(columns='idcol')

    for col in data.select_dtypes(include='object'):
        data[col] = data[col].str.lower()
    
    data['int_date'] = pd.to_datetime(data['int_date'], errors='coerce')
    data = data.drop_duplicates(keep='first')
    data['item'] = data['item'].fillna('no_item_id')

    return data.reset_index(drop=True)
    
# Visualize dataset features
def data_visualisation(data: pd.DataFrame) -> pd.DataFrame:
    def plot_feature_distribution_count(df: pd.DataFrame, feature: str, palette: str = 'viridis') -> None:
        plt.style.use('ggplot')
        plt.figure(figsize=(8, 4))
        order = df[feature].value_counts().index
        sns.countplot(data=df, x=feature, order=order, palette=palette)
        plt.xlabel(feature.capitalize())
        plt.title(f'Distribution of {feature}')
        plt.tight_layout(pad=2)
        plt.show()

    def plot_high_cardinal_features(df: pd.DataFrame, feature: str, color: str = 'red', top_n: int = 10) -> None:
        top_data = df[feature].value_counts().nlargest(top_n)
        plt.style.use('ggplot')
        plt.figure(figsize=(6, 6))
        plt.barh(top_data.index, top_data.values, color=color)
        plt.gca().invert_yaxis()
        plt.xlabel('Count')
        plt.ylabel(feature.capitalize())
        plt.title(f'Top {top_n} {feature.capitalize()} by Frequency')
        plt.tight_layout(pad=2)
        plt.show()

    def time_oriented_plot(df: pd.DataFrame) -> None:
        df['int_date'] = pd.to_datetime(df['int_date'])
        weekly = df['int_date'].dt.to_period('W-SUN').value_counts().sort_index()
        weekly.index = weekly.index.to_timestamp()
        rolling_avg = weekly.rolling(window=4).mean()
        plt.figure(figsize=(12, 5))
        plt.plot(weekly.index, weekly.values, label='Weekly Count', marker='o', color='skyblue')
        plt.plot(rolling_avg.index, rolling_avg.values, label='4-Week Rolling Avg', linestyle='--', color='orange')
        plt.title('Weekly Customer Interactions with Rolling Average')
        plt.xlabel('Week')
        plt.ylabel('Interaction Count')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_high_cardinal_features(data, 'item')
    plot_high_cardinal_features(data, 'beh_segment', 'blue')
    plot_feature_distribution_count(data, 'active_ind')
    plot_feature_distribution_count(data, 'interaction')
    plot_feature_distribution_count(data, 'segment')
    plot_feature_distribution_count(data, 'item_type')
    plot_feature_distribution_count(data, 'tod')
    plot_feature_distribution_count(data, 'page')
    time_oriented_plot(data)

    return data

# Save DataFrame as CSV in processed data folder
def write_processed_data_to_csv(file_path: Path, df: pd.DataFrame) -> None:
    df.to_csv(file_path, index=False)
    logger.info(f"Processed data saved to {file_path.as_posix()}")


# CLI Input Arguments
def cli_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Accepts command-line arguments required to process the data.")
    parser.add_argument("-r", "--raw", help="Raw dataset", required=True)
    parser.add_argument("-p", "--process", help="Raw dataset", required=True)
    
    args = parser.parse_args()
    
    return args

# Run data pipeline
def main() -> None:
    cli_parser = cli_arguments()
    print(cli_parser)
    # filepath = cli_parser.raw
    
    # data = loading_raw_data(filepath)

    # data = data_overview(data)
    # data = data_precleaning(data)

    # logger.info("Sample of cleaned data:")
    # logger.info(f"\n{data.head()}")

    # output_path = Path.cwd() / "data/processed"
    # output_path.mkdir(parents=True, exist_ok=True)
    # write_processed_data_to_csv(output_path / "processed_fnb.csv", data)


if __name__ == "__main__":
    main()
