"""
Feature Engineering Pipeline for Recommendation System
"""
from utils_engineer import *

warnings.filterwarnings('ignore')


# Logging Configuration
def setup_logging() -> logging.Logger:
    """Configure logging for the feature engineering pipeline."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "feature_engineering.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Data Loading and Preprocessing
def load_data(filepath: Path) -> pd.DataFrame:
    """Load dataset from CSV file."""
    logger.info(f"Loading data from: {filepath}")
    return pd.read_csv(filepath)

def standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names for consistency."""
    column_mapping = {
        'idcol': 'user_id',
        'item': 'item_id',
        'page': 'screen_page',
        'tod': 'time_of_day',
        'active_ind': 'active_mode'
    }
    return data.rename(columns=column_mapping).reset_index(drop=True)

def create_interaction_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create user-item interaction features."""
    # Basic interaction counts
    data['total_user_interactions'] = data.groupby('user_id')['item_id'].transform('count')
    data['user_item_interactions'] = data.groupby(['user_id', 'item_id'])['item_id'].transform('count')
    
    # Item popularity
    data['normalized_popularity'] = data.groupby('item_id')['user_id'].transform('count')
    
    # User diversity metrics
    data['user_unique_items'] = data.groupby('user_id')['item_id'].transform('nunique')
    data['user_unique_item_types'] = data.groupby('user_id')['item_type'].transform('nunique')
    data['user_unique_segments'] = data.groupby('user_id')['segment'].transform('nunique')
    data['user_unique_behavior_segments'] = data.groupby('user_id')['beh_segment'].transform('nunique')
    
    return data

def create_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features."""
    # Parse dates
    data['int_date'] = pd.to_datetime(data['int_date'], errors='coerce')
    
    # Calendar features
    data['day_of_week'] = data['int_date'].dt.dayofweek + 1
    data['week_of_month'] = ((data['int_date'].dt.day - 1) // 7 + 1).astype('Int64')
    data['month'] = data['int_date'].dt.month
    
    # Time flags
    data['is_weekend'] = data['day_of_week'].isin([6, 7]).astype(int)
    data['is_month_start'] = data['int_date'].dt.is_month_start.astype(int)
    data['is_month_end'] = data['int_date'].dt.is_month_end.astype(int)
    
    # Interaction rates by time
    data['weekend_interaction_rate'] = data.groupby('user_id')['is_weekend'].transform('mean')
    data['month_start_interaction_rate'] = data.groupby('user_id')['is_month_start'].transform('mean')
    data['month_end_interaction_rate'] = data.groupby('user_id')['is_month_end'].transform('mean')
    
    # Cyclical features
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    
    return data

def create_sequence_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create sequential interaction features."""
    # Sort by user and time
    data = data.sort_values(['user_id', 'int_date'])
    
    # Previous interaction features
    data['prev_item_id'] = data.groupby('user_id')['item_id'].shift(1).fillna('no_prev_item')
    data['prev_item_type'] = data.groupby('user_id')['item_type'].shift(1).fillna('no_action')
    
    # Time between interactions
    data['days_since_last_interaction'] = (
        data.groupby(['user_id', 'item_id'])['int_date']
        .diff()
        .dt.days
        .fillna(-1)
    )
    
    # Frequency of previous interactions
    prev_item_counts = data['prev_item_id'].value_counts(normalize=True)
    prev_type_counts = data['prev_item_type'].value_counts(normalize=True)
    
    data['prev_item_frequency'] = data['prev_item_id'].map(prev_item_counts).fillna(0)
    data['prev_type_frequency'] = data['prev_item_type'].map(prev_type_counts).fillna(0)
    
    # Clean up temporary columns
    data.drop(['prev_item_id', 'prev_item_type'], axis=1, inplace=True)
    
    return data

def encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables."""
    # Label encode ordinal features
    categorical_cols = ['beh_segment', 'segment', 'time_of_day']
    for col in categorical_cols:
        if col in data.columns:
            data[col] = pd.factorize(data[col])[0]
    
    # One-hot encode nominal features
    nominal_cols = ['item_type', 'active_mode', 'screen_page']
    for col in nominal_cols:
        if col in data.columns:
            dummies = pd.get_dummies(data[col], prefix=col, dtype=float)
            data = pd.concat([data.drop(columns=col), dummies], axis=1)
    
    return data

def process_features(data: pd.DataFrame) -> pd.DataFrame:
    """Main feature engineering pipeline."""
    # Ensure required columns exist
    required_cols = ['user_id', 'item_id', 'item_type', 'segment', 'beh_segment', 
                    'active_mode', 'time_of_day']
    for col in required_cols:
        if col not in data.columns:
            data[col] = np.nan
    
    # Fill missing values
    data['item_id'].fillna('no_item_id', inplace=True)
    
    # Apply feature engineering steps
    data = create_interaction_features(data)
    data = create_temporal_features(data)
    data = create_sequence_features(data)
    data = encode_categorical_features(data)
    
    # Rename target column
    if 'interaction' in data.columns:
        data.rename(columns={'interaction': 'target'}, inplace=True)
    
    # Reorder columns
    target_cols = ['user_id', 'target', 'item_id']
    other_cols = [col for col in data.columns if col not in target_cols]
    data = data[target_cols + other_cols]
    
    # Remove rows with no item_id
    data = data[data['item_id'] != 'no_item_id']
    
    logger.info(f"Feature engineering completed. Final shape: {data.shape}")
    return data.reset_index(drop=True)

# Train/Test Split Functions
def create_train_test_split(data: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and testing sets."""
    # Separate features and target
    exclude_cols = ['user_id', 'target', 'item_id', 'int_date']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    X = data[feature_cols]
    y = data['target']
    
    # Perform stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    # Reconstruct DataFrames with metadata
    train_meta = data.loc[X_train.index, ['user_id', 'item_id']]
    test_meta = data.loc[X_test.index, ['user_id', 'item_id']]
    
    train = pd.concat([train_meta, X_train, y_train], axis=1)
    test = pd.concat([test_meta, X_test, y_test], axis=1)
    
    return train, test

def save_datasets(data: pd.DataFrame, train_path: Path, test_path: Path) -> None:
    """Create and save train/test splits."""
    train, test = create_train_test_split(data)
    
    # Save datasets
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    
    logger.info(f"Train size: {len(train):,} | Test size: {len(test):,}")
    logger.info(f"Saved train data to: {train_path}")
    logger.info(f"Saved test data to: {test_path}")

# Command Line Interface
def parse_arguments() -> ArgumentParser:
    """Parse command line arguments."""
    parser = ArgumentParser(description="Feature Engineering Pipeline for Recommendation System")
    parser.add_argument("--process", type=str, required=True, help="Path to processed data input file")
    parser.add_argument("--train", type=str, required=True, help="Path to training data output file")
    parser.add_argument("--test", type=str, required=True, help="Path to test data output file")
    return parser.parse_args()

def setup_paths(args) -> Tuple[Path, Path, Path]:
    """Create and validate paths."""
    process_path = Path(args.process).resolve()
    train_path = Path(args.train).resolve()
    test_path = Path(args.test).resolve()
    
    # Validate input path
    if not process_path.exists():
        raise FileNotFoundError(f"Input data file does not exist: {process_path}")
    
    # Create output directories
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Process Path: {process_path}")
    logger.info(f"Train Path: {train_path}")
    logger.info(f"Test Path: {test_path}")
    
    return process_path, train_path, test_path

def main() -> None:
    """Main pipeline execution."""
    try:
        # Parse arguments and setup paths
        args = parse_arguments()
        process_path, train_path, test_path = setup_paths(args)
        
        # Load and process data
        logger.info("Starting feature engineering pipeline...")
        data = load_data(process_path)
        data = standardize_columns(data)
        data = process_features(data)
        
        # Create and save train/test splits
        logger.info("Creating train/test splits...")
        save_datasets(data, train_path, test_path)
        
        logger.info("Feature engineering pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()