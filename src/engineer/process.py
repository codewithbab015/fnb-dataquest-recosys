"""
Data processing pipeline for customer interaction data.
Loads, cleans, visualizes, and saves processed data.
"""

from utils_engineer import *

def setup_logging() -> logging.Logger:
    """Configure logging with file and console output."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "processing_summary.log", mode="w", encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def load_data(filepath: Path) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    logger.info(f"Loading data from: {filepath}")
    return pd.read_csv(filepath)


def analyze_data(data: pd.DataFrame) -> None:
    """Print data overview statistics."""
    logger.info(f"Total records: {len(data):,}")
    logger.info("=" * 50)
    
    # Missing values
    missing = data.isna().sum()
    if missing.sum() > 0:
        logger.info("Missing values per column:")
        for col, val in missing.items():
            if val > 0:
                logger.info(f"  {col}: {val}")
    else:
        logger.info("No missing values found")
    
    logger.info("=" * 50)
    logger.info(f"Duplicated records: {data.duplicated().sum():,}")
    logger.info("=" * 50)
    
    # Data types
    logger.info("Data types:")
    for col, dtype in data.dtypes.items():
        logger.info(f"  {col}: {dtype}")
    logger.info("=" * 50)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the dataset."""
    # Handle missing values
    data['item'] = data['item'].replace('NONE', np.nan).fillna('no_item_id')
    
    # Remove unnecessary columns and rename
    data = data.drop(columns=['item_descrip'], errors='ignore')
    if 'idcol' in data.columns:
        data['user_id'] = data['idcol'].astype(str)
        data = data.drop(columns=['idcol'])
    
    # Convert text to lowercase
    text_columns = data.select_dtypes(include='object').columns
    for col in text_columns:
        if col != 'user_id':  # Keep user_id as is
            data[col] = data[col].str.lower()
    
    # Convert date column
    if 'int_date' in data.columns:
        data['int_date'] = pd.to_datetime(data['int_date'], errors='coerce')
    
    # Remove duplicates
    data = data.drop_duplicates(keep='first').reset_index(drop=True)
    
    logger.info(f"Data cleaned. Final shape: {data.shape}")
    return data


def create_visualizations(data: pd.DataFrame) -> None:
    """Generate visualizations for key features."""
    plt.style.use('ggplot')
    
    def plot_top_categories(df: pd.DataFrame, feature: str, top_n: int = 10, 
                           color: str = 'skyblue', title: str = None) -> None:
        """Plot top N categories for high-cardinality features."""
        top_data = df[feature].value_counts().nlargest(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_data)), top_data.values, color=color)
        plt.yticks(range(len(top_data)), top_data.index)
        plt.gca().invert_yaxis()
        plt.xlabel('Count')
        plt.ylabel(feature.replace('_', ' ').title())
        plt.title(title or f'Top {top_n} {feature.replace("_", " ").title()}')
        plt.tight_layout()
        plt.show()
    
    def plot_distribution(df: pd.DataFrame, feature: str, palette: str = 'viridis') -> None:
        """Plot distribution for categorical features."""
        plt.figure(figsize=(10, 6))
        order = df[feature].value_counts().index
        sns.countplot(data=df, x=feature, order=order, palette=palette)
        plt.xlabel(feature.replace('_', ' ').title())
        plt.title(f'Distribution of {feature.replace("_", " ").title()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_time_series(df: pd.DataFrame) -> None:
        """Plot time series of interactions."""
        if 'int_date' not in df.columns:
            return
            
        df_temp = df.copy()
        df_temp['int_date'] = pd.to_datetime(df_temp['int_date'])
        
        # Weekly aggregation
        weekly = df_temp['int_date'].dt.to_period('W-SUN').value_counts().sort_index()
        weekly.index = weekly.index.to_timestamp()
        
        # Rolling average
        rolling_avg = weekly.rolling(window=4).mean()
        
        plt.figure(figsize=(12, 6))
        plt.plot(weekly.index, weekly.values, label='Weekly Count', 
                marker='o', color='skyblue', linewidth=2)
        plt.plot(rolling_avg.index, rolling_avg.values, 
                label='4-Week Rolling Average', linestyle='--', color='orange', linewidth=2)
        
        plt.title('Weekly Customer Interactions')
        plt.xlabel('Week')
        plt.ylabel('Interaction Count')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # Generate plots for different feature types
    high_cardinality_features = ['item', 'beh_segment']
    categorical_features = ['active_ind', 'interaction', 'segment', 'item_type', 'tod', 'page']
    
    for feature in high_cardinality_features:
        if feature in data.columns:
            plot_top_categories(data, feature)
    
    for feature in categorical_features:
        if feature in data.columns:
            plot_distribution(data, feature)
    
    plot_time_series(data)


def save_data(data: pd.DataFrame, output_path: Path) -> None:
    """Save processed data to CSV file."""
    working_dir = Path.cwd()
    filepath = (working_dir / output_path).resolve()
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    data.to_csv(filepath, index=False)
    logger.info(f"Processed data saved to {filepath}")


def parse_arguments() -> ArgumentParser:
    """Parse command line arguments."""
    parser = ArgumentParser(description="Process customer interaction data")
    parser.add_argument("-r", "--raw", required=True, help="Path to raw dataset CSV file")
    parser.add_argument("-p", "--process", required=True, help="Path to output directory for processed data")
    # parser.add_argument("--no_viz", action="store_true", help="Skip visualization generation")
    
    return parser.parse_args()

def load_config(filepath: str) -> Tuple[int, bool]:
    """Load and parse configuration from TOML file."""
    config = toml.load(filepath)
    
    size = config["DATA"]["size"]
    is_visual = config["DATA"]["is_visual"]
    
    return size, is_visual

def process_data_size(data, size: int) -> object:
    """Process data based on size configuration."""
    if size == 0:
        return data
    return data[:size]

def main() -> None:
    """Main processing pipeline."""

    args = parse_arguments()
    
    try:
        # Load configuration
        data_size, is_visual = load_config("data.config.toml")
        
        # Load and process data
        data = load_data(Path(args.raw))
        analyze_data(data)
        data = clean_data(data)
        
        # Log sample
        logger.info("Sample of cleaned data:")
        logger.info(f"\n{data.head()}")
        
        # Generate visualizations if enabled
        if is_visual:
            print("Creating visualizations...")
            # create_visualizations(data)
        
        # Process and save data
        processed_data = process_data_size(data, data_size)
        save_data(processed_data, Path(args.process))
        
        # Log completion
        final_size = len(processed_data)
        logger.info(f"Data size: {final_size}")
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()