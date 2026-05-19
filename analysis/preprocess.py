import pandas as pd

def load_data(file):
    """
    Load data from CSV file.
    
    Args:
        file: File object or path.
        
    Returns:
        pd.DataFrame: Loaded data with parsed dates.
    """
    try:
        df = pd.read_csv(file)
        if 'date' not in df.columns or 'metric_value' not in df.columns:
            raise ValueError("CSV must contain 'date' and 'metric_value' columns.")
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

def split_data(df, event_date):
    """
    Split data into pre-event and post-event windows.
    
    Args:
        df: pd.DataFrame with 'date' column.
        event_date: str or datetime, date of the event.
        
    Returns:
        tuple: (pre_event_df, post_event_df)
    """
    event_date = pd.to_datetime(event_date)
    pre_event = df[df['date'] < event_date]
    post_event = df[df['date'] >= event_date]
    return pre_event, post_event
