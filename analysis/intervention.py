import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def interrupted_time_series(df, event_date):
    """
    Perform Interrupted Time Series (ITS) analysis.
    Fits a linear trend to pre-event data and projects it to post-event.
    
    Args:
        df: pd.DataFrame with 'date' and 'metric_value'.
        event_date: str or datetime.
        
    Returns:
        dict: {
            'expected_post_mean': float,
            'actual_post_mean': float,
            'impact': float,
            'impact_pct': float,
            'model': sklearn model object (for plotting if needed),
            'predictions': pd.Series (full range predictions based on pre-trend)
        }
    """
    event_date = pd.to_datetime(event_date)
    df['time_idx'] = np.arange(len(df))
    
    pre_data = df[df['date'] < event_date]
    post_data = df[df['date'] >= event_date]
    
    if pre_data.empty or post_data.empty:
        raise ValueError("Insufficient data for ITS analysis (need both pre and post data).")
    
    # Fit trend on pre-event data
    X_pre = pre_data[['time_idx']]
    y_pre = pre_data['metric_value']
    
    model = LinearRegression()
    model.fit(X_pre, y_pre)
    
    # Predict for the whole range to see counterfactual
    X_full = df[['time_idx']]
    df['counterfactual'] = model.predict(X_full)
    
    # Calculate impact on post-event data
    post_predictions = df.loc[df['date'] >= event_date, 'counterfactual']
    actual_post_mean = post_data['metric_value'].mean()
    expected_post_mean = post_predictions.mean()
    
    impact = actual_post_mean - expected_post_mean
    impact_pct = (impact / expected_post_mean) * 100 if expected_post_mean != 0 else 0
    
    return {
        'expected_post_mean': expected_post_mean,
        'actual_post_mean': actual_post_mean,
        'impact': impact,
        'impact_pct': impact_pct,
        'predictions': df['counterfactual'].tolist()
    }
