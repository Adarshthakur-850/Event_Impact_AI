import pandas as pd
import statsmodels.api as sm

def fit_regression(df, event_date):
    """
    Fit a regression model with an event dummy variable.
    Model: metric ~ constant + time_trend + event_dummy
    
    Args:
        df: pd.DataFrame with 'date' and 'metric_value'.
        event_date: str or datetime.
        
    Returns:
        dict: {'summary': str, 'event_coef': float, 'p_value': float, 'conf_int': list}
    """
    event_date = pd.to_datetime(event_date)
    
    df = df.copy()
    df['time_idx'] = range(len(df))
    df['event_dummy'] = (df['date'] >= event_date).astype(int)
    
    X = df[['time_idx', 'event_dummy']]
    X = sm.add_constant(X)
    y = df['metric_value']
    
    model = sm.OLS(y, X).fit()
    
    return {
        'summary': model.summary().as_text(),
        'event_coef': model.params.get('event_dummy', 0),
        'p_value': model.pvalues.get('event_dummy', 1),
        'conf_int': model.conf_int().loc['event_dummy'].tolist() if 'event_dummy' in model.params else [0, 0]
    }
