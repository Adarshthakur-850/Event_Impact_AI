from scipy import stats
import pandas as pd

def calculate_ttest(pre_series, post_series):
    """
    Perform independent t-test between pre and post event metrics.
    
    Args:
        pre_series: pd.Series, metric values before event.
        post_series: pd.Series, metric values after event.
        
    Returns:
        dict: {'t_statistic': float, 'p_value': float, 'pre_mean': float, 'post_mean': float}
    """
    t_stat, p_val = stats.ttest_ind(pre_series, post_series, equal_var=False)
    
    return {
        't_statistic': t_stat,
        'p_value': p_val,
        'pre_mean': pre_series.mean(),
        'post_mean': post_series.mean()
    }
