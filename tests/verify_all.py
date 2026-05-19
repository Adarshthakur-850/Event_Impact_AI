import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis import preprocess, ttest, intervention, regression_model

def create_dummy_data():
    dates = pd.date_range(start='2024-01-01', periods=100)
    # create a jump at index 50
    values = np.random.normal(loc=10, scale=2, size=50).tolist() + \
             np.random.normal(loc=20, scale=2, size=50).tolist()
    
    df = pd.DataFrame({'date': dates, 'metric_value': values})
    csv_path = 'test_impact.csv'
    df.to_csv(csv_path, index=False)
    return csv_path, dates[50]

def test_analysis():
    print("Testing analysis modules...")
    csv_path, event_date = create_dummy_data()
    event_date_str = str(event_date.date())
    
    try:
        # Test Preprocess
        df = preprocess.load_data(csv_path)
        pre, post = preprocess.split_data(df, event_date_str)
        assert len(pre) == 50
        assert len(post) == 50
        print("Preprocess passed.")
        
        # Test T-Test
        ttest_res = ttest.calculate_ttest(pre['metric_value'], post['metric_value'])
        assert ttest_res['p_value'] < 0.05
        assert ttest_res['post_mean'] > ttest_res['pre_mean']
        print("T-test passed.")
        
        # Test ITS
        its_res = intervention.interrupted_time_series(df, event_date_str)
        assert its_res['impact'] > 5
        print("ITS passed.")
        
        # Test Regression
        reg_res = regression_model.fit_regression(df, event_date_str)
        assert reg_res['event_coef'] > 8
        print("Regression passed.")
        
        print("All analysis tests passed successfully!")
    
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

if __name__ == "__main__":
    test_analysis()
