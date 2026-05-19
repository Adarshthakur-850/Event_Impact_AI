from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pandas as pd
import io
import sys
import os

# Add parent directory to path to allow importing analysis module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis import preprocess, ttest, intervention, regression_model

app = FastAPI(title="Event Impact Model API")

@app.post("/analyze")
async def analyze_impact(
    event_date: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Preprocessing
        df = preprocess.load_data(io.BytesIO(content))
        pre_event, post_event = preprocess.split_data(df, event_date)
        
        if pre_event.empty or post_event.empty:
            raise HTTPException(status_code=400, detail="Data must cover both pre and post event periods.")
            
        # Analysis
        ttest_results = ttest.calculate_ttest(pre_event['metric_value'], post_event['metric_value'])
        its_results = intervention.interrupted_time_series(df, event_date)
        regression_results = regression_model.fit_regression(df, event_date)
        
        return {
            "event_date": event_date,
            "ttest": ttest_results,
            "its": its_results,
            "regression": regression_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
