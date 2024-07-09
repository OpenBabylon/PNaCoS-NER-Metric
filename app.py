from fastapi import FastAPI
from formats import TextsInputs, Output
from loaders import load_metric

app = FastAPI()

metric = load_metric()


@app.post("/calculate/", response_model=Output)
async def calculate_metric(input: TextsInputs):
    global metric
    pred_dict = metric.calculate(texts=input.texts)
    return {"report": pred_dict}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)