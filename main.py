import csv

import pandas as pd
import uvicorn
from fastapi import FastAPI, Path
from service.pca_service import PCAService
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/convert")
async def csv_convert():
    xlsx = pd.read_excel('data/convert.xlsx')
    csv_data = xlsx.to_csv('data/embed_test_data2.csv', index=False, quoting=csv.QUOTE_ALL)
    return {"message": "convert"}


@app.get("/plot/2d")
async def get_pca():
    p = PCAService()
    p.visualize_2d()
    return {"message": "Hello World"}

@app.get("/plot/3d")
async def get_pca():
    p = PCAService()
    p.visualize_3d()
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)