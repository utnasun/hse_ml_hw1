from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from pathlib import Path

import pandas as pd
import numpy as np
import pickle
import re
import io


PATH_MODELS = Path.cwd() / "models"


app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int | None
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

def load_pickle(filename: str):
    """Load .pkl model"""

    with open(filename, "rb") as file:
        return pickle.load(file)
    

def extract_first_word(text):
    """Extract first word of string """
    pattern = re.compile(r"^\w+")
    match = pattern.search(text)
    if match:
        return match.group()  
    return None 


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """ Function for preprocessing data

    Args:
        data (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: dataframe with correct features
    """
    imputer = load_pickle((PATH_MODELS / "imputer.pkl"))
    ohe = load_pickle((PATH_MODELS / "ohe.pkl"))
    poly = load_pickle((PATH_MODELS / "poly.pkl"))
    std_scaler = load_pickle((PATH_MODELS / "scaler.pkl"))

    # Убираем torque и selling_price
    data.drop(columns = ['selling_price', 'torque'], inplace=True)

    # Исключаем измерения и заменяем пустые значения
    units = ['mileage', 'engine', 'max_power']
    data[units] = data[units].apply(lambda x: x.str.replace(r'[^0-9.]', '', regex=True))
    data[units] = data[units].replace('', np.nan)
    data[units] = data[units].astype(float)
    data[units] = pd.DataFrame(imputer.transform(data[units]), 
                                             columns=units)
    
    # Добавляем полиномиальные признаки числовых признаков, кроме seat 
    poly_data = pd.DataFrame(poly.transform(data[['year', 'km_driven', 'mileage', 'engine', 'max_power']]),
                columns=poly.get_feature_names_out())
    
    data = pd.concat([data.drop(['year', 'km_driven', 'mileage', 'engine', 'max_power'], axis=1), poly_data], axis=1)

    # Преобразуем название в марку
    data['name'] = data['name'].apply(lambda x: extract_first_word(x))

    # Добавим категориальные признаки
    cat_features = ['fuel', 'seller_type', 'transmission', 'owner', 'seats', 'name']
    ohe_df = pd.DataFrame(ohe.transform(data[cat_features]).toarray(), columns=ohe.get_feature_names_out())
    data = pd.concat([data.drop(columns=cat_features), ohe_df], axis=1)

    data.drop(columns=['1'], inplace=True)

    # Стандартизируем числовые признаки
    numeric_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'year^2',
       'year km_driven', 'year mileage', 'year engine', 'year max_power',
       'km_driven^2', 'km_driven mileage', 'km_driven engine',
       'km_driven max_power', 'mileage^2', 'mileage engine',
       'mileage max_power', 'engine^2', 'engine max_power', 'max_power^2']
    
    data[numeric_columns] = pd.DataFrame(std_scaler.transform(data[numeric_columns]), columns=numeric_columns)

    return data 


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """ Function for predicting car price """
    model = load_pickle((PATH_MODELS / "model.pkl"))

    df = pd.DataFrame.from_dict(pd.json_normalize(item.dict()))
    df = preprocess_data(df)

    return model.predict(df)

@app.post("/predict_items")
async def predict_items(file: UploadFile) -> StreamingResponse:
    """Returns .csv file with car price predictions from input .csv"""

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    model = load_pickle((PATH_MODELS / "model.pkl"))

    df['predicted_price'] = model.predict(preprocess_data(df))

    # Сохранение данных с предсказаниями в .csv файл для передачи обратно клиенту
    csv_data = df.to_csv(index=False)
    return StreamingResponse(io.StringIO(csv_data), media_type="text/csv", headers={
        "Content-Disposition": f"attachment; filename=predictions.csv"
    })