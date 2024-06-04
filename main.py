from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np


app = FastAPI()

@app.get("/")
def root(Request: request):
    return {"Hello": "Working!"}


@app.get("/api/{entidade}")
def predict(entidade):

    ai_model = pickle.load(open("../model.pkl", "rb"))

    file = pd.read_csv("./dados.csv")
    
    entidade_especifica = entidade

    filtro = file.loc[file['Entidade'] == entidade_especifica]

    filtro_dict = filtro.to_dict(orient='records')

    X = [
        filtro_dict[0].get("Lixo plástico mal gerenciado por pessoa (kg por ano)"),
        filtro_dict[0].get("Participação na emissão global de plásticos para o oceano"),
        filtro_dict[0].get("Qualidade do Ar")
    ]
    
    filtro = filtro.drop(columns=["Entidade","Ano", "poluicao_da_agua"])

    print(filtro)

    prediction = ai_model.predict(filtro)


    match prediction.tolist():
        case [0]:
            prediction = "Extremamente sujo"
        case [1]:
            prediction = "Limpo"
        case [2]:
            prediction = "Muito sujo"
        case [3]:
            prediction = "Sujo"

    return {"prediction": prediction, "filtered_data": filtro.to_dict(orient='records')}