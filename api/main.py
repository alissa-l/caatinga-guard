from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.estado import estado
from backend.api.rotas import router
from backend.modelo import prever

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("carregando dataset e geojson...")
    estado.carregar()
    print(f"  {len(estado.df)} linhas, {len(estado.municipios)} municipios")
    print("carregando modelos...")
    prever.carregar("random_forest")
    try:
        prever.carregar("lightgbm")
    except FileNotFoundError:
        pass
    print("API pronta")
    yield