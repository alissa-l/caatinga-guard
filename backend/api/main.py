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



app = FastAPI(title="Previsor de Incendios RN", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def raiz():
    return {"status": "ok", "endpoints": [
        "/municipios", "/municipios/geojson",
        "/previsao/{data}", "/previsao/{data}/comparacao",
        "/previsao/simulacao (POST)", "/previsao/futuro?dias=N",
        "/modelo/relatorio", "/datas",
    ]}
