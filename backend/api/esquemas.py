from pydantic import BaseModel
from typing import Optional


class Municipio(BaseModel):
    codigo_ibge: str
    nome: str
    uf: str
    centro_lat: float
    centro_lon: float
    area_km2: float

class PrevisaoDia(BaseModel):
    codigo_ibge: str
    probabilidade: float

class PrevisaoDia(BaseModel):
    data: str            
    data_alvo: str       
    modelo: str
    previsoes: list[PrevisaoMunicipio]


class ComparacaoMunicipio(BaseModel):
    codigo_ibge: str
    probabilidade: float
    teve_foco: int


class ComparacaoDia(BaseModel):
    data: str
    data_alvo: str
    modelo: str
    previsoes: list[ComparacaoMunicipio]
    acerto: dict


class Ajustes(BaseModel):
    temperatura: Optional[float] = 0      
    umidade: Optional[float] = 0          
    precipitacao: Optional[float] = 1     
    vento: Optional[float] = 1            


class PedidoSimulacao(BaseModel):
    codigo_ibge: str
    data_base: str
    ajustes: Ajustes


class RespostaSimulacao(BaseModel):
    codigo_ibge: str
    data_base: str
    probabilidade_original: float
    probabilidade_simulada: float
    ajustes_aplicados: dict


class PrevisaoFuturoDia(BaseModel):
    data_alvo: str
    previsoes: list[PrevisaoMunicipio]


class PrevisaoFuturo(BaseModel):
    data_base: str
    n_dias: int
    observacao: str
    dias: list[PrevisaoFuturoDia]
