import asyncio

import logging
from pathlib import Path

from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_session
from api.schemas import (
    ViscosityBatchRequest,
    ViscosityBatchResponse,
    ViscosityPrediction,
    ViscosityPublic,
)
from api.security import AuthSubject, get_current_user

from chemai.predictor import ChemBERTPredictor
from proxy import configure_proxy

configure_proxy()

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/predictions', tags=['predictions'])

Session = Annotated[AsyncSession, Depends(get_session)]
CurrentAuth = Annotated[AuthSubject, Depends(get_current_user)]

_PREDICTOR_CACHE: dict[str, ChemBERTPredictor] = {}


def get_predictor(mode: str, architecture: str) -> ChemBERTPredictor:

    MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
    key = f"{mode}_{architecture}"
    try:
        if key not in _PREDICTOR_CACHE:
            base_dir = Path(__file__).resolve().parents[2]  
            model_path = base_dir / "models" / "torch" / key

            if not model_path.exists():
                raise HTTPException(
                    status_code=HTTPStatus.NOT_FOUND,
                    detail=f"Diretório do modelo não encontrado: {model_path}"
                )

            logger.info(f"Carregando modelo {key}...")
            _PREDICTOR_CACHE[key] = ChemBERTPredictor(
                mode=mode, model_dir=str(model_path), hf_model_name=MODEL_NAME)
            logger.info(f"Modelo {key} carregado com sucesso.")
        return _PREDICTOR_CACHE[key]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro ao carregar modelo: {e}"
        )


@router.post(
    '/viscosity',
    summary='Predição de viscosidade de mistura molecular',
)
async def predict_viscosity(
    data: ViscosityPublic,
    session: Session,
    current_auth: CurrentAuth,
    architecture: str = Query(..., description="Arquitetura do modelo ('base' ou 'lora')"),
):
    try:
        viscosity_value = await calcular_viscosidade(
            smile_1=data.smile_1,
            smile_2=data.smile_2,
            fraction=data.fraction,
            temperature=data.temperature,
            architecture=architecture
        )
        return {'viscosity': viscosity_value}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro ao calcular viscosidade: {e}"
        )

@router.post(
    '/viscosity/batch',
    response_model=ViscosityBatchResponse,
    summary='Predição em lote de viscosidade',
)
async def predict_viscosity_batch(
    batch: ViscosityBatchRequest,
    current_auth: CurrentAuth,
    architecture: str = Query(..., description="Arquitetura do modelo ('base' ou 'lora')"),

):
    if isinstance(current_auth, dict) and current_auth.get('is_service'):
        requester = f"Serviço: {current_auth['service_name']}"
    else:
        requester = f"Usuário: {current_auth.username}"
    logger.info('Predição em lote solicitada por %s', requester)
    try:
        smiles1_list = [d.smile_1 for d in batch.inputs]
        smiles2_list = [d.smile_2 for d in batch.inputs]
        fractions_list = [d.fraction for d in batch.inputs]
        temps_list = [d.temperature for d in batch.inputs]
        tamanhos = {len(smiles1_list), len(smiles2_list), len(fractions_list), len(temps_list)}
        if len(tamanhos) != 1:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Todas as listas de entrada devem ter o mesmo tamanho."
            )
        primeiro = batch.inputs[0]
        if primeiro.smile_1 and primeiro.temperature and primeiro.smile_2 is None and primeiro.fraction is None:
            mode = "pure"
        elif all(v is not None for v in [primeiro.smile_1, primeiro.smile_2, primeiro.fraction, primeiro.temperature]):
            mode = "mix"
        else:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Combinação de entrada inválida no lote: todas as entradas devem seguir o padrão 'pure' ou 'mix'."
            )
        predictor = get_predictor(mode, architecture)
        if mode == "pure":
            viscosidades = predictor.predict(smiles1=smiles1_list, temp=temps_list)
        else:
            viscosidades = predictor.predict(
                smiles1=smiles1_list,
                smiles2=smiles2_list,
                frac=fractions_list,
                temp=temps_list
            )
        resultados = [ViscosityPrediction(viscosity=float(v)) for v in viscosidades]
        return ViscosityBatchResponse(predictions=resultados)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro ao calcular predições em lote: {e}"
        )

@router.get('/status', status_code=HTTPStatus.OK, summary='Status do serviço')
async def get_prediction_status(current_auth: CurrentAuth):
    if isinstance(current_auth, dict) and current_auth.get('is_service'):
        requester = f"Serviço: {current_auth['service_name']}"
    else:
        requester = f"Usuário: {current_auth.username}"
    return {
        'status': 'ok',
        'message': f"Predições disponíveis para {requester}",
        'models_loaded': list(_PREDICTOR_CACHE.keys())
    }

async def calcular_viscosidade(
    smile_1: str,
    smile_2: str | None,
    fraction: float | None,
    temperature: float | None,
    architecture: str
) -> float:
    try:
        await asyncio.sleep(0)
        if architecture not in ('base', 'lora'):
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail=f"Arquitetura inválida '{architecture}'. Deve ser 'base' ou 'lora'."
            )
        if smile_1 and temperature and smile_2 is None and fraction is None:
            mode = "pure"
        elif all(v is not None for v in [smile_1, smile_2, fraction, temperature]):
            mode = "mix"
        else:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Parâmetros inválidos: para 'pure', informe apenas smile_1 e temperature. Para 'mix', informe todos (smile_1, smile_2, fraction, temperature)."
            )
        predictor = get_predictor(mode, architecture)
        if mode == "pure":
            viscosidade = predictor.predict(smiles1=smile_1, temp=temperature)
        else:
            viscosidade = predictor.predict(smiles1=smile_1, smiles2=smile_2, frac=fraction, temp=temperature)
        return float(viscosidade)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Erro durante a predição: {e}"
        )
