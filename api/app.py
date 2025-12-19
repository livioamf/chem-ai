
from http import HTTPStatus

from fastapi import FastAPI
from contextlib import asynccontextmanager

from api.database import engine
from api.models import table_registry
from api.routers import auth, predictions, users
from api.schemas import Message

from api.models import User
from api.schemas import UserType
from api.database import session_context
from api.security import get_password_hash

from api.routers.predictions import get_predictor
from sqlalchemy import select

import logging
logger = logging.getLogger("uvicorn") 

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(table_registry.metadata.create_all)

    async with session_context() as session:
        existing_admin = await session.scalar(
            select(User).where(User.username == "admin")
        )
        if not existing_admin:
            admin = User(
                name="Administrador do Sistema",
                username="admin",
                type=UserType.admin,
                password=get_password_hash("admin123"),
            )
            session.add(admin)
            await session.commit()
            logger.info("Usuário admin padrão criado (username='admin', senha='admin123')")
        else:
            logger.info("Usuário admin já existe")

    logger.info('Banco iniciado com sucesso!')

    try:
        logger.info("Inicializando cache de preditores ChemBERT...")
        modelos = [
            ("pure", "base"),
            ("pure", "lora"),
            ("mix", "base"),
            ("mix", "lora"),
        ]
        for mode, arch in modelos:
            get_predictor(mode, arch)
        logger.info("Cache de preditores inicializado com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao inicializar cache de preditores: {e}")

    yield
    await engine.dispose()
    logger.info('Banco finalizado e conexões fechadas.')

app = FastAPI(lifespan=lifespan)


app.include_router(users.router)
app.include_router(auth.router)
app.include_router(predictions.router)

@app.get('/', status_code=HTTPStatus.OK, response_model=Message)
async def read_root():
    return {'message': 'Olá, Mundo!'}
