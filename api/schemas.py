from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.models import (
    UserType,
)


class Message(BaseModel):
    message: str

class UserNew(BaseModel):
    name: str
    username: str
    type: UserType
    model_config = ConfigDict(from_attributes=True)


class UserPublic(UserNew):
    id: int

class UserSchema(UserNew):
    password: str


class UserList(BaseModel):
    users: list[UserPublic]


class Token(BaseModel):
    access_token: str
    token_type: str


class FilterPage(BaseModel):
    offset: int = Field(0, ge=0)
    limit: int = Field(100, ge=1)


class ViscosityPublic(BaseModel):
    smile_1: str = Field(..., description='SMILES do primeiro componente')
    smile_2: Optional[str] = Field(
        None, description='SMILES do segundo componente (opcional)'
    )
    fraction: Optional[float] = Field(
        None, description='Fração do primeiro componente da mistura'
    )
    temperature: float = Field(..., description='Temperatura em Kelvin')

    @model_validator(mode='after')
    def validate_dependencies(self):
        if self.smile_2 is None and self.fraction is not None:
            raise ValueError('Fração só deve ser informada para misturas.')
        if self.smile_2 is not None and self.fraction is None:
            raise ValueError('Fração deve ser informada para misturas.')
        return self


class ViscosityPrediction(BaseModel):
    viscosity: float


class ViscosityBatchRequest(BaseModel):
    inputs: List[ViscosityPublic]


class ViscosityBatchResponse(BaseModel):
    predictions: List[ViscosityPrediction]


class ServiceTokenRequest(BaseModel):
    client_id: str
    client_secret: str
