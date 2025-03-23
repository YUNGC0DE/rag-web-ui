from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import requests
from requests.exceptions import RequestException
import time

from app.core.config import settings
from app.db.session import get_db
from app.schemas.token import Token


router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

token_cache = {
    # структура: {token: {"id": user_id, "timestamp": time_when_cached}}
}

def get_current_user(
    token: str = Depends(oauth2_scheme)
) -> int:
    # Проверяем, есть ли токен в кеше и не устарел ли он (1 час = 3600 секунд)
    current_time = time.time()
    if token in token_cache and current_time - token_cache[token]["timestamp"] < 3600:
        return token_cache[token]["id"]
    
    # Если токена нет в кеше или он устарел, делаем запрос к API
    response = requests.get(
        "http://dev.api.aigenda.tech/api/users/me/",
        headers={"Authorization": f"Bearer {token}"}
    )
    user_id = int(response.json()["id"])
    
    # Сохраняем результат в кеше
    token_cache[token] = {"id": user_id, "timestamp": current_time}
    
    return user_id

@router.post("/token", response_model=Token)
def login_access_token(
   form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    response = requests.post(
        "http://dev.api.aigenda.tech/api/token/",
        json={"username": form_data.username, "password": form_data.password}
    )
    return {"access_token": response.json()["access"], "token_type": "bearer"}
