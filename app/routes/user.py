import jwt
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

from app.models.user_model import Login, ResponseLogin, ResponseRegister, User
from app.services.user_services import (
    get_current_user,
    hash_password,
    oauth2_scheme,
    verify_password,
)
from config import create_access_token
from database import users_collection

router = APIRouter(prefix="/user", tags=["User"])


@router.get("/get_me")
async def get_current_user_route(token: str = Depends(oauth2_scheme)):
    try:
        user = await get_current_user(token)
        print(user)
        user_db = await users_collection.find_one(
            {"_id": ObjectId(user["id"])},
            {"_id": 1, "email": 1, "name": 1, "surname": 1},
        )
        user_db["_id"] = str(user_db["_id"])
        """ user_db["success"] = True """
        return {"success": True, **user_db}
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Expired Token")


@router.post("/", response_model=User)
async def create_user(user: User):
    user_dict = user.dump()
    print(user_dict)
    result = await users_collection.insert_one(user_dict)
    user_dict["_id"] = str(result.inserted_id)
    return user_dict


@router.post("/login", response_model=ResponseLogin)
async def login(user: Login) -> ResponseLogin:
    user_db: User = await users_collection.find_one({"email": user.email})
    if not user_db:
        raise HTTPException(status_code=404, detail="Wrong email or password!")
    if verify_password(user.password, user_db["password"]):
        access_token = create_access_token(
            data={"sub": user.email, "id": str(user_db["_id"])}
        )
        return {"success": True, "accessToken": access_token}
    else:
        raise HTTPException(status_code=500, detail="Wrong email or password!")


@router.post("/register", response_model=ResponseRegister)
async def register_user(user: User) -> ResponseRegister:
    user_db = await users_collection.find_one({"email": user.email})
    if user_db:
        raise HTTPException(
            status_code=409, detail="User with this email already exists!"
        )
    hashed_password = hash_password(user.password)
    await users_collection.insert_one(
        {
            "name": user.name,
            "surname": user.surname,
            "age": user.age,
            "email": user.email,
            "password": hashed_password,
        }
    )
    return {
        "success": True,
        "name": user.name,
        "surname": user.surname,
        "age": user.age,
        "email": user.email,
        "password": hashed_password,
    }


@router.get("/{user_id}", response_model=User)
async def get_user(user_id: str):
    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user["_id"] = str(user["_id"])
    return user
