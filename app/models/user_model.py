from pydantic import BaseModel


class User(BaseModel):
    name: str
    surname: str
    email: str
    age: int
    password: str


class Login(BaseModel):
    email: str
    password: str


class ResponseLogin(BaseModel):
    success: bool
    accessToken: str


class ResponseRegister(BaseModel):
    success: bool
    name: str
    surname: str
    email: str
    age: int
    password: str
