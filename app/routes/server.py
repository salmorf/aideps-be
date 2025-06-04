from fastapi import APIRouter

router = APIRouter(prefix="/server", tags=["Server"])


@router.get("/health")
def get_server_health():
    return {"success": 200, "messagge": "Il server gode di buona salute!"}
