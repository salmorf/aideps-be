import os
from PIL import Image
import io
from fastapi import APIRouter, Body, Depends, HTTPException
from pathlib import Path
from app.models.ml_models import InputModelUserData
from app.services.ml_services import (  # type: ignore
    executeMLModelV2,
    load_model,
    preprocess_df_for_catboost_from_body,
)
from app.services.user_services import get_current_user
from fastapi import UploadFile, File, Form
import cv2
import numpy as np
import base64
from app.services.yolo_services import MedicalYOLOInference


router = APIRouter(prefix="/ml", tags=["Modello IA"])


@router.post(
    "/execute",
    summary="Predice mastopessi per ogni record Excell e calcola accuratezza",
)
async def predict_one_v2(
    token: str = Depends(get_current_user),
    input_array: InputModelUserData = Body(..., embed=True),
):
    model_type = "catboost"
    model = await load_model(model_type)
    # dataframe = await preprocess_input_to_execute(input_array)
    dataframe = await preprocess_df_for_catboost_from_body(input_array)
    predict = await executeMLModelV2(dataframe, model, model_type)
    return {
        "result": predict,
    }


@router.post("/annotations")
async def save_ann(payload: dict = Body(...)):
    image_name = payload["image_name"]
    RAW_LABELS_DIR = Path("./app/raw_labels")
    txt_path = RAW_LABELS_DIR / f"{Path(image_name).stem}.txt"
    with txt_path.open("w") as f:
        for ann in payload["annotations"]:
            class_id = ann["class_id"]
            x_center = ann["x_center"]
            y_center = ann["y_center"]
            box_width = ann["box_width"]
            box_height = ann["box_height"]
            f.write(
                f"{class_id} { x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"
            )
    return {"status": "ok", "file": str(txt_path)}


@router.post("/analyze/yolo")
async def analyze_yolo_image(
    image_file: UploadFile = File(...),
    confidence: float = Form(0.5),
    anatomical_correction: bool = Form(True),
    token: str = Depends(get_current_user),
):
    try:
        model_path = os.path.join("app", "ml_models", "best.pt")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail="Modello best.pt non trovato")

        contents = await image_file.read()

        image_pil = Image.open(io.BytesIO(contents))
        dpi = image_pil.info.get("dpi", (72, 72))  # default DPI se mancante
        dpi_x, dpi_y = dpi

        np_arr = np.frombuffer(contents, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        engine = MedicalYOLOInference()
        engine.anatomical_correction_enabled = anatomical_correction
        output_image, report = engine.analyze_image(image_np, confidence, model_path)

        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        landmarks = {d["class"]: d["bbox"] for d in engine.last_detections}

        def bbox_center(bbox):
            x1, y1, x2, y2 = bbox
            return ((x1 + x2) / 2, (y1 + y2) / 2)

        def pixel_distance(p1, p2):
            return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        def px_to_cm(px):
            return (px / dpi_y) * 2.54

        def round_cm(x: float) -> int:
            return round(x)

        print("landmarks; ", landmarks)
        measurements = {}
        try:
            if "left_jugular" in landmarks and "left_nipple" in landmarks:
                dist_px = pixel_distance(
                    bbox_center(landmarks["left_jugular"]),
                    bbox_center(landmarks["left_nipple"]),
                )
                measurements["distanza_giugulo_sx"] = round_cm(px_to_cm(dist_px))

            if "right_jugular" in landmarks and "right_nipple" in landmarks:
                dist_px = pixel_distance(
                    bbox_center(landmarks["right_jugular"]),
                    bbox_center(landmarks["right_nipple"]),
                )
                measurements["distanza_giugulo_dx"] = round_cm(px_to_cm(dist_px))

            if "left_areola" in landmarks:
                x1, y1, x2, y2 = landmarks["left_areola"]
                diameter_px = (x2 - x1 + y2 - y1) / 2
                measurements["diametro_areola_sx"] = round_cm(px_to_cm(diameter_px))

            if "right_areola" in landmarks:
                x1, y1, x2, y2 = landmarks["right_areola"]
                diameter_px = (x2 - x1 + y2 - y1) / 2
                measurements["diametro_areola_dx"] = round_cm(px_to_cm(diameter_px))

            if "left_areola" in landmarks and "left_breast" in landmarks:
                dist_px = pixel_distance(
                    bbox_center(landmarks["left_areola"]),
                    bbox_center(landmarks["left_breast"]),
                )
                measurements["distanza_areola_sx"] = round_cm(px_to_cm(dist_px))

            if "right_areola" in landmarks and "right_breast" in landmarks:
                dist_px = pixel_distance(
                    bbox_center(landmarks["right_areola"]),
                    bbox_center(landmarks["right_breast"]),
                )
                measurements["distanza_areola_dx"] = round_cm(px_to_cm(dist_px))

        except Exception as m_err:
            print("Errore nel calcolo misure:", m_err)
        expected_keys = [
            "distanza_giugulo_sx",
            "distanza_giugulo_dx",
            "diametro_areola_sx",
            "diametro_areola_dx",
            "distanza_areola_sx",
            "distanza_areola_dx",
        ]
        normalized_measurements = {key: "" for key in expected_keys}
        normalized_measurements.update(measurements)
        return {
            "success": True,
            "report": report,
            "image_base64": encoded_image,
            "measurements": normalized_measurements,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Errore durante l'inferenza: {str(e)}"
        )
