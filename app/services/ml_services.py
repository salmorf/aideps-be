import os
import re
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from typing import Dict, Any, List
from app.models.ml_models import InputModelUserData


async def load_model(type):
    models_path = "./app/ml_models/"
    if "catboost" == type:
        model = joblib.load(models_path + "best_model_clinical_data.joblib")
        """ model = CatBoostClassifier()
        model.load_model(models_path + "best_model_clinical_data.joblib") """
    elif "image" == type:
        model = joblib.load(models_path + "best_model_image.joblib")
    return model


async def executeMLModelV2(df_orig, model, type: str):
    if type == "catboost":
        # Prepara le colonne e i tipi corretti per CatBoost
        if isinstance(model, dict) and "model" in model:
            model_obj = model["model"]
            expected_cols = model.get("columns", list(df_orig.columns))
            cat_cols = model.get("cat_cols")
            if not cat_cols:
                # fallback al file salvato, se presente
                cat_cols_path = "./app/ml_models/catboost_cat_cols.joblib"
                if os.path.exists(cat_cols_path):
                    try:
                        cat_cols = joblib.load(cat_cols_path)
                    except Exception:
                        cat_cols = []
                else:
                    cat_cols = []
        else:
            model_obj = model
            expected_cols = list(df_orig.columns)
            cat_cols = []

        X = df_orig.copy()
        # Mantieni solo le colonne attese, nell'ordine atteso
        cols_in_both = [c for c in expected_cols if c in X.columns]
        X = X[cols_in_both]
        # Converte le categoriche in stringa come richiesto da CatBoost
        for c in cat_cols or []:
            if c in X.columns:
                X[c] = X[c].astype(str)
        # Se disponibile, forza i tipi numerici per le feature numeriche
        if isinstance(model, dict):
            num_cols = model.get("num_cols", [])
            for c in num_cols:
                if c in X.columns:
                    X[c] = pd.to_numeric(X[c], errors="coerce")

        return await predict(model_obj, X, cat_cols or [])
    else:
        raise ValueError("Unsupported model type. Only 'catboost' is supported here.")


async def predict(model, X, cat_cols):
    if len(cat_cols) != 0:
        if hasattr(model, "predict_proba"):
            pool = Pool(data=X, cat_features=cat_cols)
            proba = model.predict_proba(pool)[:, 1]
            preds_num = (proba > 0.5).astype(int)
        else:
            preds_num = model.predict(X, cat_features=cat_cols)
        pred_value = int(preds_num[0])
        if pred_value == 1:
            return "T Invertita"
        elif pred_value == 0:
            return "Altro"
        else:
            return "altro"
    else:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
            preds_num = (proba > 0.5).astype(int)
        else:
            preds_num = model.predict(X)
        pred_value = int(preds_num[0])
        if pred_value == 1:
            return "T Invertita"
        elif pred_value == 0:
            return "Altro"
        else:
            return "altro"


async def predict_proba_catboost(
    model_bundle: Dict[str, Any], df_orig: pd.DataFrame
) -> float:
    """Return positive class probability for CatBoost model using saved metadata."""
    if isinstance(model_bundle, dict) and "model" in model_bundle:
        model_obj = model_bundle["model"]
        expected_cols = model_bundle.get("columns", list(df_orig.columns))
        cat_cols: List[str] = model_bundle.get("cat_cols", [])
        num_cols: List[str] = model_bundle.get("num_cols", [])
    else:
        model_obj = model_bundle
        expected_cols = list(df_orig.columns)
        cat_cols, num_cols = [], []

    X = df_orig.copy()
    cols_in_both = [c for c in expected_cols if c in X.columns]
    X = X[cols_in_both]
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str)
    for c in num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    if hasattr(model_obj, "predict_proba"):
        pool = Pool(data=X, cat_features=cat_cols) if len(cat_cols) else X
        proba = model_obj.predict_proba(pool)[:, 1]
        return float(proba[0])
    preds = (
        model_obj.predict(X, cat_features=cat_cols)
        if len(cat_cols)
        else model_obj.predict(X)
    )
    return float(preds[0])


def build_image_features_from_detections(
    detections: List[Dict[str, Any]], dpi_y: float
) -> Dict[str, float]:
    """Compute image features from YOLO detections; missing keys can be imputed later."""

    def area(b):
        x1, y1, x2, y2 = b
        return max(0, x2 - x1) * max(0, y2 - y1)

    def center(b):
        x1, y1, x2, y2 = b
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def aspect(b):
        x1, y1, x2, y2 = b
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        return w / h

    def diameter(b):
        x1, y1, x2, y2 = b
        return ((x2 - x1) + (y2 - y1)) / 2.0

    def px_to_cm(px):
        return (px / (dpi_y if dpi_y else 72.0)) * 2.54

    det_map = {d["class"]: d["bbox"] for d in detections}
    feats: Dict[str, float] = {}

    rb = det_map.get("right_breast")
    lb = det_map.get("left_breast")
    ra = det_map.get("right_areola")
    la = det_map.get("left_areola")
    rj = det_map.get("right_jugular")
    lj = det_map.get("left_jugular")
    rn = det_map.get("right_nipple")
    ln = det_map.get("left_nipple")

    if rb:
        feats["area_seno_destro"] = area(rb)
        cx, cy = center(rb)
        feats["x_center_seno_destro"] = cx
        feats["y_center_seno_destro"] = cy
        feats["aspect_seno_destro"] = aspect(rb)
    if lb:
        feats["area_seno_sinistro"] = area(lb)
        cx, cy = center(lb)
        feats["x_center_seno_sinistro"] = cx
        feats["y_center_seno_sinistro"] = cy
        feats["aspect_seno_sinistro"] = aspect(lb)
    if ra:
        feats["area_areola_destra"] = area(ra)
        cx, cy = center(ra)
        feats["x_center_areola_destra"] = cx
        feats["y_center_areola_destra"] = cy
        feats["aspect_areola_destra"] = aspect(ra)
        feats["diametro_areola_destra"] = diameter(ra)
    if la:
        feats["area_areola_sinistra"] = area(la)
        cx, cy = center(la)
        feats["x_center_areola_sinistra"] = cx
        feats["y_center_areola_sinistra"] = cy
        feats["aspect_areola_sinistra"] = aspect(la)
        feats["diametro_areola_sinistra"] = diameter(la)
    if rj:
        feats["area_giugulare_dx"] = area(rj)
    if lj:
        feats["area_giugulare_sx"] = area(lj)

    # Ratios and asymmetries
    if (
        "area_areola_destra" in feats
        and "area_seno_destro" in feats
        and feats["area_seno_destro"]
    ):
        feats["areola_to_breast_ratio_destra"] = (
            feats["area_areola_destra"] / feats["area_seno_destro"]
        )
    if (
        "area_areola_sinistra" in feats
        and "area_seno_sinistro" in feats
        and feats["area_seno_sinistro"]
    ):
        feats["areola_to_breast_ratio_sinistra"] = (
            feats["area_areola_sinistra"] / feats["area_seno_sinistro"]
        )
    if "area_seno_destro" in feats and "area_seno_sinistro" in feats:
        a_r, a_l = feats["area_seno_destro"], feats["area_seno_sinistro"]
        feats["area_asymmetry"] = (a_r - a_l) / (a_r + a_l + 1e-6)
    if "aspect_seno_destro" in feats and "aspect_seno_sinistro" in feats:
        ar, al = feats["aspect_seno_destro"], feats["aspect_seno_sinistro"]
        feats["aspect_asymmetry"] = (ar - al) / (ar + al + 1e-6)

    # Jugular to nipple distances (cm)
    def center_point(b):
        x1, y1, x2, y2 = b
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def euclid(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    if rj and rn:
        feats["giugulo_to_capezzolo_destra"] = px_to_cm(
            euclid(center_point(rj), center_point(rn))
        )
    if lj and ln:
        feats["giugulo_to_capezzolo_sinistra"] = px_to_cm(
            euclid(center_point(lj), center_point(ln))
        )

    return feats


def predict_proba_image_model(
    model_bundle: Dict[str, Any], features: Dict[str, float]
) -> float:
    """Reindex, impute medians, scale and return positive class probability."""
    cols: List[str] = model_bundle.get("columns", [])
    med: Dict[str, float] = model_bundle.get("medians", {})
    scaler = model_bundle.get("scaler")
    model = model_bundle.get("model")
    row = {c: features.get(c, np.nan) for c in cols}
    for c in cols:
        if pd.isna(row[c]) and c in med:
            row[c] = med[c]
    X = pd.DataFrame([row], columns=cols)
    X_f = scaler.transform(X) if scaler is not None else X.values
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_f)[:, 1]
        return float(proba[0])
    preds = model.predict(X_f)
    return float(preds[0])


async def preprocess_input_to_execute(input_data: InputModelUserData) -> pd.DataFrame:
    data = {
        "Età": input_data.eta,
        "Grado di ptosi /pseudoptosi": input_data.grado_ptosi,
        "Qualità della pelle (elasticità/lassità/eccesso cutaneo)": input_data.qualita_pelle,
        "Volume del seno (ipoplasia/normale/gigantomastia + volume stimato da rimuovere </>500)": input_data.volume_seno,
        "Desiderio della paziente (aumento di volume?cicatrici +/- visibili)": input_data.desiderio_paziente,
        "BMI": input_data.bmi,
        "FUMO": input_data.fumo,
        "DISTURBI COAGULAZIONE /INR": input_data.disturbi_coagulazione,
        "Distanza giugulo-capezzolo": f"dx: {input_data.distanza_giugulo_dx} cm; sn: {input_data.distanza_giugulo_sx} cm",
        "Diametro areola": f"dx: {input_data.diametro_areola_dx} cm; sn: {input_data.diametro_areola_sx} cm",
        "Distanza areola-solco": f"dx: {input_data.distanza_areola_dx} cm; sx: {input_data.distanza_areola_sx} cm",
    }
    return pd.DataFrame([data])


async def preprocess_df_for_tabpfn_keras_executing(original_df):
    colonne_da_rimuovere = [
        "Corrispondenza Foto",
        "Nome e Cognome",
        "Numero di telefono",
        "Pre-op",
        "Pre-op.1",
    ]
    df_cleaned = original_df.drop(
        columns=[col for col in colonne_da_rimuovere if col in original_df.columns]
    )
    df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.contains("^Unnamed")]
    df = df_cleaned
    df["ptosi_lvl"] = df["Grado di ptosi /pseudoptosi"].apply(map_ptosis)
    df["skin_quality"] = df[
        "Qualità della pelle (elasticità/lassità/eccesso cutaneo)"
    ].apply(map_skin)
    vol_cols = df[
        "Volume del seno (ipoplasia/normale/gigantomastia + volume stimato da rimuovere </>500)"
    ].apply(lambda x: map_volume(x))
    df["volume_cat"] = vol_cols.apply(lambda x: x[0])
    df["volume_gt500"] = vol_cols.apply(lambda x: x[1])
    df["desire_cat"] = df[
        "Desiderio della paziente (aumento di volume?cicatrici +/- visibili)"
    ].apply(map_desire)
    df = pd.get_dummies(df, columns=["desire_cat"], prefix="desire", dummy_na=True)
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df["smoker"] = (
        df["FUMO"].str.contains("si|yes|/die|1", case=False, na=False).astype(int)
    )
    df["coag_disorder"] = (
        df["DISTURBI COAGULAZIONE /INR"]
        .str.contains("si|yes|posit", case=False, na=False)
        .astype(int)
    )
    for col, newcol in [
        ("Distanza giugulo-capezzolo", "jugulo_nipple_cm"),
        ("Diametro areola", "areola_diam_cm"),
        ("Distanza areola-solco", "areola_fold_cm"),
    ]:
        df[newcol] = df[col].apply(
            lambda x: (
                (sum(extract_float_list(x)) / len(extract_float_list(x)))
                if extract_float_list(x)
                else None
            )
        )
    df["age_bucket"] = pd.cut(
        df["Età"], bins=[0, 35, 50, float("inf")], labels=[0, 1, 2], right=False
    )
    df["bmi_bucket"] = pd.cut(
        df["BMI"], bins=[0, 30, 35, float("inf")], labels=[0, 1, 2], right=False
    )
    df["fold_dist_bucket"] = pd.cut(
        df["areola_fold_cm"],
        bins=[0, 7, 10, float("inf")],
        labels=[0, 1, 2],
        right=False,
    )
    df["areola_diam_big"] = (df["areola_diam_cm"] > 5).astype(int)
    df["jugulo_bucket"] = pd.cut(
        df["jugulo_nipple_cm"],
        bins=[0, 7, 10, float("inf")],
        labels=[0, 1, 2],
        right=False,
    )
    cols_to_drop = [
        "Grado di ptosi /pseudoptosi",
        "Qualità della pelle (elasticità/lassità/eccesso cutaneo)",
        "Volume del seno (ipoplasia/normale/gigantomastia + volume stimato da rimuovere </>500)",
        "Desiderio della paziente (aumento di volume?cicatrici +/- visibili)",
        "FUMO",
        "DISTURBI COAGULAZIONE /INR",
        "Distanza giugulo-capezzolo",
        "Diametro areola",
        "Distanza areola-solco",
    ]
    model_df = df.drop(columns=cols_to_drop)
    return model_df


async def preprocess_df_for_catboost_executing(original_df):
    colonne_da_rimuovere = [
        "Corrispondenza Foto",
        "Nome e Cognome",
        "Numero di telefono",
        "Pre-op",
        "Pre-op.1",
    ]
    df_cleaned = original_df.drop(
        columns=[col for col in colonne_da_rimuovere if col in original_df.columns]
    )
    df_cleaned = df_cleaned.loc[:, ~df_cleaned.columns.str.contains("^Unnamed")]
    df = df_cleaned
    cat_cols = df.select_dtypes(include="object").columns
    for col, newcol in [
        ("Distanza giugulo-capezzolo", "jugulo_nipple_cm"),
        ("Diametro areola", "areola_diam_cm"),
        ("Distanza areola-solco", "areola_fold_cm"),
    ]:
        df[newcol] = df[col].apply(
            lambda x: (
                (sum(extract_float_list(x)) / len(extract_float_list(x)))
                if extract_float_list(x)
                else None
            )
        )
    return df


async def preprocess_df_for_catboost_from_body(data: dict) -> pd.DataFrame:
    mapped = {
        "Età": data.eta,
        "Grado di ptosi /pseudoptosi": data.grado_ptosi,
        "Volume del seno (ipoplasia/normale/gigantomastia + volume stimato da rimuovere </>500)": data.volume_seno,
        "Desiderio della paziente (aumento di volume?cicatrici +/- visibili)": data.desiderio_paziente,
        "BMI": data.bmi,
        "FUMO": data.fumo,
        "Qualità della pelle (elasticità/lassità/eccesso cutaneo)": data.qualita_pelle,
        "DISTURBI COAGULAZIONE /INR": data.disturbi_coagulazione,
        "Distanza giugulo-capezzolo dx": np.mean(
            [data.distanza_giugulo_sx, data.distanza_giugulo_dx]
        ),
        "Distanza giugulo-capezzolo sx": np.mean(
            [data.distanza_giugulo_sx, data.distanza_giugulo_dx]
        ),
        "Diametro areola": np.mean([data.diametro_areola_sx, data.diametro_areola_dx]),
        "Distanza areola-solco": np.mean(
            [data.distanza_areola_sx, data.distanza_areola_dx]
        ),
    }
    cat_features = [
        "Distanza giugulo-capezzolo dx",
        "Distanza giugulo-capezzolo sx",
        "Diametro areola",
        "Distanza areola-solco",
        "Grado di ptosi /pseudoptosi",
        "Qualità della pelle (elasticità/lassità/eccesso cutaneo)",
        "Volume del seno (ipoplasia/normale/gigantomastia + volume stimato da rimuovere </>500)",
        "Desiderio della paziente (aumento di volume?cicatrici +/- visibili)",
        "FUMO",
        "DISTURBI COAGULAZIONE /INR",
        "BMI",
    ]
    df = pd.DataFrame([mapped])
    for col in cat_features:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


# Funzioni helper -----------------------------------------------------------------------------------------------


# Funzione per pulire le stringhe: lowercase, accenti, spazi, simboli
def clean_string(val):
    if pd.isna(val):
        return None
    val = str(val).lower().strip()
    val = val.replace("é", "e").replace("è", "e")
    val = val.replace(".", "").replace(",", "")
    val = val.replace("-", " ").replace("_", " ").replace("  ", " ")
    return val


def map_desire(val):
    if pd.isna(val):
        return None
    t = str(val).lower()
    if "aument" in t or "+" in t:
        return "aumentare"
    if "riduc" in t or "minimiz" in t:
        return "ridurre"
    return "mantenere"


def map_volume(txt):
    if pd.isna(txt):
        return None, None
    t = str(txt).lower()
    cat = None
    if "ipo" in t:
        cat = 0
    elif "norm" in t:
        cat = 1
    elif "giganto" in t or "macro" in t:
        cat = 2
    # flag >500 cc
    flag = 1 if re.search(r">?\s*500", t) else 0
    return cat, flag


def map_skin(val):
    skin_map = {
        "buona": 0,
        "media": 1,
        "scarsa": 2,
        "scadente": 2,
        "elastica": 0,
    }
    if pd.isna(val):
        return None
    txt = str(val).lower()
    for k, v in skin_map.items():
        if k in txt:
            return v
    return None


# scgliamo un ordine perchè suppongo che il grado di ptosi sia un valore progressivo, II < II < III
def map_ptosis(val):
    if pd.isna(val):
        return None
    # capture Roman numerals I,II,III or digit
    m = re.search(r"(I{1,3}|[1-3])", str(val))
    if not m:
        return None
    roman = m.group(1)
    mapping = {"I": 1, "II": 2, "III": 3, "1": 1, "2": 2, "3": 3}
    return mapping.get(roman, None)


def extract_float_list(text):
    if pd.isna(text):
        return []
    # replace comma decimal with dot
    txt = str(text).replace(",", ".")
    # find all numbers (int or float)
    nums = re.findall(r"\d+(?:\.\d+)?", txt)
    return [float(n) for n in nums]
