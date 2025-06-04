import os
import re
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from app.models.ml_models import InputModelUserData


async def load_model(type):
    models_path = "./app/ml_models/"
    if "catboost" == type:
        model = CatBoostClassifier()
        model.load_model(models_path + "catboost.cbm")
    elif "tabpfn" == type:
        model = joblib.load(models_path + "tabpfn.joblib")
    elif "keras" == type:
        model = joblib.load(models_path + "keras.joblib")
    return model


async def executeMLModelV2(df_orig, model, type: str):
    catboost_cat_cols_path = "./app/ml_models/catboost_cat_cols.joblib"
    if type == "catboost":
        """df_proc = await preprocess_df_for_catboost_from_body(df_orig.copy())
        print(df_proc.head())"""
        print("df_orig: ", df_orig)
        if os.path.exists(catboost_cat_cols_path):
            cat_cols = joblib.load(catboost_cat_cols_path)
        return await predict(model, df_orig, cat_cols)
    else:  # tabpfn o keras
        df_proc = await preprocess_df_for_tabpfn_keras_executing(df_orig.copy())
        return await predict(model, df_proc, [])


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
        "Distanza giugulo-capezzolo": np.mean(
            [data.distanza_giugulo_sx, data.distanza_giugulo_dx]
        ),
        "Diametro areola": np.mean([data.diametro_areola_sx, data.diametro_areola_dx]),
        "Distanza areola-solco": np.mean(
            [data.distanza_areola_sx, data.distanza_areola_dx]
        ),
    }
    cat_features = [
        "Distanza giugulo-capezzolo",
        "Diametro areola",
        "Distanza areola-solco",
        "Grado di ptosi /pseudoptosi",
        "Qualità della pelle (elasticità/lassità/eccesso cutaneo)",
        "Volume del seno (ipoplasia/normale/gigantomastia + volume stimato da rimuovere </>500)",
        "Desiderio della paziente (aumento di volume?cicatrici +/- visibili)",
        "FUMO",
        "DISTURBI COAGULAZIONE /INR",
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
