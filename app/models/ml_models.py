from pydantic import BaseModel, Field


class InputModelUserData(BaseModel):
    eta: int = Field(..., description="Età del paziente")
    grado_ptosi: str = Field(..., description="Grado di ptosi")
    volume_seno: str = Field(..., description="Volume del seno")
    desiderio_paziente: str = Field(..., description="Desiderio espresso dal paziente")
    bmi: int = Field(..., description="Indice di massa corporea")
    fumo: str = Field(
        ..., description="Consumo di fumo (es. 0 = non fumante, 1 = fumante)"
    )
    qualita_pelle: str = Field(..., description="Valutazione della qualità della pelle")
    disturbi_coagulazione: str = Field(
        ..., description="Presenza di disturbi della coagulazione (0 o 1)"
    )
    distanza_giugulo_sx: int = Field(..., description="Distanza dal giugulo sinistro")
    distanza_giugulo_dx: int = Field(..., description="Distanza dal giugulo destro")
    diametro_areola_sx: int = Field(..., description="Diametro dell'areola sinistra")
    diametro_areola_dx: int = Field(..., description="Diametro dell'areola destra")
    distanza_areola_sx: int = Field(..., description="Distanza dell'areola sinistra")
    distanza_areola_dx: int = Field(..., description="Distanza dell'areola destra")


class ExecuteResult(BaseModel):
    success: bool = Field(
        ..., description="Indica se l'operazione è andata a buon fine"
    )
    result: str = Field(..., description="Risultato dell'operazione")
    user_data: InputModelUserData = Field(..., description="Dati forniti dall'utente")


class ExecuteResultRealOneRequest(BaseModel):
    eta: int
    fumo: str  # "Sì" o "No"
    bmi: float
    distanza_giugulo_capezzolo_sx: float  # in cm
    distanza_giugulo_capezzolo_dx: float  # in cm
    diametro_areola_sx: float  # in cm
    diametro_areola_dx: float  # in cm
    distanza_areola_solco_sx: float  # in cm
    distanza_areola_solco_dx: float  # in cm
    grado_ptosi: str  # Es. "I sec. Regnault"
    volume_seno: str  # Es. "Gigantomastia"
    disturbi_coagulazione: str  # "Sì" o "No"
    desiderio_paziente: str  # "Sì" o "No"
    qualita_pelle: str  # Es. "Scarsa"


class ExecuteResultRealOneResponse(BaseModel):
    success: bool = Field(
        ..., description="Indica se l'operazione è andata a buon fine"
    )
    result: str = Field(..., description="Risultato dell'operazione")
