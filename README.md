# AIDEPS

## Descrizione
Project-M è un backend sviluppato in **FastAPI** per la gestione dell'autenticazione, API protette e Machine Learning.  
Il sistema permette di:
- **Autenticazione con JWT** (Access Token & Refresh Token)
- **Protezione API con Bearer Token**
- **Gestione utenti con MongoDB**
- **Caricare dataset in formato Excel**
- **Allenare modelli di Machine Learning** (Logistic Regression, Random Forest, SVC, KNN)
- **Eseguire i modelli e ottenere predizioni**
- **Visualizzare i plot delle metriche dei modelli**
- **Gestione sicura delle variabili d'ambiente con `.env`**

---

## ⚙️ **Tecnologie utilizzate**
- 🐍 **FastAPI** - Framework Python per API REST
- 🗄 **MongoDB** - Database NoSQL
- 🔑 **JWT (JSON Web Token)** - Per l'autenticazione sicura
- 📊 **Pandas** - Manipolazione dati e gestione dataset
- 🤖 **Scikit-learn** - Machine Learning
- 📈 **Matplotlib / Seaborn** - Visualizzazione delle metriche ML
- 🔧 **Motor** - Driver asincrono per MongoDB
- 🛠 **Python-Decouple** - Gestione variabili di configurazione `.env`
- 🏗 **Uvicorn** - Server ASGI per FastAPI

---

## **Come avviare il progetto**
### 1️⃣ **Clonare il repository**
```sh
git clone https://github.com/tuo-utente/project-m.git
cd project-m
```
### 2️⃣ **Avviare il progetto**
```sh
docker compose build --no-cache 
docker compose up -d

oppure

docker compose up --build -d
```# aideps-be
# aideps-be
