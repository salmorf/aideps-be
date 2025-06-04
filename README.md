# AIDEPS

## Description
Aideps is a backend developed with **FastAPI** for managing authentication, protected APIs, and Machine Learning workflows.  
The system supports:
- **JWT-based authentication** (Access Token & Refresh Token)
- **API protection via Bearer Token**
- **User management with MongoDB**
- **Uploading datasets in Excel format**
- **Training Machine Learning models**
- **Running models and generating predictions**
- **Displaying plots of model evaluation metrics**
- **Secure environment variable handling with `.env` files**

---

## Technologies Used
- **FastAPI** – Python framework for building RESTful APIs  
- **MongoDB** – NoSQL database for storing user and dataset information  
- **JWT (JSON Web Token)** – Secure token-based authentication  
- **Pandas** – Data manipulation and dataset handling  
- **Scikit-learn** – Machine Learning library  
- **Matplotlib / Seaborn** – Visualization of ML metrics  
- **Motor** – Asynchronous driver for MongoDB  
- **Python-Decouple** – Environment variable management via `.env` files  
- **Uvicorn** – ASGI server for running FastAPI applications  

---

## How to Run the Project

```bash
git clone https://github.com/your-username/project-m.git
cd project-m
docker compose build --no-cache
docker compose up -d
or
docker compose up --build -d