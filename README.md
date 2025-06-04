# AIDEPS

## Description

AIDEPS is a backend developed with **FastAPI** for managing authentication, protected APIs, and Machine Learning workflows.  
The system supports:
- **JWT-based authentication** (Access Token & Refresh Token)
- **API protection via Bearer Token**
- **User management with MongoDB**
- **Running models and generating predictions**

---

## Technologies Used

- **FastAPI** – Python framework for building RESTful APIs  
- **MongoDB** – NoSQL database for storing user and dataset information  
- **JWT (JSON Web Token)** – Secure token-based authentication  
- **Pandas** – Data manipulation and dataset handling  
- **Scikit-learn** – Machine Learning library  
- **Motor** – Asynchronous driver for MongoDB  
- **Python-Decouple** – Environment variable management via `.env` files  
- **Uvicorn** – ASGI server for running FastAPI applications  

---

## How to Run the Project

1. Clone the repository and move into the project directory:

   ```bash
   git clone <your-repo-url>
   cd your-project
   ```

2. Choose one of the two ways to run the backend:

---

### Option 1: Manual Python Setup (Local)

This method is recommended for development and debugging.

- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

- Start the backend server:

  ```bash
  uvicorn main:app --reload
  ```

- **Note:** You must have **MongoDB installed and running locally** on your machine.  
  Make sure your `.env` file includes a valid connection string, e.g.:

  ```
  MONGO_URI=mongodb://localhost:27017/your-db-name
  ```

---

### Option 2: Docker Compose (Backend + MongoDB)

This method is recommended for isolated, reproducible environments.

- Start everything with:

  ```bash
  docker-compose up --build
  ```

- This will automatically start both:
  - The FastAPI backend server
  - A MongoDB instance

---

3. Configure environment variables:

- Create a `.env` (or `.env.local`) file based on the provided `.env.example`
- Fill in the required variables (e.g., database URI, JWT secret, token expiration settings)

---

## Notes

- This backend is designed to work in conjunction with the [Mastopexy Predictive Interface (Frontend)](https://github.com/salmorf/aideps-be) built in React.
- APIs are protected and require a valid JWT for access.
- If you run via Docker, you do **not** need MongoDB installed on your system.
- If you run manually, make sure MongoDB is available and accessible locally.

---

