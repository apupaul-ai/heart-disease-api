# Heart Disease Prediction API (FastAPI + Docker + Render)

A small FastAPI service that predicts the presence of heart disease from 13 medical attributes.
A scikit-learn `RandomForestClassifier` is trained on the Heart Disease dataset (Kaggle, johnsmith88),
saved with `joblib`, served with FastAPI, packaged with Docker and deployed on Render.

> Educational project - not a medical tool.

**Live URL (Render):** `https://heart-disease-api-8jfd.onrender.com`  <!-- replace after deploying -->
**Swagger UI:** `https://heart-disease-api-8jfd.onrender.com/docs`

---

## Project structure

```
heart-disease-api/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app: /health, /info, /predict
│   └── schemas.py         # Pydantic models (input validation + responses)
├── model/
│   ├── heart_model.joblib # trained model (loaded when the API starts)
│   └── model_info.json    # training metadata shown by /info
├── data/heart.csv         # dataset (only needed for training)
├── train_model.py         # trains the model and saves it to model/
├── requirements.txt       # runtime dependencies (used by Docker)
├── requirements-train.txt # runtime + pandas (used to train)
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── README.md
```

## Endpoints

| Method | Path       | Description                                                  |
|--------|------------|--------------------------------------------------------------|
| GET    | `/health`  | Health check -> `{"status": "ok", "model_loaded": true}`     |
| GET    | `/info`    | Model type, feature list and training details                |
| POST   | `/predict` | Takes the 13 features, returns `heart_disease` (true/false) and `probability` |
| GET    | `/docs`    | Swagger UI                                                   |

Invalid input (missing field, out-of-range value, wrong type) returns **HTTP 422** with a clear message.

### Input fields

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`
(ranges are validated in `app/schemas.py`).

---

## 1. Train the model (optional - the trained model is already included)

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1      Mac/Linux: source .venv/bin/activate
pip install -r requirements-train.txt
python train_model.py
```

This writes `model/heart_model.joblib` and `model/model_info.json`.

> **Important:** the scikit-learn version used for training must match the one in `requirements.txt`,
> because a joblib model can break when loaded with a different version. If you re-train, use the pinned versions.

## 2. Run locally (without Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs

## 3. Run with Docker Compose

```bash
docker-compose build
docker-compose up
```

(With newer Docker versions the command is `docker compose ...` without the hyphen.)

Open http://localhost:8000/docs . Stop with `Ctrl+C`, and clean up with `docker-compose down`.

Without compose:

```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```

## 4. Test with curl

Replace `http://localhost:8000` with your Render URL to test the live service.

```bash
curl http://localhost:8000/health

curl http://localhost:8000/info

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

Example response:

```json
{"heart_disease": true, "probability": 0.5923}
```

On Windows PowerShell, `curl` is an alias of `Invoke-WebRequest`; use `curl.exe` instead, or use the Swagger UI
(`/docs` -> `POST /predict` -> *Try it out*), or Postman.

PowerShell alternative:

```powershell
$body = '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body $body
```

---

## 5. Deploy on Render

1. **Push the project to GitHub**

   ```bash
   git init
   git add .
   git commit -m "FastAPI heart disease API with Docker"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

   Make sure `model/heart_model.joblib` is committed (it is not git-ignored), because the Docker image copies it.

2. On https://render.com : **New +  ->  Web Service**, connect your GitHub account and select the repository.
3. Settings:
   - **Language / Environment:** `Docker`
   - **Branch:** `main`
   - **Root Directory:** leave empty (the `Dockerfile` is in the repository root)
   - **Instance type:** Free
   - Under *Advanced*, set **Health Check Path** to `/health`
4. Click **Create Web Service**. Render builds the Dockerfile and starts the container.
   It provides a `PORT` environment variable and the container's start command already listens on it.
5. When the deploy is live, open `https://heart-disease-api-8jfd.onrender.com/docs` and test the endpoints.

Note: on the free plan the service sleeps after ~15 minutes without traffic, so the first request afterwards can
take 30-60 seconds.

---

## Dockerfile overview

- `python:3.12-slim` base image
- `requirements.txt` is copied and installed **before** the code so Docker can cache that layer
- copies `app/` and `model/` only (see `.dockerignore`)
- runs as a non-root user
- starts the app with `uvicorn app.main:app --host 0.0.0.0 --port ${PORT}` (`PORT` defaults to 8000 and is overridden by Render)

## Notes about the dataset

- **Duplicates:** the CSV has 1025 rows but only 302 unique ones. `train_model.py` removes the duplicates before
  splitting, otherwise identical patients would end up in both the train and test sets and inflate the accuracy.
  The honest test accuracy on the 61 held-out patients is about 77 %. Model accuracy is not the focus of this project.
- **Target direction:** the API follows the dataset's documented convention: `target = 1` -> `heart_disease: true`.
  However, in this CSV the rows labelled `1` have, on average, a *higher* maximum heart rate (158 vs 139), *lower* ST depression
  (0.6 vs 1.6) and *fewer* blocked vessels (0.3 vs 1.2) than the rows labelled `0`, which is the opposite of what is expected
  for patients with heart disease. So the labels in this file may be inverted compared with their description. If your
  course expects the medically intuitive direction, change `y = df[TARGET].to_numpy()` in `train_model.py` to
  `y = (1 - df[TARGET]).to_numpy()`, re-run the training script and rebuild the image.
