# Explainable AI for Linguistic Pattern Detection (web app)

Flask website around the from-scratch NumPy model in `xai_linguistic.py`.
The model trains in about 0.5 s at start-up, so there is no model file or database to manage.

## Files
| File | Purpose |
|---|---|
| `app.py` | Flask server (`/`, `/api/explain`, `/api/model`, `/healthz`) |
| `xai_linguistic.py` | The model: tokenizer, TF-IDF, logistic regression, pattern detector, XAI |
| `templates/index.html` | The web page (HTML + CSS + JS in one file) |
| `requirements.txt`, `Procfile`, `runtime.txt`, `render.yaml`, `Dockerfile` | Deployment config |

## Run locally
```
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Deploy on Render (free, recommended)
1. Create a GitHub repo and push this folder:
   ```
   git init && git add . && git commit -m "XAI web app"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
2. Go to https://render.com, sign in with GitHub, then **New > Web Service** and pick the repo.
   (Or **New > Blueprint**, which reads `render.yaml` automatically.)
3. Settings if asked: Runtime **Python 3**, Build `pip install -r requirements.txt`,
   Start `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`, Plan **Free**.
4. Click **Create Web Service**. After about 2 minutes you get a public URL like
   `https://xai-linguistic-patterns.onrender.com`.

Free instances sleep after about 15 minutes idle, so the first visit afterwards takes 30-60 s to wake.

## Other hosts
- **Railway / Heroku-style:** push to GitHub and connect the repo. The `Procfile` is picked up automatically.
- **Docker (Fly.io, Google Cloud Run, Hugging Face Spaces, any VPS):**
  ```
  docker build -t xai-web .
  docker run -p 8080:8080 xai-web
  ```
  Open http://localhost:8080. On Hugging Face Spaces choose the Docker SDK and set `PORT` to 7860.
- **PythonAnywhere:** upload the files, create a Flask web app, and point the WSGI file at `from app import app as application`.

## API
`POST /api/explain` with `{"text": "The movie was not bad."}` returns the label, confidence, per-token scores,
detected patterns and explanation notes as JSON. Input is limited to 400 characters.
