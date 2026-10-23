"""
Web front-end for "Explainable AI for Linguistic Pattern Detection".
Run locally :  python app.py          -> http://127.0.0.1:5000
Production  :  gunicorn app:app       (see Procfile / Dockerfile)
"""
import os

from flask import Flask, jsonify, render_template, request

import xai_linguistic as X

app = Flask(__name__)

# The model trains from scratch in ~0.5 s, so we simply train once at start-up.
MODEL, METRICS, SANITY, FAITH = X.train_model(verbose=False)
POS, NEG = MODEL.global_importance(10)
MAX_CHARS = 400


def _f(v):
    return round(float(v), 4)


def build_payload(text):
    e = MODEL.explain(text)
    conf = e["prob_pos"] if e["label"] else 1 - e["prob_pos"]
    label = MODEL.LABELS[e["label"]]
    if not e["features"]:                      # no word the model knows -> do not pretend
        label = "UNCERTAIN"
    return {
        "text": e["text"],
        "label": label,
        "prob_pos": _f(e["prob_pos"]),
        "confidence": _f(conf),
        "tokens": [
            {"t": t, "score": _f(s), "occl": _f(o)}
            for t, s, o in zip(e["tokens"], e["token_score"], e["occlusion"])
        ],
        "patterns": [
            {"type": p["type"], "start": p["start"], "text": p["text"],
             "scope": p["scope"], "desc": X.PATTERN_DESC[p["type"]]}
            for p in e["patterns"]
        ],
        "notes": e["notes"],
        "features": [{"f": f, "c": _f(c)} for f, c in e["features"][:8]],
    }


@app.get("/")
def index():
    return render_template("index.html", examples=X.DEMO)


@app.post("/api/explain")
def explain():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify(error="Please enter a sentence."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Please keep it under {MAX_CHARS} characters."), 400
    if not X.tokenize(text):
        return jsonify(error="No readable words found in that text."), 400
    return jsonify(build_payload(text))


@app.get("/api/model")
def model_info():
    return jsonify(
        accuracy=_f(METRICS["acc"]), f1=_f(METRICS["f1"]),
        precision=_f(METRICS["prec"]), recall=_f(METRICS["rec"]),
        confusion=METRICS["cm"], sanity_accuracy=_f(SANITY["acc"]),
        faithfulness={"top_tokens": _f(FAITH[0]), "random_tokens": _f(FAITH[1])},
        features=len(MODEL.vec.index),
        top_positive=[{"f": f, "w": _f(w)} for f, w in POS],
        top_negative=[{"f": f, "w": _f(w)} for f, w in NEG],
        pattern_types=X.PATTERN_DESC,
    )


@app.get("/favicon.ico")
def favicon():
    return "", 204


@app.get("/healthz")
def health():
    return "ok"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
