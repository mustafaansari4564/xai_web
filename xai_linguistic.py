#!/usr/bin/env python3
"""
Explainable AI for Linguistic Pattern Detection
================================================
NLP mini project - built from scratch (only NumPy + Python standard library).

What it does
------------
1. Tokenizer, n-gram TF-IDF vectorizer and Logistic Regression  -> all written by hand.
2. Rule-based LINGUISTIC PATTERN DETECTOR: negation (with scope), intensifiers,
   hedges, contrast ("but"/"although"), modals, first-person, questions, exclamations.
3. EXPLAINABILITY (XAI) layers:
     a) Global   : most influential words / bigrams learned by the model
     b) Local    : per-token contribution (weight x tf-idf) for one sentence
     c) Occlusion: remove a token, see how the prediction changes (LIME-like, model-agnostic)
     d) Pattern-aware: shows how linguistic patterns (e.g. negation) change the decision
     e) Faithfulness test: do the tokens marked "important" really matter to the model?
4. Writes an HTML report with colour-highlighted tokens and pattern badges.

Usage
-----
    python xai_linguistic.py                      # full demo + report.html
    python xai_linguistic.py --text "The food was not good, but the service was great!"
    python xai_linguistic.py --interactive        # type your own sentences
"""
import argparse
import html
import math
import random
import re

import numpy as np

# ----------------------------------------------------------------------------
# 1. DATASET (synthetic, template based so that linguistic patterns are present)
# ----------------------------------------------------------------------------
POS_ADJ = ["good", "great", "excellent", "amazing", "wonderful", "fantastic",
           "brilliant", "lovely", "superb", "delightful", "impressive", "enjoyable"]
NEG_ADJ = ["bad", "terrible", "awful", "horrible", "poor", "disappointing",
           "boring", "dreadful", "mediocre", "annoying", "useless", "unpleasant"]
NOUNS = ["movie", "food", "service", "phone", "hotel", "book", "teacher",
         "app", "story", "performance", "camera", "battery"]

PLAIN = ["The {n} was {a}.", "I thought the {n} was {a}.", "This {n} is really {a}!",
         "What a {a} {n}!", "Honestly the {n} seemed {a} to me.",
         "The {n} is very {a}.", "Such a {a} {n}."]
NEGATED = ["The {n} was not {a}.", "The {n} is never {a}.", "I don't think the {n} was {a}.",
           "The {n} wasn't {a}.", "The {n} was not at all {a}.", "The {n} is hardly {a}."]
CONTRAST = ["The {n} looked {a1} but it turned out {a2}.",
            "Although the {n} seemed {a1}, it was {a2}.",
            "At first the {n} was {a1}, however in the end it was {a2}."]
NEG_CONTRAST = ["The {n} was not {a1}, but it was {a2}.", "The {n} wasn't {a1} but the {m} was {a2}."]
VERBS = [("I loved the {n}.", 1), ("I hated the {n}.", 0), ("I really liked the {n}.", 1),
         ("I did not like the {n}.", 0), ("I did not hate the {n}.", 1),
         ("I really enjoyed the {n}!", 1), ("I truly disliked the {n}.", 0)]


def build_dataset(seed=7, per_template=90):
    """Returns list of (sentence, label) ; label 1 = positive, 0 = negative."""
    rng = random.Random(seed)
    data = {}

    def pick_adj():
        return rng.choice([(a, 1) for a in POS_ADJ] + [(a, 0) for a in NEG_ADJ])

    for t in PLAIN:
        for _ in range(per_template):
            a, y = pick_adj()
            data[t.format(n=rng.choice(NOUNS), a=a)] = y
    for t in NEGATED:                      # negation flips the polarity
        for _ in range(per_template):
            a, y = pick_adj()
            data[t.format(n=rng.choice(NOUNS), a=a)] = 1 - y
    for t in CONTRAST:                     # the clause after "but" decides
        for _ in range(per_template):
            a1, _y1 = pick_adj()
            a2, y2 = pick_adj()
            data[t.format(n=rng.choice(NOUNS), a1=a1, a2=a2)] = y2
    for t in NEG_CONTRAST:                 # negation + contrast together
        for _ in range(per_template):
            a1, _y1 = pick_adj()
            a2, y2 = pick_adj()
            data[t.format(n=rng.choice(NOUNS), m=rng.choice(NOUNS), a1=a1, a2=a2)] = y2
    for t, y in VERBS:
        for n in NOUNS:
            data[t.format(n=n)] = y
    items = list(data.items())
    rng.shuffle(items)
    return items


# Hand-written sentences, never produced by a template -> sanity check
SANITY = [
    ("The hotel was not good.", 0), ("I really loved the story!", 1),
    ("The service was never disappointing.", 1),
    ("The camera looked amazing but it turned out terrible.", 0),
    ("The battery looked poor but it turned out excellent.", 1),
    ("I hated the app.", 0), ("I don't think the book was boring.", 1),
    ("What a wonderful teacher!", 1), ("The food was not at all enjoyable.", 0),
    ("This phone is really useless.", 0),
    ("Honestly the performance seemed lovely to me.", 1), ("The movie was not bad.", 1),
]

# ----------------------------------------------------------------------------
# 2. TOKENIZER + VECTORIZER (from scratch)
# ----------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[!?.,;]")
CLAUSE_PUNCT = {",", ".", ";"}          # kept for pattern scope, ignored by the model


def tokenize(text):
    return TOKEN_RE.findall(text.lower().replace("\u2019", "'"))


class TfidfVectorizer:
    """Unigram + bigram TF-IDF with L2 normalisation."""

    def __init__(self, min_df=2):
        self.min_df = min_df
        self.index, self.idf = {}, None

    @staticmethod
    def ngram_spans(words, tags):
        """yield (feature_string, start, n).
        Besides plain unigrams/bigrams we add PATTERN-TAGGED unigrams:
        '[neg]good'   = 'good' inside a negation scope
        '[final]good' = 'good' inside the final (deciding) clause after but/however/although
        '[pre]good'   = 'good' in the clause BEFORE the contrast (usually overruled)"""
        for i, w in enumerate(words):
            yield w, i, 1
            if "N" in tags[i]:
                yield "[neg]" + w, i, 1
            if "F" in tags[i]:
                yield "[final]" + w, i, 1
            if "B" in tags[i]:
                yield "[pre]" + w, i, 1
            if i + 1 < len(words):
                yield w + " " + words[i + 1], i, 2

    def fit(self, docs):
        """docs = list of (words, tags)"""
        df = {}
        for words, tags in docs:
            for f in {f for f, _, _ in self.ngram_spans(words, tags)}:
                df[f] = df.get(f, 0) + 1
        feats = sorted(f for f, c in df.items() if c >= self.min_df)
        self.index = {f: j for j, f in enumerate(feats)}
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[f])) + 1 for f in feats])
        return self

    def transform_one(self, words, tags):
        x = np.zeros(len(self.index))
        occ = {}
        for f, s, n in self.ngram_spans(words, tags):
            j = self.index.get(f)
            if j is not None:
                x[j] += 1
                occ.setdefault(f, []).append((s, n))
        x *= self.idf
        norm = np.linalg.norm(x)
        if norm > 0:
            x /= norm
        return x, occ

    def transform(self, docs):
        return np.vstack([self.transform_one(w, t)[0] for w, t in docs])


# ----------------------------------------------------------------------------
# 3. LOGISTIC REGRESSION (from scratch, batch gradient descent + L2)
# ----------------------------------------------------------------------------
class LogisticRegression:
    def __init__(self, lr=1.0, epochs=600, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.w, self.b = None, 0.0

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):
            p = self.sigmoid(X @ self.w + self.b)
            gw = X.T @ (p - y) / n + self.l2 * self.w
            gb = float(np.mean(p - y))
            self.w -= self.lr * gw
            self.b -= self.lr * gb
        return self

    def proba(self, X):
        return self.sigmoid(X @ self.w + self.b)


# ----------------------------------------------------------------------------
# 4. LINGUISTIC PATTERN DETECTOR (rule based, returns token spans)
# ----------------------------------------------------------------------------
LEXICONS = {
    "negation": {"not", "no", "never", "n't", "don't", "doesn't", "didn't", "isn't", "wasn't",
                 "aren't", "weren't", "can't", "won't", "cannot", "hardly", "barely",
                 "neither", "nor", "nothing", "without"},
    "intensifier": {"very", "really", "extremely", "absolutely", "totally", "so", "incredibly",
                    "highly", "truly", "completely", "utterly", "quite"},
    "hedge": {"maybe", "perhaps", "possibly", "probably", "might", "seems", "seemed",
              "somewhat", "apparently", "guess", "sort", "kind", "honestly"},
    "contrast": {"but", "however", "although", "though", "yet", "while"},
    "modal": {"should", "could", "would", "must", "can", "will", "may", "shall"},
    "first_person": {"i", "my", "me", "we", "our", "mine"},
}
PATTERN_DESC = {
    "negation": "flips / weakens the polarity of nearby words",
    "intensifier": "amplifies the following word",
    "hedge": "softens the speaker's commitment",
    "contrast": "the clause AFTER it usually carries the final opinion",
    "modal": "expresses possibility, obligation or ability",
    "first_person": "subjective / personal opinion",
    "question": "interrogative sentence",
    "exclamation": "emphatic / emotional tone",
}
NEGATION_SCOPE = 5                      # words affected after a negator


def detect_patterns(tokens):
    """Return list of dicts: {type, start, end(exclusive), text, scope:[idx..]}"""
    found = []
    for i, t in enumerate(tokens):
        for ptype, lex in LEXICONS.items():
            if t in lex:
                p = {"type": ptype, "start": i, "end": i + 1, "text": t, "scope": []}
                if ptype == "negation":
                    j = i + 1
                    while j < len(tokens) and len(p["scope"]) < NEGATION_SCOPE:
                        if tokens[j] in CLAUSE_PUNCT or tokens[j] in "!?" \
                                or tokens[j] in LEXICONS["contrast"]:
                            break
                        p["scope"].append(j)
                        j += 1
                found.append(p)
        if t == "?":
            found.append({"type": "question", "start": i, "end": i + 1, "text": t, "scope": []})
        if t == "!":
            found.append({"type": "exclamation", "start": i, "end": i + 1, "text": t, "scope": []})
    return found


# ----------------------------------------------------------------------------
# 5. THE EXPLAINABLE PIPELINE
# ----------------------------------------------------------------------------
class ExplainableClassifier:
    LABELS = {1: "POSITIVE", 0: "NEGATIVE"}

    def __init__(self):
        self.vec = TfidfVectorizer(min_df=2)
        self.clf = LogisticRegression(lr=10.0, epochs=1500, l2=1e-4)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def pattern_flags(tokens):
        """Per-token flags: 'N' = inside a negation scope, 'F' = in the final clause
        (after but / however / yet, or after the comma of a sentence-initial although),
        'B' = in the clause before that contrast."""
        flags = [""] * len(tokens)
        for p in detect_patterns(tokens):
            if p["type"] == "negation":
                for j in p["scope"]:
                    flags[j] += "N"
        for i, t in enumerate(tokens):
            start = None
            if t in {"but", "however", "yet"}:
                start = i + 1
            elif t in {"although", "though", "while"} and i == 0:
                commas = [j for j in range(i, len(tokens)) if tokens[j] == ","]
                if commas:
                    start = commas[0] + 1
            if start is not None:
                for j in range(start, len(tokens)):
                    if tokens[j] not in CLAUSE_PUNCT:
                        flags[j] += "F"
                first = 1 if t in {"although", "though", "while"} else 0
                for j in range(first, i if t in {"but", "however", "yet"} else start):
                    if tokens[j] not in CLAUSE_PUNCT and "F" not in flags[j]:
                        flags[j] += "B"
        return flags

    @classmethod
    def prepare(cls, tokens):
        """-> (indices of tokens the model sees, their words, their pattern flags)"""
        flags = cls.pattern_flags(tokens)
        keep = [i for i, t in enumerate(tokens) if t not in CLAUSE_PUNCT]
        return keep, [tokens[i] for i in keep], [flags[i] for i in keep]

    def fit(self, sentences, labels):
        docs = [self.prepare(tokenize(s))[1:] for s in sentences]
        self.vec.fit(docs)
        self.clf.fit(self.vec.transform(docs), np.array(labels, dtype=float))
        return self

    def predict_proba_tokens(self, tokens):
        _, words, tags = self.prepare(tokens)
        x, _ = self.vec.transform_one(words, tags)
        return float(self.clf.proba(x[None, :])[0])

    def predict_proba(self, text):
        return self.predict_proba_tokens(tokenize(text))

    def predict(self, text):
        return int(self.predict_proba(text) >= 0.5)

    # -- XAI: global ---------------------------------------------------------
    def global_importance(self, k=10):
        feats = sorted(self.vec.index, key=self.vec.index.get)
        order = np.argsort(self.clf.w)
        neg = [(feats[j], float(self.clf.w[j])) for j in order[:k]]
        pos = [(feats[j], float(self.clf.w[j])) for j in order[::-1][:k]]
        return pos, neg

    # -- XAI: local ----------------------------------------------------------
    def explain(self, text):
        tokens = tokenize(text)
        keep, words, tags = self.prepare(tokens)
        x, occ = self.vec.transform_one(words, tags)
        z = float(x @ self.clf.w + self.clf.b)
        p = float(self.clf.sigmoid(z))
        contrib = self.clf.w * x                      # exact additive contribution

        # (a) distribute feature contributions onto tokens
        tok_score = np.zeros(len(tokens))
        for f, spans in occ.items():
            c = contrib[self.vec.index[f]]
            members = sum(n for _, n in spans)
            share = c / members
            for s, n in spans:
                for k in range(s, s + n):
                    tok_score[keep[k]] += share

        # (b) occlusion: delete one token and watch the probability move
        occl = np.zeros(len(tokens))
        for orig in keep:
            reduced = tokens[:orig] + tokens[orig + 1:]
            occl[orig] = p - self.predict_proba_tokens(reduced)

        patterns = detect_patterns(tokens)
        return {
            "text": text, "tokens": tokens, "prob_pos": p,
            "label": int(p >= 0.5), "bias": float(self.clf.b),
            "token_score": tok_score, "occlusion": occl,
            "features": sorted(((f, float(contrib[self.vec.index[f]])) for f in occ),
                               key=lambda t: -abs(t[1])),
            "patterns": patterns,
            "notes": self.pattern_notes(tokens, patterns, tok_score),
        }

    def pattern_notes(self, tokens, patterns, tok_score):
        """Human-readable explanation of what the patterns did to the decision."""
        notes = []
        feats = self.vec.index
        w = self.clf.w
        for p in patterns:
            if p["type"] == "negation" and p["scope"]:
                scoped = [tokens[j] for j in p["scope"]]
                polarity = []
                for t in scoped:                       # how does the word read ON ITS OWN?
                    if t in feats:
                        pa = self.predict_proba_tokens([t])
                        if abs(pa - 0.5) > 0.15:
                            polarity.append((t, "positive" if pa > 0.5 else "negative"))
                bigram = p["text"] + " " + scoped[0]
                learned = bigram in feats
                if polarity:
                    t, pol = polarity[-1]
                    notes.append(
                        f"NEGATION: '{p['text']}' scopes over {scoped}. Word '{t}' reads {pol} on its "
                        f"own, but inside the negation scope its effect is reversed"
                        + (f" (bigram '{bigram}' was also learned)." if learned else "."))
                else:
                    notes.append(f"NEGATION: '{p['text']}' scopes over {scoped}.")
            elif p["type"] == "contrast":
                after = [i for i in range(p["end"], len(tokens))
                         if tokens[i] not in CLAUSE_PUNCT and tokens[i] not in "!?"]
                if after:
                    s = float(sum(tok_score[i] for i in after))
                    notes.append(
                        f"CONTRAST: '{p['text']}' - the clause after it contributes "
                        f"{s:+.2f} to the decision (clause after contrast usually dominates).")
            elif p["type"] == "intensifier":
                notes.append(f"INTENSIFIER: '{p['text']}' strengthens the next word.")
            elif p["type"] == "hedge":
                notes.append(f"HEDGE: '{p['text']}' softens the statement.")
        return notes

    # -- XAI: text report ----------------------------------------------------
    def print_explanation(self, e):
        label = self.LABELS[e["label"]]
        conf = e["prob_pos"] if e["label"] else 1 - e["prob_pos"]
        print(f'\nSentence   : "{e["text"]}"')
        print(f"Prediction : {label}  (confidence {conf:.1%}, P(positive)={e['prob_pos']:.3f})")
        print("Token influence  (+ pushes POSITIVE, - pushes NEGATIVE)")
        mx = max(1e-9, np.abs(e["token_score"]).max())
        for t, s, o in zip(e["tokens"], e["token_score"], e["occlusion"]):
            if t in CLAUSE_PUNCT:
                continue
            bar = ("+" if s >= 0 else "-") * int(round(abs(s) / mx * 20))
            print(f"   {t:<14}{s:+.3f}  occl:{o:+.3f}  {bar}")
        if e["patterns"]:
            kinds = {}
            for p in e["patterns"]:
                kinds.setdefault(p["type"], []).append(p["text"])
            print("Linguistic patterns found: " +
                  "; ".join(f"{k} {v}" for k, v in kinds.items()))
        for n in e["notes"]:
            print("   ->", n)

    # -- evaluation ----------------------------------------------------------
    def evaluate(self, sentences, labels):
        pred = np.array([self.predict(s) for s in sentences])
        y = np.array(labels)
        tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        acc = (tp + tn) / len(y)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": [[tn, fp], [fn, tp]]}

    def faithfulness(self, sentences, k=2, seed=0):
        """Deleting the k tokens the explanation calls most important should hurt the
        prediction more than deleting k random tokens."""
        rng = random.Random(seed)
        drop_expl, drop_rand = [], []
        for s in sentences:
            e = self.explain(s)
            tokens = e["tokens"]
            keep, words, _ = self.prepare(tokens)
            if len(words) <= k:
                continue
            p0 = e["prob_pos"]
            sign = 1 if p0 >= 0.5 else -1
            scores = [e["token_score"][i] * sign for i in keep]     # support for predicted class
            top = {keep[j] for j in np.argsort(scores)[::-1][:k]}
            rnd = {keep[j] for j in rng.sample(range(len(keep)), k)}
            for chosen, store in ((top, drop_expl), (rnd, drop_rand)):
                red = [t for i, t in enumerate(tokens) if i not in chosen]
                p1 = self.predict_proba_tokens(red)
                store.append(sign * (p0 - p1))                      # drop in confidence
        return float(np.mean(drop_expl)), float(np.mean(drop_rand))


# ----------------------------------------------------------------------------
# 6. HTML REPORT
# ----------------------------------------------------------------------------
BADGE_COLOURS = {"negation": "#d9480f", "intensifier": "#7048e8", "hedge": "#868e96",
                 "contrast": "#1c7ed6", "modal": "#0b7285", "first_person": "#5c940d",
                 "question": "#c2255c", "exclamation": "#e67700"}


def token_html(e):
    mx = max(1e-9, np.abs(e["token_score"]).max())
    neg_idx = {j for p in e["patterns"] if p["type"] == "negation" for j in p["scope"]}
    pat_idx = {p["start"]: p["type"] for p in e["patterns"]}
    out = []
    for i, t in enumerate(e["tokens"]):
        s = e["token_score"][i]
        a = min(1.0, abs(s) / mx) * 0.75
        col = f"rgba(46,160,67,{a:.2f})" if s >= 0 else f"rgba(218,54,51,{a:.2f})"
        style = f"background:{col};"
        if i in neg_idx:
            style += "border-bottom:2px dashed #d9480f;"
        tip = f"score {s:+.3f} | occlusion {e['occlusion'][i]:+.3f}"
        if i in pat_idx:
            tip += f" | pattern: {pat_idx[i]}"
            style += f"outline:2px solid {BADGE_COLOURS[pat_idx[i]]};"
        out.append(f'<span class="tok" style="{style}" title="{tip}">{html.escape(t)}</span>')
    return " ".join(out)


def write_report(model, explanations, metrics, sanity, faith, path="report.html"):
    pos, neg = model.global_importance(10)
    rows = ""
    for e in explanations:
        label = model.LABELS[e["label"]]
        conf = e["prob_pos"] if e["label"] else 1 - e["prob_pos"]
        badges = "".join(
            f'<span class="badge" style="background:{BADGE_COLOURS[p["type"]]}">'
            f'{p["type"]}: {html.escape(p["text"])}</span>' for p in e["patterns"])
        notes = "".join(f"<li>{html.escape(n)}</li>" for n in e["notes"])
        feats = ", ".join(f"{html.escape(f)} ({c:+.2f})" for f, c in e["features"][:6])
        cls = "pos" if e["label"] else "neg"
        rows += (f'<div class="card"><div class="sent">{token_html(e)}</div>'
                 f'<div class="pred {cls}">{label} &middot; {conf:.1%}</div>'
                 f'<div>{badges}</div><ul>{notes}</ul>'
                 f'<div class="small">Top features: {feats}</div></div>')
    gl = lambda lst: "".join(f"<tr><td>{html.escape(f)}</td><td>{w:+.2f}</td></tr>" for f, w in lst)
    cm = metrics["cm"]
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>XAI Linguistic Pattern Detection</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#222}}
h1{{margin-bottom:0}} .sub{{color:#666;margin-top:4px}}
.card{{border:1px solid #ddd;border-radius:10px;padding:14px;margin:12px 0}}
.sent{{font-size:19px;line-height:2}} .tok{{padding:2px 5px;border-radius:5px}}
.badge{{color:#fff;border-radius:10px;padding:2px 9px;font-size:12px;margin:2px 4px 2px 0;display:inline-block}}
.pred{{font-weight:700;margin:6px 0}} .pos{{color:#2b8a3e}} .neg{{color:#c92a2a}}
.small{{color:#666;font-size:13px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:4px 10px}}
.grid{{display:flex;gap:30px;flex-wrap:wrap}}
</style></head><body>
<h1>Explainable AI for Linguistic Pattern Detection</h1>
<p class="sub">Green = pushes POSITIVE, red = pushes NEGATIVE (hover a word for exact scores).
Coloured outline = detected linguistic pattern, dashed underline = inside a negation scope.</p>
<h2>Model performance</h2>
<p>Held-out test accuracy <b>{metrics['acc']:.1%}</b>, F1 <b>{metrics['f1']:.3f}</b>
(confusion matrix [[TN,FP],[FN,TP]] = {cm}).<br>
Hand-written sanity set accuracy: <b>{sanity['acc']:.1%}</b>.<br>
Faithfulness: deleting top-2 explained tokens drops confidence by <b>{faith[0]:.3f}</b>
vs <b>{faith[1]:.3f}</b> for random tokens.</p>
<h2>Global explanation (learned weights)</h2><div class="grid">
<div><h3>Most positive</h3><table>{gl(pos)}</table></div>
<div><h3>Most negative</h3><table>{gl(neg)}</table></div></div>
<h2>Local explanations</h2>{rows}</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


# ----------------------------------------------------------------------------
# 7. MAIN
# ----------------------------------------------------------------------------
def train_model(verbose=True):
    data = build_dataset()
    split = int(0.8 * len(data))
    train, test = data[:split], data[split:]
    model = ExplainableClassifier().fit([s for s, _ in train], [y for _, y in train])
    m = model.evaluate([s for s, _ in test], [y for _, y in test])
    sn = model.evaluate([s for s, _ in SANITY], [y for _, y in SANITY])
    faith = model.faithfulness([s for s, _ in test][:60])
    if verbose:
        print(f"Dataset: {len(train)} train / {len(test)} test sentences, "
              f"{len(model.vec.index)} features (unigrams + bigrams)")
        print(f"Test accuracy   : {m['acc']:.1%}  precision {m['prec']:.3f}  "
              f"recall {m['rec']:.3f}  F1 {m['f1']:.3f}")
        print(f"Confusion matrix [[TN,FP],[FN,TP]]: {m['cm']}")
        print(f"Sanity (unseen hand-written, n={len(SANITY)}): {sn['acc']:.1%}")
        print(f"Faithfulness: top-2 explained tokens removed -> confidence drop {faith[0]:.3f}"
              f"  | random tokens -> {faith[1]:.3f}")
    return model, m, sn, faith


DEMO = [
    "The food was not good, but the service was great!",
    "I don't think the book was boring.",
    "The camera looked amazing but it turned out terrible.",
    "Honestly the performance seemed really lovely to me.",
    "The movie was not bad.",
    "Was the hotel really that awful?",
]


def main():
    ap = argparse.ArgumentParser(description="Explainable AI for Linguistic Pattern Detection")
    ap.add_argument("--text", help="explain a single sentence")
    ap.add_argument("--interactive", action="store_true", help="type sentences interactively")
    ap.add_argument("--report", default="report.html", help="HTML report path")
    args = ap.parse_args()

    print("=" * 70 + "\n Explainable AI for Linguistic Pattern Detection\n" + "=" * 70)
    model, metrics, sanity, faith = train_model()

    pos, neg = model.global_importance(8)
    print("\nGLOBAL EXPLANATION - strongest learned features")
    print("  positive:", ", ".join(f"{f}({w:+.1f})" for f, w in pos))
    print("  negative:", ", ".join(f"{f}({w:+.1f})" for f, w in neg))

    if args.text:
        model.print_explanation(model.explain(args.text))
        return
    if args.interactive:
        print("\nType a sentence (empty line to quit).")
        while True:
            try:
                s = input("> ").strip()
            except EOFError:
                break
            if not s:
                break
            model.print_explanation(model.explain(s))
        return

    exps = [model.explain(s) for s in DEMO + [s for s, _ in SANITY[:4]]]
    for e in exps[:len(DEMO)]:
        model.print_explanation(e)
    path = write_report(model, exps, metrics, sanity, faith, args.report)
    print(f"\nHTML report written to: {path}  (open it in your browser)")


if __name__ == "__main__":
    main()
