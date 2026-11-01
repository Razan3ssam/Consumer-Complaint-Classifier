import os
import re

import gradio as gr
import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1) Setup: NLTK + model 
for pkg in ("stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
id2label = model.config.id2label 

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

THRESHOLD = 0.60 

PRETTY = {
    "credit_card": "Credit Card",
    "credit_reporting": "Credit Reporting",
    "debt_collection": "Debt Collection",
    "mortgages_and_loans": "Mortgages & Loans",
    "retail_banking": "Retail Banking",
}
ICONS = {
    "credit_card": "💳",
    "credit_reporting": "📄",
    "debt_collection": "📞",
    "mortgages_and_loans": "🏠",
    "retail_banking": "🏦",
}

EXAMPLES = [
    ("💳 Late fee", "I was charged a late fee on my credit card even though I paid before the due date. Capital One refuses to remove it."),
    ("📞 Debt collector", "A debt collector keeps calling me at work about a debt I do not owe and will not send me written validation."),
    ("📄 Credit report", "There is an account on my Experian credit report that does not belong to me and they refuse to remove it after my dispute."),
]


# 2) Preprocessing + prediction (same cleaning used in training)
def clean_text(text: str) -> str:
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return " ".join(
        LEMMATIZER.lemmatize(w) for w in text.split() if w not in STOP_WORDS
    )


def predict_scores(text: str) -> dict:
    enc = tok(clean_text(text), truncation=True, max_length=256, return_tensors="pt")
    with torch.inference_mode():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].numpy()
    return {id2label[i]: float(probs[i]) for i in range(len(probs))}


# 3) Styling 
CSS = """
<style>
:root{
  --blue:#7195BC; --blue-soft:#E6EDF6; --blue-dark:#3F5F86;
  --wine:#650014; --wine-dark:#3A000C; --wine-light:#A3123A;
  --ink:#1D2330; --muted:#5B6678;
}
body, .gradio-container{
  background: linear-gradient(135deg,#EEF3F9 0%,#F9F4F5 100%) !important;
}
.gradio-container{ max-width:100% !important; width:100% !important; margin:0 !important;
  padding:0 clamp(16px,4vw,64px) 24px !important;
  font-family:'Segoe UI',system-ui,-apple-system,sans-serif !important; }
.gradio-container .main, .gradio-container .fillable, .gradio-container .contain,
.gradio-container .wrap{ max-width:100% !important; }

@keyframes fadeUp{ from{opacity:0; transform:translateY(18px)} to{opacity:1; transform:none} }
@keyframes floaty{ 0%,100%{transform:translateY(0) scale(1)} 50%{transform:translateY(-18px) scale(1.06)} }
@keyframes grow{ from{width:0} }
@keyframes pop{ 0%{transform:scale(.85); opacity:0} 70%{transform:scale(1.05)} 100%{transform:scale(1); opacity:1} }
@keyframes glow{ 0%,100%{box-shadow:0 0 0 0 rgba(101,0,20,.35)} 50%{box-shadow:0 0 0 7px rgba(101,0,20,0)} }
@keyframes shine{ to{left:130%} }
@keyframes bounce{ 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }

/* ---------- hero ---------- */
.hero{ position:relative; overflow:hidden; border-radius:26px; padding:38px 34px;
  background:linear-gradient(135deg,var(--wine) 0%,var(--wine-dark) 100%);
  color:#fff; margin:10px 0 22px; animation:fadeUp .7s ease both;
  box-shadow:0 18px 40px rgba(101,0,20,.25); }
.hero .blob{ position:absolute; border-radius:50%; background:var(--blue); opacity:.28; filter:blur(2px); }
.hero .b1{ width:220px; height:220px; right:-50px; top:-70px; animation:floaty 7s ease-in-out infinite; }
.hero .b2{ width:130px; height:130px; right:150px; bottom:-60px; opacity:.2; animation:floaty 9s ease-in-out infinite reverse; }
.hero .b3{ width:70px; height:70px; left:55%; top:20px; opacity:.18; animation:floaty 6s ease-in-out infinite; }
.hero-badge{ display:inline-block; padding:5px 14px; border-radius:999px; font-size:12px;
  letter-spacing:.08em; background:rgba(113,149,188,.28); color:#DCE8F6; margin-bottom:14px; }
.hero h1{ margin:0 0 8px; font-size:2.3rem; line-height:1.15; color:#fff !important; font-weight:800; }
.hero h1 span{ color:#B9D0EA; }
.hero p{ margin:0 0 18px; max-width:560px; color:#F1DDE2 !important; font-size:1.02rem; }
.chips{ display:flex; gap:10px; flex-wrap:wrap; }
.chips span{ padding:7px 14px; border-radius:12px; font-size:.85rem; font-weight:600;
  background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.22); color:#fff; transition:.25s; }
.chips span:hover{ background:var(--blue); transform:translateY(-3px); }

/* ---------- input ---------- */
#complaint-box{ animation:fadeUp .8s .1s ease both; background:transparent !important; border:none !important; }
#complaint-box label span{ color:var(--wine) !important; font-weight:700 !important; }
#complaint-box textarea{
  background:#fff !important; color:var(--ink) !important;
  border:2px solid #D3DEEC !important; border-radius:16px !important;
  padding:14px !important; font-size:1rem !important; transition:.25s !important; }
#complaint-box textarea:focus{ border-color:var(--blue) !important;
  box-shadow:0 0 0 5px rgba(113,149,188,.28) !important; }
#complaint-box textarea::placeholder{ color:#98A3B5 !important; }

/* ---------- buttons ---------- */
#submit-btn{ position:relative; overflow:hidden; border:none !important; border-radius:14px !important;
  color:#fff !important; font-weight:700 !important; font-size:1.02rem !important;
  background:linear-gradient(135deg,var(--wine),var(--wine-light)) !important;
  transition:transform .2s, box-shadow .2s !important; }
#submit-btn:hover{ transform:translateY(-3px); box-shadow:0 12px 26px rgba(101,0,20,.35); }
#submit-btn:active{ transform:translateY(0) scale(.98); }
#submit-btn::after{ content:""; position:absolute; top:0; left:-80%; width:45%; height:100%;
  background:linear-gradient(120deg,transparent,rgba(255,255,255,.4),transparent); transform:skewX(-20deg); }
#submit-btn:hover::after{ animation:shine .8s; }

#clear-btn{ background:#fff !important; color:var(--blue-dark) !important;
  border:2px solid var(--blue) !important; border-radius:14px !important; font-weight:700 !important;
  transition:.2s !important; }
#clear-btn:hover{ background:var(--blue) !important; color:#fff !important; transform:translateY(-3px); }

.ex-title{ margin:14px 0 4px; font-weight:700; color:var(--blue-dark); font-size:.9rem; }
.ex-btn{ background:var(--blue-soft) !important; color:var(--blue-dark) !important;
  border:1px solid var(--blue) !important; border-radius:999px !important; font-weight:600 !important;
  font-size:.85rem !important; transition:.2s !important; }
.ex-btn:hover{ background:var(--blue) !important; color:#fff !important; transform:translateY(-2px) scale(1.03); }

/* ---------- result card ---------- */
.res-card{ background:#fff; border-radius:22px; padding:24px; border-top:6px solid var(--blue);
  box-shadow:0 14px 34px rgba(63,95,134,.15); animation:fadeUp .5s ease both; color:var(--ink); }
.res-label{ color:var(--muted); font-size:.8rem; letter-spacing:.1em; text-transform:uppercase; font-weight:700; }
.top-cat{ font-size:1.9rem; font-weight:800; color:var(--wine); margin:6px 0 2px; animation:pop .55s ease both; }
.top-conf{ font-size:1rem; color:var(--blue-dark); font-weight:700; margin-bottom:12px; }
.badge{ display:inline-block; padding:7px 14px; border-radius:12px; font-size:.88rem; font-weight:700; margin-bottom:18px; }
.badge.ok{ background:rgba(113,149,188,.2); color:#2F4A6B; }
.badge.warn{ background:rgba(101,0,20,.1); color:var(--wine); animation:glow 1.8s infinite; }
.bar-row{ margin:12px 0; animation:fadeUp .5s ease both; }
.bar-head{ display:flex; justify-content:space-between; font-size:.92rem; margin-bottom:5px; color:var(--ink); }
.bar-head b{ color:var(--blue-dark); }
.track{ height:11px; background:#E8EEF6; border-radius:999px; overflow:hidden; }
.fill{ height:100%; width:var(--w); border-radius:999px; background:var(--blue);
  animation:grow 1s cubic-bezier(.2,.8,.2,1) both; }
.fill.top{ background:linear-gradient(90deg,var(--wine),var(--wine-light)); }

.empty{ text-align:center; padding:46px 20px; background:#fff; border-radius:22px;
  border:2px dashed var(--blue); color:var(--muted); animation:fadeUp .8s .2s ease both; }
.empty .ico{ font-size:3rem; display:block; animation:bounce 2.2s ease-in-out infinite; }
.empty b{ color:var(--wine); }

.footer{ text-align:center; color:var(--muted); font-size:.82rem; margin:22px 0 6px; }
.footer b{ color:var(--wine); }

/* ---------- colour overrides (beat Gradio's default text colours) ---------- */
.gradio-container .hero{ padding:clamp(28px,4vw,52px) clamp(24px,4vw,56px); }
.gradio-container .hero h1{ font-size:clamp(2rem,3.6vw,3.2rem); }
.gradio-container .hero p{ max-width:720px; font-size:clamp(1rem,1.3vw,1.15rem); }
.gradio-container .hero .hero-badge{ color:#DCE8F6 !important; }
.gradio-container .hero h1, .gradio-container .hero h1 *{ color:#fff !important; }
.gradio-container .hero h1 span{ color:#B9D0EA !important; }
.gradio-container .hero p{ color:#F1DDE2 !important; }
.gradio-container .hero .chips span{ color:#fff !important; }

.gradio-container .res-label{ color:#5B6678 !important; }
.gradio-container .top-cat{ color:#650014 !important; }
.gradio-container .top-conf{ color:#3F5F86 !important; }
.gradio-container .badge.ok{ color:#2F4A6B !important; }
.gradio-container .badge.warn{ color:#650014 !important; }
.gradio-container .bar-head span{ color:#1D2330 !important; }
.gradio-container .bar-head b{ color:#3F5F86 !important; }
.gradio-container .ex-title{ color:#3F5F86 !important; }
.gradio-container .footer, .gradio-container .empty p{ color:#5B6678 !important; }
.gradio-container .footer b, .gradio-container .empty b{ color:#650014 !important; }

#complaint-box, #complaint-box .block, #complaint-box label{
  background:transparent !important; box-shadow:none !important; border:none !important; }

@media (max-width:700px){ .hero{padding:26px 20px} .hero h1{font-size:1.7rem} }
</style>
"""

HERO = """
<div class="hero">
  <div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div>
  <div class="hero-badge">NLP · DistilBERT · 5 CATEGORIES</div>
  <h1>Consumer Complaint <span>Classifier</span></h1>
  <p>Paste a customer complaint and get the predicted category with a confidence score,
     so it can be routed to the right team automatically.</p>
  <div class="chips">
    <span>🎯 85.3% accuracy</span><span>📊 0.843 macro F1</span><span>⚡ Instant routing</span>
  </div>
</div>
"""

EMPTY_HTML = """
<div class="empty">
  <span class="ico">📨</span>
  <p><b>No complaint yet.</b><br>Type or paste one on the left, or try an example,<br>then press <b>Classify</b>.</p>
</div>
"""

FOOTER = '<div class="footer">Fine-tuned <b>DistilBERT</b> on ~124k CFPB consumer complaints</div>'


# ------------------------------------------------------------------
# 4) Result rendering
# ------------------------------------------------------------------
def render_result(scores: dict) -> str:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_p = ranked[0]

    if top_p >= THRESHOLD:
        badge = f'<span class="badge ok">✓ Route to the {PRETTY[top_label]} team</span>'
    else:
        badge = '<span class="badge warn">⚠ Low confidence — suggest manual review</span>'

    rows = ""
    for i, (label, p) in enumerate(ranked):
        fill_cls = "fill top" if i == 0 else "fill"
        rows += (
            f'<div class="bar-row" style="animation-delay:{i * 0.08:.2f}s">'
            f'<div class="bar-head"><span>{ICONS[label]} {PRETTY[label]}</span>'
            f"<b>{p * 100:.1f}%</b></div>"
            f'<div class="track"><div class="{fill_cls}" '
            f'style="--w:{p * 100:.1f}%; animation-delay:{0.15 + i * 0.1:.2f}s"></div></div>'
            f"</div>"
        )

    return (
        '<div class="res-card">'
        '<div class="res-label">Predicted category</div>'
        f'<div class="top-cat">{ICONS[top_label]} {PRETTY[top_label]}</div>'
        f'<div class="top-conf">{top_p * 100:.1f}% confidence</div>'
        f"{badge}{rows}</div>"
    )


def classify(text):
    if not text or not text.strip():
        return EMPTY_HTML
    return render_result(predict_scores(text))


# 5) UI
try:  
    blocks = gr.Blocks(title="Consumer Complaint Classifier", fill_width=True)
except TypeError:
    blocks = gr.Blocks(title="Consumer Complaint Classifier")

with blocks as demo:
    gr.HTML(CSS + HERO)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            box = gr.Textbox(
                label="Complaint narrative",
                lines=9,
                placeholder="Type or paste a consumer complaint here...",
                elem_id="complaint-box",
            )
            with gr.Row():
                clear_btn = gr.Button("Clear", elem_id="clear-btn")
                submit_btn = gr.Button("Classify", elem_id="submit-btn", variant="primary")

            gr.HTML('<div class="ex-title">Try an example</div>')
            with gr.Row():
                ex_buttons = [
                    gr.Button(name, elem_classes=["ex-btn"], size="sm") for name, _ in EXAMPLES
                ]

        with gr.Column(scale=5):
            result = gr.HTML(EMPTY_HTML)

    gr.HTML(FOOTER)

    submit_btn.click(classify, inputs=box, outputs=result)
    clear_btn.click(lambda: ("", EMPTY_HTML), inputs=None, outputs=[box, result])
    for btn, (_, sample) in zip(ex_buttons, EXAMPLES):
        btn.click(lambda s=sample: s, inputs=None, outputs=box)

if __name__ == "__main__":
    demo.launch(inbrowser=True)