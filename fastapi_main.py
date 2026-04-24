import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
import tensorflow as tf
import plotly.graph_objs as go
from feature_extractor import extract_mel_spectrogram
import librosa

# ---------- MODEL ----------
MODEL_PATH = "models/best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/samples", StaticFiles(directory="samples"), name="samples")


# ---------- UTILS ----------
def get_waveform_plot(audio_path, sr=8000):
    y, _ = librosa.load(audio_path, sr=sr)
    x = np.arange(len(y)) / sr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))
    fig.update_layout(height=220, template="plotly_white", margin=dict(l=10,r=10,t=10,b=10))
    return fig.to_html(full_html=False)


def predict_logic(file_path):
    x = extract_mel_spectrogram(file_path)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)[0]
    top_indices = preds.argsort()[-3:][::-1]
    top_values = preds[top_indices]
    top_results = [(int(i), float(v)*100) for i, v in zip(top_indices, top_values)]

    predicted_class = top_results[0][0]
    confidence = top_results[0][1]
    return predicted_class, confidence, top_results


def generate_result_html(file_path, filename):
    predicted_class, confidence, top_results = predict_logic(file_path)

    top_html = "".join([
        f"<div class='top-item'><span>{digit}</span><span>{conf:.2f}%</span></div>"
        for digit, conf in top_results
    ])

    waveform_div = get_waveform_plot(file_path)

    return f"""
    <html>
    <head>
        <style>
            body {{
                margin:0;
                font-family: 'Segoe UI', sans-serif;
                background: radial-gradient(circle at top, #0f0f1a, #05050a);
                color:white;
                display:flex;
                justify-content:center;
                align-items:center;
                height:100vh;
            }}
            .card {{
                width:420px;
                padding:30px;
                border-radius:20px;
                backdrop-filter: blur(16px);
                background: rgba(255,255,255,0.05);
                border:1px solid rgba(255,255,255,0.1);
                box-shadow: 0 20px 60px rgba(0,0,0,0.7);
                animation: fadeIn 0.6s ease;
            }}
            @keyframes fadeIn {{
                from {{ opacity:0; transform: translateY(20px); }}
                to {{ opacity:1; transform: translateY(0); }}
            }}
            .top-item {{
                display:flex;
                justify-content:space-between;
                padding:6px 0;
                border-bottom:1px solid rgba(255,255,255,0.1);
            }}
            a {{
                color:#6c63ff;
                text-decoration:none;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Prediction</h2>
            <h3>Digit: {predicted_class}</h3>
            <h4>Confidence: {confidence:.2f}%</h4>

            <h3>Top Predictions</h3>
            {top_html}

            <audio controls>
                <source src="/{filename}">
            </audio>

            <div>{waveform_div}</div>

            <br><a href="/">← Back</a>
        </div>
    </body>
    </html>
    """


# ---------- HOME UI ----------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Munaf AI</title>
<meta name="viewport" content="width=device-width, initial-scale=1">

<style>
*{margin:0;padding:0;box-sizing:border-box;font-family:'Segoe UI';scroll-behavior:smooth}

body{
    background:#05050a;
    color:white;
    overflow-x:hidden;
}

/* PARALLAX BACKGROUND */
.bg{
    position:fixed;
    width:100%;
    height:100%;
    background: radial-gradient(circle at 20% 20%, #1a1a2e, transparent),
                radial-gradient(circle at 80% 80%, #0f3460, transparent);
    z-index:-1;
}

/* HEADER */
header{
    position:fixed;
    width:100%;
    padding:20px;
    backdrop-filter: blur(10px);
    background:rgba(0,0,0,0.3);
}

/* SECTION */
section{
    min-height:100vh;
    display:flex;
    align-items:center;
    justify-content:center;
    flex-direction:column;
}

/* GLASS CARD */
.card{
    backdrop-filter: blur(15px);
    background: rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.1);
    padding:30px;
    border-radius:20px;
    width:320px;
    text-align:center;
    transition:0.4s;
}

.card:hover{
    transform:translateY(-10px) scale(1.02);
}

/* BUTTONS */
button,input[type="submit"]{
    margin-top:10px;
    padding:10px 15px;
    border:none;
    border-radius:8px;
    background:linear-gradient(135deg,#6c63ff,#00c6ff);
    color:white;
    cursor:pointer;
    transition:0.3s;
}

button:hover{
    transform:scale(1.1);
    box-shadow:0 0 20px rgba(108,99,255,0.8);
}

/* DIGITS */
.digits button{
    width:45px;
    height:45px;
    margin:5px;
}

/* SCROLL ANIMATION */
.reveal{
    opacity:0;
    transform:translateY(40px);
    transition:1s;
}
.reveal.active{
    opacity:1;
    transform:translateY(0);
}

</style>
</head>

<body>

<div class="bg"></div>

<header>
<h3>Munaf AI</h3>
</header>

<section>
<h1 class="reveal">🎤 AI Voice Recognition</h1>
</section>

<section>
<div class="card reveal">
<h3>Upload Audio</h3>
<form action="/predict" method="post" enctype="multipart/form-data">
<input type="file" name="file" required><br>
<input type="submit" value="Predict">
</form>
</div>
</section>

<section>
<div class="card reveal">
<h3>Quick Demo</h3>
<div class="digits">
<a href="/sample/0"><button>0</button></a>
<a href="/sample/1"><button>1</button></a>
<a href="/sample/2"><button>2</button></a>
<a href="/sample/3"><button>3</button></a>
<a href="/sample/4"><button>4</button></a>
<a href="/sample/5"><button>5</button></a>
<a href="/sample/6"><button>6</button></a>
<a href="/sample/7"><button>7</button></a>
<a href="/sample/8"><button>8</button></a>
<a href="/sample/9"><button>9</button></a>
</div>
</div>
</section>

<script>
// SCROLL REVEAL
function reveal(){
    const reveals=document.querySelectorAll('.reveal');
    for(let i=0;i<reveals.length;i++){
        let windowHeight=window.innerHeight;
        let elementTop=reveals[i].getBoundingClientRect().top;
        if(elementTop<windowHeight-100){
            reveals[i].classList.add('active');
        }
    }
}
window.addEventListener('scroll',reveal);
</script>

</body>
</html>
"""


# ---------- ROUTES ----------
@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return generate_result_html(file_path, f"uploads/{file.filename}")


@app.get("/sample/{digit}", response_class=HTMLResponse)
def sample_test(digit: int):
    file_path = f"samples/{digit}.wav"
    return generate_result_html(file_path, f"samples/{digit}.wav")


# ---------- RUN ----------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)