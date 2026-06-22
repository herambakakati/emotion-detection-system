import streamlit as st
import tensorflow as tf

from transformers import (
    DistilBertTokenizer,
    TFDistilBertForSequenceClassification
)
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="EmotionSense AI",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>            
/* ==========================================
   REMOVE TOP SPACE
========================================== */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.block-container{

padding-top:0rem !important;

padding-bottom:1rem !important;

max-width:95%;

}

div[data-testid="stVerticalBlock"]{

gap:0.4rem;

}
            
/* ==========================
   PREMIUM INPUT CARD
========================== */
.input-card{
margin-bottom:2px !important;
}

.stTextArea{
margin-top:-2px !important;
margin-bottom:0px !important;
}
            
.input-card{

padding:25px 35px;

border-radius:28px;

background:
linear-gradient(
135deg,
rgba(255,255,255,0.04),
rgba(0,245,255,0.03),
rgba(139,92,246,0.03)
);

border:1px solid rgba(0,245,255,0.18);

backdrop-filter:blur(20px);

box-shadow:
0 0 35px rgba(0,245,255,0.08);

margin-bottom:10px;
}

.input-title{
display:flex;
align-items:center;
gap:12px;
font-size:38px;
font-weight:800;
margin-bottom:8px;

background:
linear-gradient(
90deg,
#00F5FF,
#39FF14
);

-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.input-line{
display:flex;
align-items:center;
gap:12px;
margin-bottom:3px;
font-size:18px;
line-height:1.4;
color:#f8fafc;
}

.input-line:last-child{
margin-bottom:0px;
}

.input-icon{

font-size:22px;
min-width:28px;
display:flex;
align-items:center;

justify-content:center;
}

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.stApp{
background:
radial-gradient(circle at top left,#00F5FF22,transparent 30%),
radial-gradient(circle at top right,#00FF8822,transparent 30%),
radial-gradient(circle at bottom left,#FFD70022,transparent 30%),
linear-gradient(
135deg,
#020617,
#0f172a,
#111827
);
}

/* Hero Banner */
.hero{
padding:20px 45px;
border-radius:30px;

background:
linear-gradient(
135deg,
rgba(0,245,255,0.12),
rgba(139,92,246,0.12),
rgba(0,255,136,0.12)
);

backdrop-filter:blur(20px);

border:1px solid rgba(255,255,255,0.10);

text-align:center;

margin-bottom:20px;

box-shadow:
0px 10px 50px rgba(0,245,255,0.15),
0px 10px 50px rgba(139,92,246,0.15);
}

.hero-title{
font-size:78px;
font-weight:900;
line-height:1;
letter-spacing:-2px;
background:
linear-gradient(
90deg,
#39FF14 0%,
#A3FF12 40%,
#FFD700 100%
);

-webkit-background-clip:text;
-webkit-text-fill-color:transparent;

text-shadow:
0 0 20px rgba(57,255,20,0.15);

margin:0;
}
            

.hero-sub{

font-size:20px;
font-weight:700;
color:#f8fafc;

margin-top:-2px;
margin-bottom:4px;

}

.hero-desc{
max-width:1000px;
margin:0 auto;
font-size:13px;
line-height:1.35;
color:#dbe4ee;
padding-top:0px;
margin-top:0px;
}
            

/* Glass Card */

.glass{
padding:22px 28px;

background:
linear-gradient(
135deg,
rgba(255,255,255,0.05),
rgba(0,245,255,0.04),
rgba(139,92,246,0.04)
);

backdrop-filter:blur(20px);

padding:30px;
border-radius:28px;
border:1px solid rgba(255,255,255,0.10);
box-shadow:
0px 10px 40px rgba(0,245,255,0.08);
margin-bottom:15px;

}

/* Text Area */
.stTextArea textarea{

background:
linear-gradient(
135deg,
rgba(255,255,255,0.05),
rgba(0,245,255,0.04),
rgba(139,92,246,0.04)
) !important;

color:white !important;

border-radius:22px !important;

border:1px solid rgba(0,245,255,0.18) !important;

padding:18px !important;

font-size:18px !important;

line-height:1.8 !important;

box-shadow:
0px 8px 25px rgba(0,245,255,0.08);

}
.stTextArea textarea:focus{

border:1px solid #00F5FF !important;
box-shadow:
0px 0px 25px rgba(0,245,255,0.30) !important;

}

/* Button */

.stButton > button{

width:100%;

height:65px;

border:none;

border-radius:20px;

font-size:22px;

font-weight:800;

background:
linear-gradient(
90deg,
#00F5FF,
#8B5CF6,
#00FF88
);

color:white;

transition:0.4s;

box-shadow:
0px 10px 30px rgba(0,245,255,0.20);

}

.stButton > button:hover{

transform:translateY(-3px);

box-shadow:
0px 15px 40px rgba(0,245,255,0.40);

}


/* Footer */

.footer{

text-align:center;

padding:25px;

color:#cbd5e1;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# NLTK
# =====================================================

try:
    stopwords.words("english")
except:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except:
    nltk.download("wordnet")

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    tokenizer = DistilBertTokenizer.from_pretrained(
        "./distilbert_emotion_model",
        local_files_only=True
    )

    model = TFDistilBertForSequenceClassification.from_pretrained(
        "distilbert_emotion_model"
    )

    return tokenizer, model


tokenizer, model = load_model()

def predict_emotion(text):

    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="tf"
    )

    outputs = model(encoded)

    prediction = int(
        tf.argmax(
            outputs.logits,
            axis=1
        ).numpy()[0]
    )

    return prediction

# =====================================================
# NLP PREPROCESSING
# =====================================================

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stop_words = set(
    stopwords.words("english")
)

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z]",
        " ",
        text
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(
            ps.stem(word)
        )
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# =====================================================
# LABEL MAP
# =====================================================

emotion_map = {

    0:"Anger 😠",
    1:"Fear 😨",
    2:"Joy 😄",
    3:"Love ❤️",
    4:"Sadness 😢",
    5:"Surprise 😲"

}

emotion_colors = {

    "Anger 😠":"#ef4444",
    "Fear 😨":"#f59e0b",
    "Joy 😄":"#22c55e",
    "Love ❤️":"#ec4899",
    "Sadness 😢":"#3b82f6",
    "Surprise 😲":"#8b5cf6"

}

# =====================================================
# HERO SECTION
# =====================================================
st.markdown("""
<div class='hero'>

<div style="
max-width:1000px;
margin:auto;
text-align:center;
">

<div style="
display:flex;
align-items:center;
justify-content:center;
gap:22px;
margin-bottom:2px;
">

<img
src="https://cdn-icons-png.flaticon.com/512/2103/2103633.png"
style="
height:95px;
width:auto;
filter:
drop-shadow(0 0 8px rgba(57,255,20,0.5))
drop-shadow(0 0 20px rgba(57,255,20,0.3));
"
/>

<div style="
display:flex;
flex-direction:column;
justify-content:center;
align-items:flex-start;
">

<div class="hero-title">
EmotionSense AI
</div>

<div style="
display:inline-flex;
align-items:center;
justify-content:center;
width:fit-content;
padding:5px 14px;
border-radius:20px;
background:rgba(57,255,20,0.08);
border:1px solid rgba(57,255,20,0.18);
color:#A3FF12;
font-size:10px;
font-weight:700;
letter-spacing:1px;
line-height:1;
margin-top:-4px;
margin-bottom:-6px;
box-shadow:0 0 12px rgba(57,255,20,0.08);
">
NLP • ML • DL • TRANSFORMERS
</div>
</div>

</div>

<div class='hero-sub'>
Understand Emotions Through Artificial Intelligence
</div>
      
<div class='hero-desc'>
Discover emotions hidden in text through advanced NLP, Machine Learning, Deep Learning, and Transformer-based AI technologies.
</div>

</div>

</div>
""", unsafe_allow_html=True)
# =====================================================
# INPUT SECTION
# =====================================================

st.markdown("""
<div class="input-card">

<div class="input-title">
<span>💭</span>
<span>What's On Your Mind?</span>
</div>

<div class="input-line">
<span class="input-icon">💬</span>
<span>Share your thoughts, feelings, reviews, experiences, opinions, or personal stories.</span>
</div>

<div class="input-line">
<span class="input-icon">🧠</span>
<span>EmotionSense AI will intelligently analyze your text and identify the underlying emotion.</span>
</div>

</div>
""", unsafe_allow_html=True)

user_text = st.text_area(
    "",
    height=180,
    placeholder="Example: I am feeling very excited about my new opportunity..."
)

# =====================================================
# PREDICT
# =====================================================

if st.button("🚀 Analyze Emotion"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")

    else:

        prediction = predict_emotion(user_text)
        emotion = emotion_map[prediction]
        color = emotion_colors[emotion]

        emoji_map = {
            "Anger 😠":"😠",
            "Fear 😨":"😨",
            "Joy 😄":"😄",
            "Love ❤️":"❤️",
            "Sadness 😢":"😢",
            "Surprise 😲":"😲"
        }

        emotion_name = emotion.rsplit(" ",1)[0]

        st.markdown(
            f"""
        <div style="
        padding:18px 25px;
        border-radius:30px;
        text-align:center;
        margin-top:20px;
        background:{color};
        border:3px solid {color};
        box-shadow:0 0 25px {color},0 0 60px {color}55;
        ">

        <h2 style="
        color:white;
        font-size:30px;
        font-weight:800;
        margin-bottom:6px;
        ">
        🎯 Emotion Detected
        </h2>

        <div style="
        font-size:65px;
        margin-bottom:4px;
        ">
        {emoji_map[emotion]}
        </div>

        <div style="
        font-size:36px;
        font-weight:900;
        color:white;
        margin-bottom:6px;
        ">
        {emotion_name}
        </div>

        <div style="
        color:#f8fafc;
        font-size:16px;
        line-height:1.4;
        margin-top:0px;
        ">
        EmotionSense AI successfully analyzed your text and detected the underlying emotion.
        </div>
        </div>
        """,
            unsafe_allow_html=True
        )        

    
# =====================================================
# PROJECT DETAILS
# =====================================================

st.markdown("""
<div style="
padding:30px;
border-radius:35px;
margin-top:20px;
background:linear-gradient(
135deg,
rgba(4,15,35,0.95),
rgba(10,25,47,0.95),
rgba(15,23,42,0.95)
);
border:1px solid rgba(0,245,255,0.15);
box-shadow:
0 0 40px rgba(0,245,255,0.10),
0 0 80px rgba(139,92,246,0.10);
">

<h1 style="
font-size:52px;
font-weight:900;
margin-bottom:10px;
background:linear-gradient(90deg,#00F5FF,#39FF14);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
">
📌 About The Project
</h1>

<div style="
display:grid;
grid-template-columns:1fr 1fr;
gap:15px;
margin-bottom:25px;
align-items:start;
">

<!-- CARD 1 -->

<div style="
padding:25px;
border-radius:22px;
background:linear-gradient(
135deg,
rgba(0,245,255,0.08),
rgba(0,245,255,0.03)
);
border:1px solid rgba(0,245,255,0.18);
height:100%;
display:flex;
flex-direction:column;
">

<div style="
font-size:24px;
font-weight:800;
color:#00F5FF;
margin-bottom:15px;
">
🧠 Emotion Intelligence
</div>

<div style="
color:#e2e8f0;
font-size:16px;
line-height:1.8;
text-align:justify;
flex-grow:1;
">

EmotionSense AI is an advanced emotion classification system developed using
Natural Language Processing (NLP), Machine Learning (ML), Deep Learning (DL),
and Transformer-based architectures.

The system classifies textual content into six emotions:

• Joy 😄
• Sadness 😢
• Anger 😠
• Fear 😨
• Love ❤️
• Surprise 😲

The project includes Machine Learning models, Deep Learning architectures,
and state-of-the-art Transformer models such as BERT, DistilBERT, and RoBERTa.

Transformer models achieved the highest performance, reaching over 92%
classification accuracy on the evaluation dataset.
            
The deployed application uses DistilBERT, a state-of-the-art Transformer-based Deep Learning model that achieved 92.60% classification accuracy. DistilBERT provides strong contextual language understanding while maintaining efficient inference speed, making it suitable for real-time emotion detection.
</div>
            
</div>

<!-- CARD 2 -->

<div style="
padding:25px;
border-radius:22px;
background:linear-gradient(
135deg,
rgba(139,92,246,0.08),
rgba(139,92,246,0.03)
);
border:1px solid rgba(139,92,246,0.18);
height:100%;
display:flex;
flex-direction:column;
">

<div style="
font-size:24px;
font-weight:800;
color:#8B5CF6;
margin-bottom:15px;
">
⚡ Advanced AI Architecture
</div>

<div style="
color:#e2e8f0;
font-size:16px;
line-height:1.8;
text-align:justify;
flex-grow:1;
">

EmotionSense AI combines traditional Machine Learning, Deep Learning,
and Transformer-based architectures to perform intelligent emotion analysis.

The system applies advanced NLP preprocessing techniques and leverages
multiple modeling approaches including RNN, LSTM, GRU, BERT,
DistilBERT, and RoBERTa.

For deployment, DistilBERT was selected as the final model due to its high accuracy and efficient inference performance.

By learning contextual and semantic relationships within textual data,
the platform delivers accurate emotion detection across diverse text inputs,
making it suitable for real-world sentiment and emotion understanding tasks.
</div>
</div>
</div>
           

<h3 style="
color:#00F5FF;
margin-top:25px;
margin-bottom:18px;
font-size:32px;
font-weight:900;
background:linear-gradient(
90deg,
#00F5FF,
#39FF14
);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
">
🚀 Potential Use Cases
</h3>

<div style="
display:grid;
grid-template-columns:1fr 1fr;
gap:10px;
">

<div style="
padding:18px;
border-radius:18px;
background:rgba(0,245,255,0.05);
border:1px solid rgba(0,245,255,0.15);
">
✅ Customer Feedback Analysis
</div>

<div style="
padding:18px;
border-radius:18px;
background:rgba(139,92,246,0.05);
border:1px solid rgba(139,92,246,0.15);
">
📱 Social Media Monitoring
</div>

<div style="
padding:18px;
border-radius:18px;
background:rgba(34,197,94,0.05);
border:1px solid rgba(34,197,94,0.15);
">
⭐ Product Review Analytics
</div>

<div style="
padding:18px;
border-radius:18px;
background:rgba(236,72,153,0.05);
border:1px solid rgba(236,72,153,0.15);
">
💙 Mental Wellness Applications
</div>

<div style="
padding:18px;
border-radius:18px;
background:rgba(6,182,212,0.05);
border:1px solid rgba(6,182,212,0.15);
">
🤖 Intelligent Chat Systems
</div>

<div style="
padding:18px;
border-radius:18px;
background:rgba(168,85,247,0.05);
border:1px solid rgba(168,85,247,0.15);
">
📊 User Sentiment Understanding
</div>

</div>

<div style='
margin-top:20px;
padding:28px;
border-radius:24px;
background:linear-gradient(135deg,rgba(0,245,255,0.06),rgba(139,92,246,0.06));
border:1px solid rgba(255,255,255,0.10);
box-shadow:0 0 25px rgba(0,245,255,0.08);
'>

<h3 style='
margin:0 0 15px 0;
font-size:30px;
font-weight:800;
background:linear-gradient(90deg,#00F5FF,#39FF14);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
'>
🛠 Technology Stack
</h3>

<div style='
display:grid;
grid-template-columns:1fr 1fr;
gap:12px;
margin-top:15px;
'>

<div style='padding:12px;border-radius:12px;background:rgba(255,255,255,0.04);'>
🔹 Natural Language Processing
</div>

<div style='padding:12px;border-radius:12px;background:rgba(255,255,255,0.04);'>
🔹 TF-IDF Vectorization
</div>

<div style='padding:12px;border-radius:12px;background:rgba(255,255,255,0.04);'>
🔹 Machine Learning
</div>

<div style='padding:12px;border-radius:12px;background:rgba(255,255,255,0.04);'>
🔹 Deep Learning
</div>

<div style='padding:12px;border-radius:12px;background:rgba(255,255,255,0.04);'>
🔹 BERT Transformer
</div>

<div style='padding:12px;border-radius:12px;background:rgba(255,255,255,0.04);'>
🔹 DistilBERT Transformer
</div>

<div style='padding:12px;border-radius:12px;background:rgba(255,255,255,0.04);'>
🔹 RoBERTa Transformer
</div>

<div style='padding:12px;border-radius:12px;background:rgba(255,255,255,0.04);'>
🔹 TensorFlow & Keras
</div>

<div style='padding:12px;border-radius:12px;background:rgba(255,255,255,0.04);'>
🔹 Scikit-Learn
</div>

<div style='padding:12px;border-radius:12px;background:rgba(255,255,255,0.04);'>
🔹 Streamlit Deployment
</div>

</div>

<hr style='
margin:20px 0;
border:none;
height:1px;
background:rgba(255,255,255,0.10);
'>

<div style='
font-size:16px;
line-height:1.8;
color:#dbe4ee;
'>
🚀 EmotionSense AI leverages Natural Language Processing, Machine Learning, Deep Learning, and Transformer architectures to convert textual data into meaningful emotional insights, enabling accurate understanding of human sentiments and behavioral patterns.
</div>

</div>

""", unsafe_allow_html=True)    


# =====================================================
# PREMIUM FOOTER
# =====================================================

st.markdown("""
<div style='
margin-top:10px;
padding:12px 10px;
text-align:center;
border-top:1px solid rgba(255,255,255,0.08);
line-height:1.2;
'>

<div style='
font-size:12px;
font-weight:900;
letter-spacing:2px;
background:linear-gradient(90deg,#00F5FF,#00FF88,#39FF14);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
display:inline-block;
margin-bottom:0px;
'>
HERAMBA KAKATI
</div>

<div style='
font-size:12px;
color:#cbd5e1;
margin-top:0px;
font-weight:500;
'>
AI-Powered Emotion Intelligence • NLP • ML • DL • Transformers
</div>

<div style='
font-size:10px;
color:#94a3b8;
margin-top:1px;
'>
Transforming Text into Intelligent Emotional Insights Through AI
</div>

<div style='
font-size:10px;
color:#64748b;
margin-top:2px;
'>
© 2026 EmotionSense AI
</div>

</div>
""", unsafe_allow_html=True)