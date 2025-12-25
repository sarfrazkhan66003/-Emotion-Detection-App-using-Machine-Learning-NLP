import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_mapping = joblib.load("label_mapping.pkl")

if isinstance(label_mapping, dict):
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
else:
    inv_label_mapping = label_mapping

emojis = {
    'joy': "😄",
    'fear': "😨",
    'anger': "😡",
    'sadness': "😢",
    'surprise': "😲",
    'love': "❤️"
}

def predict_emotion(text):
    x_input = vectorizer.transform([text])
    probs = model.predict_proba(x_input)[0]
    pred_id = np.argmax(probs)
    pred_label = inv_label_mapping[pred_id]
    pred_emoji = emojis.get(pred_label, "")
    return pred_id, pred_label, pred_emoji, probs

st.markdown("""
    <style>
        .stApp { background-color: #ffffff; color: #222; }
        h1 { text-align: center; color: #2c3e50; font-weight: 800; font-size: 40px; }
        .glass-card {
            background: rgba(255, 255, 255, 0.9);
            padding: 20px; border-radius: 15px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08); margin: 15px 0;
        }
        div[data-testid="stSidebar"] {
            background: #f8f9fa;
            padding: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("ℹ️ About This App")
st.sidebar.markdown("""
This **Emotion Detection App** analyzes the text you enter and predicts the **emotion** behind it.  
It uses **Machine Learning (ML)** with Natural Language Processing (NLP).

### 🎯 Features:
- Detects **6 emotions**: Joy, Fear, Anger, Sadness, Surprise, Love  
- Shows **confidence levels** with an interactive chart  
- Emoji support for better visualization  
- Stylish & modern UI  

### 🚀 How to use:
1. Enter your sentence in the text box.  
2. Click **Predict Emotion**.  
3. See the detected emotion + probabilities.  

> Made with ❤️ using **Streamlit & Machine Learning**.
""")

st.markdown("<h1>🎭 Emotion Detector</h1>", unsafe_allow_html=True)

user_input = st.text_area("✍️ Type your text here:", "")

if st.button("✨ Predict Emotion"):
    if user_input.strip():
        pred_id, pred_label, pred_emoji, probs = predict_emotion(user_input)

        st.markdown(
            f"<div class='glass-card'><h2 style='text-align:center; font-size:28px;'>Prediction: {pred_label} {pred_emoji}</h2></div>", 
            unsafe_allow_html=True
        )

        # Confidence Data
        prob_df = pd.DataFrame({
            "Emotion": [inv_label_mapping[i] for i in range(len(probs))],
            "Confidence": probs
        }).sort_values("Confidence", ascending=True)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=prob_df["Confidence"] * 100,
            y=prob_df["Emotion"],
            orientation="h",
            marker=dict(
                color=prob_df["Confidence"] * 100,
                colorscale="Blues",
                line=dict(color="rgba(0,0,0,0.1)", width=1),
            ),
            text=[f"{c*100:.1f}%" for c in prob_df["Confidence"]],
            textposition="outside",
            textfont=dict(size=18, color="darkblue", family="Arial Black"),
            hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
        ))

        fig.update_layout(
            title="Confidence Levels",
            title_x=0.5,
            title_font=dict(size=24, color="black"),
            xaxis=dict(title="%", showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=16, color="#2c3e50")),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=80, r=40, t=60, b=40),
            height=450,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Confidence Table
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.dataframe(
            prob_df.sort_values("Confidence", ascending=False)
            .style.set_properties(**{"font-size": "16px", "color": "#2c3e50"})
            .background_gradient(cmap="Blues")
        )
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please enter some text!")
