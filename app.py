import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# -----------------------------------
# HEADER SECTION
# -----------------------------------
st.title("🚢 Titanic Survival Prediction System")

st.subheader(
    "Deep Learning Based Passenger Survival Prediction"
)

st.markdown("---")

# -----------------------------------
# PROJECT DESCRIPTION
# -----------------------------------
st.markdown("""
## 📘 Project Description

This application predicts passenger survival
using a manually implemented Artificial Neural Network.

### Technologies Used
- Artificial Neural Network
- Python
- Streamlit
- NumPy

The ANN uses:
- Passenger Class
- Age
- Fare

to predict survival probability.
""")

st.markdown("---")

# -----------------------------------
# MANUAL WEIGHTS & BIASES
# -----------------------------------

# input -> hidden weights
w1 = 0.1101
w2 = 0.1402
w3 = 0.1705

w4 = 0.2101
w5 = 0.2402
w6 = 0.2706

# hidden -> output weights
w7 = 0.3152
w8 = 0.3454

# biases
h1 = 0.1007
h2 = 0.1007
h3 = 0.1090

# -----------------------------------
# SIGMOID FUNCTION
# -----------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -----------------------------------
# PREDICTION FUNCTION
# -----------------------------------
def predict(x1, x2, x3):

    # hidden neuron 1
    zh1 = (x1 * w1) + (x2 * w2) + (x3 * w3) + h1
    ah1 = sigmoid(zh1)

    # hidden neuron 2
    zh2 = (x1 * w4) + (x2 * w5) + (x3 * w6) + h2
    ah2 = sigmoid(zh2)

    # output neuron
    zh3 = (ah1 * w7) + (ah2 * w8) + h3
    ah3 = sigmoid(zh3)

    return ah3

# -----------------------------------
# INPUT SECTION
# -----------------------------------
st.header("🧾 Passenger Information")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        24
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=120.0
    )

st.markdown("---")

# -----------------------------------
# NORMALIZATION
# -----------------------------------

# same normalization used earlier

pclass_norm = (pclass - 1) / (3 - 1)

age_norm = age / 80

fare_norm = fare / 600

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------
if st.button("Predict Survival"):

    probability = predict(
        pclass_norm,
        age_norm,
        fare_norm
    )

    probability = float(probability)

    # prediction logic
    if probability > 0.5:
        result = "✅ Survived"
    else:
        result = "❌ Not Survived"

    # -----------------------------------
    # OUTPUT SECTION
    # -----------------------------------
    st.header("📊 Prediction Result")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Prediction",
            result
        )

    with c2:
        st.metric(
            "Survival Probability",
            f"{probability:.4f}"
        )

    with c3:
        confidence = max(
            probability,
            1 - probability
        )

        st.metric(
            "Confidence Score",
            f"{confidence:.4f}"
        )

    st.markdown("---")

    # -----------------------------------
    # VISUALIZATION
    # -----------------------------------
    st.header("📈 Probability Visualization")

    survive_prob = probability
    nonsurvive_prob = 1 - probability

    labels = [
        "Survived",
        "Not Survived"
    ]

    values = [
        survive_prob,
        nonsurvive_prob
    ]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    ax.set_ylabel("Probability")

    st.pyplot(fig)

st.markdown("---")

st.success("Manual ANN Deployment Successful 🚀")