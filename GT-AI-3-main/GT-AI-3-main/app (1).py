"""Run with: streamlit run app.py"""

from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="5G Performance Explorer", page_icon="📶", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(ROOT / "model.pkl")
    scaler = joblib.load(ROOT / "scaler.pkl")
    metadata = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
    if list(scaler.feature_names_in_) != metadata["features"]:
        raise ValueError("The scaler and metadata feature schemas do not match.")
    return model, scaler, metadata


required_files = ["model.pkl", "scaler.pkl", "metrics.json"]
missing_files = [name for name in required_files if not (ROOT / name).is_file()]
if missing_files:
    st.error("The deployed project is missing required files: " + ", ".join(missing_files))
    st.info("Add the complete deployment package to the same repository directory as app.py, preserving the assets folder, then redeploy.")
    st.stop()

model, scaler, metrics = load_artifacts()
cover_path = ROOT / "assets" / "network_cover.png"
if not cover_path.is_file():
    cover_path = ROOT / "network_cover.png"

st.markdown(
    """<style>
    .block-container { max-width: 1150px; padding-top: 2.2rem; }
    section[data-testid="stSidebar"] {
        background-color: #e6f1ee !important;
        border-right: 1px solid #c8dfda;
    }
    section[data-testid="stSidebar"] > div {
        background-color: #e6f1ee !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] small {
        color: #14323e !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #44636d !important;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: #d3e9e4;
        border-radius: 8px;
    }
    </style>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("5G Performance")
    page = st.radio("Navigate", ["Cover", "Make a prediction", "Dataset and method"])
    st.caption("Educational project using a derived target")


if page == "Cover":
    if cover_path.is_file():
        st.image(str(cover_path), use_container_width=True)
    else:
        st.warning("Cover image missing. Add network_cover.png beside app.py or in assets/.")
    st.title("5G network performance explorer")
    st.write(
        "Explore a transparent performance tier computed from signal strength, "
        "download and upload speed, latency, and jitter. Open **Make a prediction** "
        "from the sidebar to try a measurement set."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", "50,000")
    col2.metric("Held-out accuracy", f"{metrics['test_accuracy']:.1%}")
    col3.metric("Majority baseline", f"{metrics['baseline_accuracy']:.1%}")
    st.info(
        "The tier is an illustrative label calculated from the same five measurements "
        "entered in the app. Accuracy measures agreement with that rule, not independent "
        "prediction of real congestion or service quality."
    )

elif page == "Make a prediction":
    st.title("Predict a performance tier")
    st.write("Enter a snapshot of five network measurements. All units match the source CSV.")
    with st.form("network_inputs"):
        left, right = st.columns(2)
        with left:
            signal = st.number_input("Signal strength (dBm)", min_value=-120.0, max_value=-50.0, value=-85.0, step=1.0)
            download = st.number_input("Download speed (Mbps)", min_value=0.0, max_value=1000.0, value=550.0, step=10.0)
            upload = st.number_input("Upload speed (Mbps)", min_value=0.0, max_value=150.0, value=85.0, step=5.0)
        with right:
            latency = st.number_input("Latency (ms)", min_value=0.0, max_value=25.0, value=10.5, step=0.5)
            jitter = st.number_input("Jitter (ms)", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
        submitted = st.form_submit_button("Predict tier", type="primary")

    if submitted:
        values = {
            "Signal Strength (dBm)": signal,
            "Download Speed (Mbps)": download,
            "Upload Speed (Mbps)": upload,
            "Latency (ms)": latency,
            "Jitter (ms)": jitter,
        }
        row = pd.DataFrame([values], columns=metrics["features"])
        transformed = scaler.transform(row)
        prediction = model.predict(transformed)[0]
        st.success(f"Predicted performance tier: **{prediction}**")
        st.caption(
            "Model output for a derived teaching label. It is not a measured congestion level "
            "and should not guide network operations."
        )
        with st.expander("Review the input and method"):
            st.dataframe(row.T.rename(columns={0: "Entered value"}), use_container_width=True)
            st.write(
                "A StandardScaler fitted on training data transforms this row. A depth-limited "
                "decision tree then predicts the tier."
            )

else:
    st.title("Dataset and method")
    st.write(
        "The supplied CSV has 50,000 rows and 21 source columns. The model uses five numeric "
        "measurements and predicts a new, explicitly derived **Performance Tier**."
    )
    st.subheader("Illustrative scoring rule")
    st.latex(
        r"s = 0.30D/1000 + 0.15U/150 + 0.25(1-L/25) + "
        r"0.15(1-J/5) + 0.15(S+120)/70"
    )
    st.caption(
        "Each individual component is clipped to [0, 1]. D is download speed, U upload speed, "
        "L latency, J jitter, and S signal strength in dBm."
    )
    st.write("Limited: score < 0.45; Standard: 0.45 ≤ score < 0.60; Strong: score ≥ 0.60.")
    st.write(
        "An 80/20 stratified split yielded "
        f"{metrics['split']['train_rows']:,} training rows and "
        f"{metrics['split']['test_rows']:,} held-out rows. "
        f"Test accuracy: **{metrics['test_accuracy']:.2%}**; "
        f"balanced accuracy: **{metrics['balanced_accuracy']:.2%}**; "
        f"majority baseline: **{metrics['baseline_accuracy']:.2%}**."
    )
    st.bar_chart(pd.Series(metrics["class_counts"], name="Rows"))
    st.warning(
        "The original Network Congestion Level column is not used. These labels are calculated "
        "from the inputs, so the held-out score demonstrates rule replication only. "
        "Independent measured labels are needed to validate a real-world predictor."
    )
