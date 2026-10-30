
import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Steel Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# 2. INTERACTIVE LIGHT / DARK DASHBOARD THEME
# --------------------------------------------------
# Streamlit reruns the app when the user changes this setting.
# The selected theme is applied to the main page and sidebar.
if "dashboard_theme" not in st.session_state:
    st.session_state["dashboard_theme"] = "Light"

theme = st.sidebar.selectbox(
    "🎨 Dashboard appearance",
    options=["Light", "Dark"],
    key="dashboard_theme",
    help="Switch the complete dashboard between light and dark themes."
)

THEMES = {
    "Light": {
        "page_bg": "linear-gradient(135deg, #f7f9ff 0%, #eef4ff 52%, #f8f5ff 100%)",
        "surface": "#ffffff",
        "surface_alt": "#f1f5ff",
        "text": "#172554",
        "heading": "#1e3a8a",
        "muted": "#64748b",
        "border": "#dbeafe",
        "shadow": "rgba(30, 64, 175, 0.08)",
        "sidebar_bg": "linear-gradient(180deg, #eef4ff 0%, #e8edff 55%, #f3eefe 100%)",
        "sidebar_text": "#1e293b",
        "sidebar_muted": "#475569",
        "input_bg": "#ffffff",
        "plot_bg": "#ffffff",
        "plot_text": "#172554",
    },
    "Dark": {
        "page_bg": "radial-gradient(circle at top left, #172554 0%, #0b1220 48%, #111827 100%)",
        "surface": "#172033",
        "surface_alt": "#202c43",
        "text": "#e5edff",
        "heading": "#bfdbfe",
        "muted": "#a7b4ca",
        "border": "#334155",
        "shadow": "rgba(0, 0, 0, 0.24)",
        "sidebar_bg": "linear-gradient(180deg, #080f1d 0%, #111c35 55%, #241b43 100%)",
        "sidebar_text": "#f8fafc",
        "sidebar_muted": "#cbd5e1",
        "input_bg": "#111a2b",
        "plot_bg": "#172033",
        "plot_text": "#e5edff",
    },
}
C = THEMES[theme]

css = """
<style>
:root {
    color-scheme: __COLOR_SCHEME__;
}
.stApp, [data-testid="stAppViewContainer"] {
    background: __PAGE_BG__ !important;
    color: __TEXT__ !important;
}
.block-container {padding-top: 1.8rem; padding-bottom: 2.4rem; max-width: 1500px;}
h1, h2, h3, h4, h5, h6 {
    color: __HEADING__ !important;
    font-weight: 800 !important;
    letter-spacing: -0.35px;
}
p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] {
    color: __TEXT__;
}
[data-testid="stCaptionContainer"] {color: __MUTED__ !important;}
hr {border-color: __BORDER__ !important; opacity: .8;}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, __SURFACE__, __SURFACE_ALT__);
    border: 1px solid __BORDER__;
    padding: 17px 19px;
    border-radius: 18px;
    box-shadow: 0 7px 22px __SHADOW__;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {color: __MUTED__ !important;}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: __HEADING__ !important; font-weight: 800;
}
div[data-testid="stFileUploader"] {
    background: __SURFACE__;
    border: 1.5px dashed #8b5cf6;
    border-radius: 18px;
    padding: 14px;
}
[data-testid="stFileUploader"] section {
    background: __SURFACE_ALT__;
    border-radius: 12px;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea,
[data-baseweb="select"] > div {
    background: __INPUT_BG__ !important;
    color: __TEXT__ !important;
    border-color: __BORDER__ !important;
}
.stProgress > div > div > div > div {background: linear-gradient(90deg, #2563eb, #8b5cf6);}
.stButton > button, div.stDownloadButton > button {
    border: 0; border-radius: 12px; font-weight: 750;
    background: linear-gradient(100deg, #2563eb, #7c3aed);
    color: #ffffff !important;
    padding: .55rem 1rem;
    box-shadow: 0 5px 14px rgba(79, 70, 229, .20);
    transition: transform .18s ease, box-shadow .18s ease;
}
.stButton > button:hover, div.stDownloadButton > button:hover {
    color: #ffffff !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(79, 70, 229, .28);
}
section[data-testid="stSidebar"] {
    background: __SIDEBAR_BG__ !important;
    border-right: 1px solid __BORDER__;
}
section[data-testid="stSidebar"] > div {background: transparent;}
section[data-testid="stSidebar"] * {color: __SIDEBAR_TEXT__;}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: __SIDEBAR_MUTED__ !important;
}
section[data-testid="stSidebar"] hr {border-color: __BORDER__ !important;}
.sidebar-brand {
    background: linear-gradient(135deg, rgba(37,99,235,.88), rgba(124,58,237,.88));
    border: 1px solid rgba(255,255,255,.24);
    border-radius: 18px;
    padding: 16px; margin: 8px 0 18px 0;
    box-shadow: 0 8px 22px rgba(79,70,229,.18);
}
.sidebar-brand h2 {color: #fff !important; font-size: 20px; margin: 0 0 5px 0;}
.sidebar-brand p {color: #eaf0ff !important; font-size: 12px; margin: 0;}
.sidebar-chip {
    display: inline-block; background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.22); border-radius: 999px;
    padding: 5px 10px; margin: 5px 4px 2px 0; font-size: 11px;
    color: #fff !important;
}
[data-testid="stExpander"] {
    background: __SURFACE__;
    border: 1px solid __BORDER__;
    border-radius: 14px;
}
[data-testid="stAlert"] {border-radius: 14px;}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid __BORDER__; border-radius: 12px; overflow: hidden;
}
footer {visibility: hidden;}
</style>
"""
replacements = {
    "__COLOR_SCHEME__": "light" if theme == "Light" else "dark",
    "__PAGE_BG__": C["page_bg"], "__TEXT__": C["text"],
    "__HEADING__": C["heading"], "__MUTED__": C["muted"],
    "__SURFACE__": C["surface"], "__SURFACE_ALT__": C["surface_alt"],
    "__BORDER__": C["border"], "__SHADOW__": C["shadow"],
    "__SIDEBAR_BG__": C["sidebar_bg"], "__SIDEBAR_TEXT__": C["sidebar_text"],
    "__SIDEBAR_MUTED__": C["sidebar_muted"], "__INPUT_BG__": C["input_bg"],
}
for token, value in replacements.items():
    css = css.replace(token, value)

st.markdown(css, unsafe_allow_html=True)


st.title("🔍 AI-Based Steel Surface Defect Detection")
st.markdown(
    """
    Detect steel surface defects using **MobileNetV2 deep learning**,
    visualize important image regions with **Grad-CAM**, and retrieve
    inspection guidance using a **TF-IDF-based RAG module**.
    """
)

st.divider()


# --------------------------------------------------
# 2. DEFECT KNOWLEDGE BASE
# --------------------------------------------------

knowledge = {
    "crazing": {
        "description": "Fine cracks or network-like lines on the steel surface.",
        "causes": "Thermal stress, excessive rolling stress, or surface fatigue.",
        "inspection": "Inspect crack patterns, density, length, and distribution.",
        "action": "Review rolling conditions and evaluate crack depth before acceptance."
    },
    "inclusion": {
        "description": "Foreign particles or non-metallic inclusions embedded in steel.",
        "causes": "Impurities, slag entrapment, or contamination during steelmaking.",
        "inspection": "Check inclusion size, distribution, and proximity to critical areas.",
        "action": "Review steelmaking cleanliness and apply the applicable quality standard."
    },
    "patches": {
        "description": "Irregular patch-like areas with a different surface appearance.",
        "causes": "Uneven surface processing, oxidation, or inconsistent rolling conditions.",
        "inspection": "Compare patch size, shape, contrast, and affected area.",
        "action": "Review surface preparation and processing consistency."
    },
    "pitted_surface": {
        "description": "Small cavities or pit-like depressions on the steel surface.",
        "causes": "Corrosion, surface contamination, or processing-related damage.",
        "inspection": "Examine pit density, depth, and distribution.",
        "action": "Assess pit severity and check the relevant surface-quality requirements."
    },
    "rolled-in_scale": {
        "description": "Oxide scale pressed into the steel surface during rolling.",
        "causes": "Scale formation and inadequate scale removal before rolling.",
        "inspection": "Look for irregular embedded scale and surface discontinuities.",
        "action": "Review descaling effectiveness and rolling process parameters."
    },
    "scratches": {
        "description": "Linear marks or grooves on the steel surface.",
        "causes": "Mechanical contact, handling, or damaged processing equipment.",
        "inspection": "Check scratch length, direction, depth, and concentration.",
        "action": "Inspect handling equipment and evaluate whether the marks exceed quality limits."
    }
}


# --------------------------------------------------
# 3. LOAD TRAINED MODEL AND CLASS LABELS
# --------------------------------------------------

@st.cache_resource
def load_model_and_labels():
    model_path = "defect_model.keras"
    labels_path = "class_names.json"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Place it in the same folder as app.py."
        )

    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Class labels file not found: {labels_path}."
        )

    # Custom mapping for models saved with a Lambda
    # layer wrapping MobileNetV2 preprocess_input.
    preprocess_fn = (
        tf.keras.applications.mobilenet_v2.preprocess_input
    )

    try:
        loaded_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "preprocess_input": preprocess_fn,
                "mobilenet_v2.preprocess_input": preprocess_fn
            },
            compile=False,
            safe_mode=False
        )
    except TypeError:
        # Compatibility fallback for older TensorFlow/Keras
        loaded_model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "preprocess_input": preprocess_fn,
                "mobilenet_v2.preprocess_input": preprocess_fn
            },
            compile=False
        )

    with open(labels_path, "r", encoding="utf-8") as file:
        labels = json.load(file)

    if isinstance(labels, dict):
        # Supports either {"0": "crazing", ...}
        # or {"crazing": 0, ...} label mappings.
        try:
            ordered_labels = [
                labels[str(i)] for i in range(len(labels))
            ]
            labels = ordered_labels
        except (KeyError, TypeError):
            labels = sorted(labels, key=labels.get)

    labels = [str(label) for label in labels]

    if len(labels) != int(loaded_model.output_shape[-1]):
        raise ValueError(
            "The number of class labels does not match "
            "the model's output classes. Check class_names.json."
        )

    return loaded_model, labels


# --------------------------------------------------
# 4. TF-IDF RETRIEVAL MODULE
# --------------------------------------------------

@st.cache_resource
def build_retrieval_index():
    class_names = list(knowledge.keys())

    documents = [
        (
            f"{name} "
            f"{knowledge[name]['description']} "
            f"{knowledge[name]['causes']} "
            f"{knowledge[name]['inspection']} "
            f"{knowledge[name]['action']}"
        )
        for name in class_names
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    knowledge_vectors = vectorizer.fit_transform(documents)

    return class_names, vectorizer, knowledge_vectors


def retrieve_guidance(query, top_k=2):
    class_names, vectorizer, knowledge_vectors = (
        build_retrieval_index()
    )

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        knowledge_vectors
    )[0]

    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in ranked_indices:
        results.append({
            "defect": class_names[index],
            "similarity": float(similarities[index]),
            "details": knowledge[class_names[index]]
        })

    return results


# --------------------------------------------------
# 5. GRAD-CAM VISUALIZATION
# --------------------------------------------------

def make_gradcam(image_array, model, predicted_index):
    """
    Attempts Grad-CAM on a nested MobileNetV2 model.
    Returns None if the saved architecture does not expose
    a compatible intermediate feature layer.
    """

    try:
        base_model = next(
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.Model)
            and "mobilenetv2" in layer.name.lower()
        )

        conv_layer = base_model.get_layer("out_relu")

        grad_model = tf.keras.Model(
            inputs=base_model.inputs,
            outputs=[
                conv_layer.output,
                base_model.output
            ]
        )

        # Find the tensor entering the nested base model.
        base_input_tensor = base_model.input

        input_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=base_input_tensor
        )

        processed_input = input_model(
            image_array,
            training=False
        )

        with tf.GradientTape() as tape:
            conv_outputs, base_outputs = grad_model(
                processed_input,
                training=False
            )

            x = base_outputs

            # Apply subsequent top-level layers after MobileNetV2.
            base_position = model.layers.index(base_model)

            for layer in model.layers[base_position + 1:]:
                x = layer(x, training=False)

            score = x[:, predicted_index]

        gradients = tape.gradient(score, conv_outputs)

        if gradients is None:
            return None

        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(
            conv_outputs * pooled_gradients,
            axis=-1
        )

        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.reduce_max(heatmap)

        heatmap = heatmap / (max_value + tf.keras.backend.epsilon())

        return heatmap.numpy()

    except Exception:
        # The classifier remains usable even if Grad-CAM
        # is incompatible with the saved model architecture.
        return None


# --------------------------------------------------
# 6. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h2>🔍 SteelVision AI</h2>
            <p>Intelligent surface inspection dashboard</p>
            <span class="sidebar-chip">DEEP LEARNING</span>
            <span class="sidebar-chip">COMPUTER VISION</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ⚙️ Project Overview")
    st.markdown("**Model**  \nMobileNetV2")
    st.markdown("**Dataset**  \nNEU Surface Defect Database")
    st.markdown("**Defect Classes**  \n6 categories")
    st.markdown("**Explainability**  \nGrad-CAM (architecture dependent)")
    st.markdown("**Guidance Retrieval**  \nTF-IDF + Cosine Similarity")

    st.divider()
    st.markdown("### 🧩 Supported Defects")
    st.markdown(
        "- Crazing\n"
        "- Inclusion\n"
        "- Patches\n"
        "- Pitted Surface\n"
        "- Rolled-in Scale\n"
        "- Scratches"
    )

    st.divider()
    st.markdown("### 📌 Usage Note")
    st.caption(
        "Academic demonstration only. Predictions should be reviewed by "
        "a qualified inspector. Production use requires independent validation."
    )
    st.markdown(
        '<div class="sidebar-chip">Python</div>'
        '<div class="sidebar-chip">TensorFlow</div>'
        '<div class="sidebar-chip">Streamlit</div>',
        unsafe_allow_html=True
    )


# 8. LOAD MODEL
# --------------------------------------------------

try:
    model, class_names = load_model_and_labels()
except Exception as error:
    st.error("Unable to load the trained model.")
    st.code(str(error))
    st.info(
        "Check that defect_model.keras and class_names.json "
        "are in the same folder as app.py."
    )
    st.stop()


# --------------------------------------------------
# 8. IMAGE UPLOAD
# --------------------------------------------------

st.subheader("📤 Upload Steel Surface Image")

uploaded_file = st.file_uploader(
    "Choose a steel surface image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(
                image,
                caption="Uploaded Steel Surface Image",
                use_container_width=True
            )

        # Keep preprocessing consistent with the saved model.
        resized = image.resize((224, 224))
        image_array = np.asarray(
            resized,
            dtype=np.float32
        )
        batch = np.expand_dims(image_array, axis=0)

        # --------------------------------------------------
        # 9. PREDICTION
        # --------------------------------------------------

        with st.spinner("Analyzing image..."):
            predictions = model.predict(
                batch,
                verbose=0
            )[0]

        if len(predictions) != len(class_names):
            st.error(
                "Model output size and class label count do not match."
            )
            st.stop()

        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[predicted_index])

        with col2:
            st.subheader("🧠 Prediction Result")

            st.metric(
                "Predicted Defect",
                predicted_class.replace("_", " ").title()
            )

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.progress(
                min(max(confidence, 0.0), 1.0)
            )

            if confidence < 0.60:
                st.warning(
                    "Confidence is relatively low. "
                    "Review the image and prediction carefully."
                )
            else:
                st.success("Prediction generated successfully.")

        # --------------------------------------------------
        # 10. CLASS PROBABILITIES
        # --------------------------------------------------

        st.divider()
        st.subheader("📊 Class-wise Prediction Scores")

        probability_data = {
            name.replace("_", " ").title(): float(score) * 100
            for name, score in zip(class_names, predictions)
        }

        st.bar_chart(
            probability_data,
            height=350
        )
        st.caption(
            "Scores are displayed as percentages. They represent model outputs, "
            "not guaranteed defect probabilities."
        )

        # --------------------------------------------------
        # 11. GRAD-CAM
        # --------------------------------------------------

        st.divider()
        st.subheader("🔥 Grad-CAM Explainability")

        heatmap = make_gradcam(
            batch,
            model,
            predicted_index
        )

        if heatmap is not None:
            heatmap_resized = Image.fromarray(
                np.uint8(heatmap * 255)
            ).resize((224, 224))

            heatmap_array = (
                np.asarray(heatmap_resized) / 255.0
            )

            original_array = np.asarray(
                resized,
                dtype=np.uint8
            )

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            fig.patch.set_facecolor(C["plot_bg"])
            for axis in axes:
                axis.set_facecolor(C["plot_bg"])

            axes[0].imshow(original_array)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(original_array)
            axes[1].imshow(
                heatmap_array,
                cmap="jet",
                alpha=0.45
            )
            axes[1].set_title("Grad-CAM Heatmap")
            axes[1].axis("off")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Grad-CAM highlights image regions that influence "
                "the model's prediction; it is not a defect boundary."
            )
        else:
            st.warning(
                "Grad-CAM is unavailable for this saved model "
                "architecture. The classification result is still available."
            )

        # --------------------------------------------------
        # 12. DEFECT DETAILS
        # --------------------------------------------------

        st.divider()
        st.subheader("📝 Defect Analysis")

        if predicted_class in knowledge:
            details = knowledge[predicted_class]

            st.markdown(
                f"**Description:** {details['description']}"
            )
            st.markdown(
                f"**Possible Causes:** {details['causes']}"
            )
            st.markdown(
                f"**Inspection Guidance:** {details['inspection']}"
            )
            st.markdown(
                f"**Suggested Action:** {details['action']}"
            )
        else:
            st.warning(
                "No knowledge-base entry was found for this label."
            )

        # --------------------------------------------------
        # 13. RAG-STYLE RETRIEVAL
        # --------------------------------------------------

        st.divider()
        st.subheader("📚 Similar Defect Guidance (TF-IDF Retrieval)")

        query = (
            f"{predicted_class} "
            f"{knowledge.get(predicted_class, {}).get('description', '')} "
            f"{knowledge.get(predicted_class, {}).get('inspection', '')}"
        )

        retrieval_results = retrieve_guidance(
            query,
            top_k=2
        )

        for rank, result in enumerate(retrieval_results, start=1):
            with st.expander(
                f"{rank}. {result['defect'].replace('_', ' ').title()} "
                f"— Similarity: {result['similarity']:.3f}",
                expanded=(rank == 1)
            ):
                st.write(
                    f"**Description:** "
                    f"{result['details']['description']}"
                )
                st.write(
                    f"**Possible Causes:** "
                    f"{result['details']['causes']}"
                )
                st.write(
                    f"**Inspection:** "
                    f"{result['details']['inspection']}"
                )
                st.write(
                    f"**Suggested Action:** "
                    f"{result['details']['action']}"
                )

        st.caption(
            "The retrieval module ranks predefined defect descriptions "
            "using TF-IDF and cosine similarity. It does not generate "
            "new answers using a generative language model."
        )

        # --------------------------------------------------
        # 14. DOWNLOAD PREDICTION REPORT
        # --------------------------------------------------

        report = (
            "AI-Based Steel Surface Defect Detection Report\n"
            "-----------------------------------------------\n"
            f"Predicted defect: {predicted_class}\n"
            f"Model confidence: {confidence * 100:.2f}%\n\n"
            f"Description: "
            f"{knowledge.get(predicted_class, {}).get('description', 'N/A')}\n"
            f"Possible causes: "
            f"{knowledge.get(predicted_class, {}).get('causes', 'N/A')}\n"
            f"Inspection guidance: "
            f"{knowledge.get(predicted_class, {}).get('inspection', 'N/A')}\n"
            f"Suggested action: "
            f"{knowledge.get(predicted_class, {}).get('action', 'N/A')}\n\n"
            "Note: This report is an academic demonstration, not a "
            "certified industrial quality decision."
        )

        st.download_button(
            label="⬇️ Download Prediction Report",
            data=report,
            file_name="defect_prediction_report.txt",
            mime="text/plain"
        )

    except Exception as error:
        st.error("An error occurred while processing the image.")
        st.code(str(error))

else:
    st.info(
        "Upload a steel surface image to start defect classification."
    )


# --------------------------------------------------
# 15. FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "AI-Based Product Defect Detection | "
    "MobileNetV2 + Grad-CAM + TF-IDF Retrieval | "
    "Academic Project"
)
