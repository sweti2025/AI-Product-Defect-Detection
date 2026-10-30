
import os
import json

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Steel Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 AI-Based Steel Surface Defect Detection")
st.markdown(
    """
    Deep learning-based steel surface defect classification
    with prediction confidence, Grad-CAM explainability,
    retrieval-based inspection guidance and batch analysis.
    """
)

st.divider()


# =========================================================
# 2. KNOWLEDGE BASE
# =========================================================

knowledge = {
    "crazing": {
        "description": "Fine cracks or network-like lines on the steel surface.",
        "causes": "Thermal stress, rolling stress or surface fatigue.",
        "inspection": "Inspect crack patterns, density, length and distribution.",
        "action": "Review processing conditions and evaluate crack severity."
    },
    "inclusion": {
        "description": "Foreign particles or non-metallic inclusions in steel.",
        "causes": "Impurities, slag entrapment or contamination during steelmaking.",
        "inspection": "Check inclusion size, distribution and affected areas.",
        "action": "Review steelmaking cleanliness and applicable quality standards."
    },
    "patches": {
        "description": "Irregular patch-like areas with a different surface appearance.",
        "causes": "Uneven processing, oxidation or inconsistent rolling conditions.",
        "inspection": "Compare patch size, shape, contrast and distribution.",
        "action": "Review surface preparation and process consistency."
    },
    "pitted_surface": {
        "description": "Small cavities or pit-like depressions on the steel surface.",
        "causes": "Corrosion, surface contamination or processing damage.",
        "inspection": "Examine pit density, depth and distribution.",
        "action": "Assess severity against the relevant surface-quality requirements."
    },
    "rolled-in_scale": {
        "description": "Oxide scale pressed into the steel surface during rolling.",
        "causes": "Scale formation and insufficient removal before rolling.",
        "inspection": "Look for irregular embedded scale and surface discontinuities.",
        "action": "Review descaling effectiveness and rolling parameters."
    },
    "scratches": {
        "description": "Linear marks or grooves on the steel surface.",
        "causes": "Mechanical contact, handling or damaged equipment.",
        "inspection": "Check scratch length, direction, depth and concentration.",
        "action": "Inspect handling equipment and assess the applicable quality limits."
    }
}


# =========================================================
# 3. MODEL LOADING
# =========================================================

@st.cache_resource
def load_model_and_labels():
    model_path = "defect_model.keras"
    labels_path = "class_names.json"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model file: {model_path}"
        )

    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Missing label file: {labels_path}"
        )

    preprocess_fn = (
        tf.keras.applications.mobilenet_v2.preprocess_input
    )

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "preprocess_input": preprocess_fn,
            "mobilenet_v2.preprocess_input": preprocess_fn
        },
        compile=False,
        safe_mode=False
    )

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    if isinstance(labels, dict):
        try:
            labels = [labels[str(i)] for i in range(len(labels))]
        except KeyError:
            labels = sorted(labels, key=labels.get)

    labels = [str(x) for x in labels]

    if len(labels) != int(model.output_shape[-1]):
        raise ValueError(
            "Class label count does not match model output size."
        )

    return model, labels


try:
    model, class_names = load_model_and_labels()
except Exception as error:
    st.error("Could not load the model.")
    st.code(str(error))
    st.info(
        "Check model file, label file and TensorFlow/Keras compatibility."
    )
    st.stop()


# =========================================================
# 4. MODEL DIAGNOSTICS
# =========================================================

def get_layer_summary(model):
    rows = []

    for i, layer in enumerate(model.layers):
        try:
            output_shape = str(layer.output.shape)
        except Exception:
            output_shape = "Unavailable"

        rows.append({
            "index": i,
            "layer_name": layer.name,
            "layer_type": type(layer).__name__,
            "output_shape": output_shape
        })

    return pd.DataFrame(rows)


with st.sidebar:
    st.header("⚙️ Project Information")
    st.write("**Architecture:** MobileNetV2")
    st.write("**Dataset:** NEU-DET")
    st.write("**Classes:** 6")
    st.write("**Explainability:** Grad-CAM attempt")
    st.write("**Retrieval:** TF-IDF + cosine similarity")

    with st.expander("Model diagnostics"):
        st.write("Input shape:", model.input_shape)
        st.write("Output shape:", model.output_shape)
        st.dataframe(
            get_layer_summary(model),
            use_container_width=True
        )

    st.divider()
    st.caption(
        "Academic prototype. Predictions require human verification."
    )


# =========================================================
# 5. TF-IDF RETRIEVAL
# =========================================================

@st.cache_resource
def build_retrieval_index():
    names = list(knowledge.keys())

    documents = [
        (
            f"{name} "
            f"{knowledge[name]['description']} "
            f"{knowledge[name]['causes']} "
            f"{knowledge[name]['inspection']} "
            f"{knowledge[name]['action']}"
        )
        for name in names
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    vectors = vectorizer.fit_transform(documents)

    return names, vectorizer, vectors


def retrieve_guidance(query, top_k=2):
    names, vectorizer, vectors = build_retrieval_index()

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, vectors)[0]

    indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "defect": names[i],
            "similarity": float(scores[i]),
            "details": knowledge[names[i]]
        }
        for i in indices
    ]


# =========================================================
# 6. IMAGE PREPROCESSING
# =========================================================

def prepare_image(image):
    image = image.convert("RGB").resize((224, 224))
    array = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)

    # The trained model includes preprocessing internally.
    # Do not apply preprocess_input again here.
    return image, batch


# =========================================================
# 7. GRAD-CAM
# =========================================================

def find_nested_backbone(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            name = layer.name.lower()

            if "mobilenet" in name:
                return layer

    return None


def make_gradcam(image_batch, model, predicted_index):
    """
    Grad-CAM for a saved model with a nested MobileNetV2
    backbone whose output is a spatial feature map.

    Returns (heatmap, error_message).
    A None heatmap means Grad-CAM could not be computed.
    """

    try:
        backbone = find_nested_backbone(model)

        if backbone is None:
            return None, (
                "A nested MobileNetV2 backbone was not found. "
                "Inspect Model diagnostics to identify the feature layer."
            )

        # Build a feature extractor from the backbone.
        # This avoids relying on a hard-coded 'out_relu' name.
        candidate_layers = [
            layer for layer in backbone.layers
            if len(getattr(layer, "output_shape", ())) == 4
        ]

        if not candidate_layers:
            return None, "No spatial feature layer was found in the backbone."

        # The last 4-D layer is typically the final convolutional feature map.
        feature_layer = candidate_layers[-1]

        feature_extractor = tf.keras.Model(
            inputs=backbone.inputs,
            outputs=feature_layer.output
        )

        # Create a model returning both feature maps and predictions.
        # This approach works only when the backbone's output is
        # connected to the original model in the expected way.
        feature_input = tf.keras.Input(
            shape=tuple(backbone.input_shape[1:])
        )

        feature_maps = feature_extractor(feature_input)

        x = feature_maps

        # Replay layers after the backbone feature output is not
        # generally possible if the model includes operations inside
        # the backbone after this feature layer. Therefore this method
        # uses the original classifier graph only when compatible.
        #
        # Prefer an intermediate-output model built from the full graph.
        try:
            full_graph = tf.keras.Model(
                inputs=model.inputs,
                outputs=[
                    feature_extractor(backbone.input),
                    model.output
                ]
            )

            with tf.GradientTape() as tape:
                feature_values, predictions = full_graph(
                    image_batch,
                    training=False
                )
                score = predictions[:, predicted_index]

            gradients = tape.gradient(score, feature_values)

        except Exception as graph_error:
            return None, (
                "Could not connect the backbone feature map to the "
                f"classifier graph: {graph_error}"
            )

        if gradients is None:
            return None, (
                "No gradients were returned for the chosen feature layer. "
                "A graph-compatible Grad-CAM model must be constructed."
            )

        weights = tf.reduce_mean(
            gradients,
            axis=(1, 2),
            keepdims=True
        )

        heatmap = tf.reduce_sum(
            weights * feature_values,
            axis=-1
        )[0]

        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (
            tf.reduce_max(heatmap) + 1e-8
        )

        return heatmap.numpy(), None

    except Exception as error:
        return None, str(error)


def show_gradcam(original_image, heatmap):
    heatmap_image = Image.fromarray(
        np.uint8(np.clip(heatmap, 0, 1) * 255)
    ).resize(original_image.size)

    heatmap_array = np.asarray(heatmap_image) / 255.0
    original_array = np.asarray(original_image)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(original_array)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(original_array)
    axes[1].imshow(
        heatmap_array,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )
    axes[1].set_title("Grad-CAM Overlay")
    axes[1].axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# 8. PREDICTION HELPER
# =========================================================

def predict_image(image):
    resized, batch = prepare_image(image)

    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))

    return {
        "image": resized,
        "batch": batch,
        "probabilities": probabilities,
        "index": index,
        "label": class_names[index],
        "confidence": float(probabilities[index])
    }


# =========================================================
# 9. SINGLE IMAGE CLASSIFICATION
# =========================================================

st.header("📤 Single Image Inspection")

uploaded_file = st.file_uploader(
    "Upload a steel surface image",
    type=["jpg", "jpeg", "png", "bmp"],
    key="single_upload"
)

if uploaded_file is not None:
    try:
        source_image = Image.open(uploaded_file).convert("RGB")
        result = predict_image(source_image)

        label = result["label"]
        confidence = result["confidence"]
        probabilities = result["probabilities"]

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                source_image,
                caption="Uploaded image",
                use_container_width=True
            )

        with col2:
            st.subheader("🧠 Prediction")

            st.metric(
                "Predicted defect",
                label.replace("_", " ").title()
            )

            st.metric(
                "Top-class confidence",
                f"{confidence * 100:.2f}%"
            )

            st.progress(
                min(max(confidence, 0.0), 1.0)
            )

            if confidence < 0.60:
                st.warning(
                    "Low confidence: manual inspection recommended. "
                    "The 60% cutoff is a provisional review rule, "
                    "not a validated reliability threshold."
                )
            else:
                st.info(
                    "Model prediction generated. Verify against "
                    "the applicable inspection procedure."
                )

        st.divider()
        st.subheader("📊 Class-wise scores")

        score_df = pd.DataFrame({
            "Defect": [
                name.replace("_", " ").title()
                for name in class_names
            ],
            "Score": probabilities
        }).set_index("Defect")

        st.bar_chart(score_df)

        st.divider()
        st.subheader("🔥 Grad-CAM Explainability")

        with st.spinner("Attempting to compute Grad-CAM..."):
            heatmap, cam_error = make_gradcam(
                result["batch"],
                model,
                result["index"]
            )

        if heatmap is not None:
            show_gradcam(result["image"], heatmap)
            st.caption(
                "Heatmap highlights regions influencing the prediction; "
                "it is not a verified defect segmentation mask."
            )
        else:
            st.warning("Grad-CAM could not be generated for this model.")
            with st.expander("Grad-CAM diagnostic details"):
                st.code(cam_error or "Unknown Grad-CAM error")

        st.divider()
        st.subheader("📝 Defect guidance")

        details = knowledge.get(label)

        if details:
            st.write("**Description:**", details["description"])
            st.write("**Possible causes:**", details["causes"])
            st.write("**Inspection:**", details["inspection"])
            st.write("**Suggested action:**", details["action"])

        st.divider()
        st.subheader("📚 Similar guidance retrieval")

        query = (
            f"{label} "
            f"{details['description'] if details else ''} "
            f"{details['inspection'] if details else ''}"
        )

        for rank, item in enumerate(
            retrieve_guidance(query, top_k=2),
            start=1
        ):
            with st.expander(
                f"{rank}. {item['defect'].replace('_', ' ').title()} "
                f"(similarity {item['similarity']:.3f})"
            ):
                st.write(item["details"]["description"])
                st.write("**Causes:**", item["details"]["causes"])
                st.write("**Inspection:**", item["details"]["inspection"])
                st.write("**Action:**", item["details"]["action"])

        report = (
            "AI Steel Surface Defect Detection Report\n"
            "=========================================\n"
            f"File: {uploaded_file.name}\n"
            f"Predicted class: {label}\n"
            f"Confidence: {confidence * 100:.2f}%\n"
            f"Description: {details['description'] if details else 'N/A'}\n"
            f"Possible causes: {details['causes'] if details else 'N/A'}\n"
            f"Inspection: {details['inspection'] if details else 'N/A'}\n"
            f"Suggested action: {details['action'] if details else 'N/A'}\n\n"
            "This report is an academic model output, not a certified "
            "industrial quality decision."
        )

        st.download_button(
            "⬇️ Download inspection report",
            data=report,
            file_name="defect_inspection_report.txt",
            mime="text/plain"
        )

    except Exception as error:
        st.error("Image processing failed.")
        st.code(str(error))


# =========================================================
# 10. BATCH PREDICTION
# =========================================================

st.divider()
st.header("📦 Batch Defect Inspection")

batch_files = st.file_uploader(
    "Upload multiple steel images",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
    key="batch_upload"
)

if batch_files:
    batch_results = []

    for file in batch_files:
        try:
            image = Image.open(file).convert("RGB")
            result = predict_image(image)

            batch_results.append({
                "filename": file.name,
                "predicted_defect": result["label"],
                "confidence_percent": round(
                    result["confidence"] * 100, 2
                ),
                "review_status": (
                    "Needs manual review"
                    if result["confidence"] < 0.60
                    else "Verify model prediction"
                )
            })

        except Exception as error:
            batch_results.append({
                "filename": file.name,
                "predicted_defect": "Processing error",
                "confidence_percent": None,
                "review_status": str(error)
            })

    batch_df = pd.DataFrame(batch_results)

    st.dataframe(
        batch_df,
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download batch CSV",
        data=batch_df.to_csv(index=False).encode("utf-8"),
        file_name="steel_defect_batch_report.csv",
        mime="text/csv"
    )


# =========================================================
# 11. FOOTER
# =========================================================

st.divider()
st.caption(
    "AI-Based Steel Surface Defect Detection | "
    "MobileNetV2 | Streamlit | TF-IDF Retrieval | "
    "Academic prototype"
)
