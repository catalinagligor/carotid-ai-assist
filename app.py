import os
import io
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Carotid Ultrasound Risk Assistant",
    page_icon="🩺",
    layout="wide"
)

MODEL_PATH = "models/attention_unet_final.pth"
IMAGE_SIZE = 256

# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.metric-card {
    padding: 1rem;
    border-radius: 0.9rem;
    background: #f5f7fb;
    border: 1px solid #e4e8f0;
}
.small-note {
    font-size: 0.92rem;
    color: #5c667a;
}
.legend-box {
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 3px;
    margin-right: 8px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL
# =========================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = 1
        for f in features:
            self.encoders.append(DoubleConv(in_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.attentions.append(AttentionGate(F_g=f, F_l=f, F_int=f // 2))
            self.decoders.append(DoubleConv(f * 2, f))

        self.output = nn.Conv2d(features[0], 1, 1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for up, att, dec, skip in zip(self.upconvs, self.attentions, self.decoders, skips):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            skip = att(g=x, x=skip)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return torch.sigmoid(self.output(x))


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNet().to(device)

    model_loaded = False
    model_message = ""

    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        model_loaded = True
        model_message = f"Model încărcat din: {MODEL_PATH}"
    else:
        model_message = (
            f"Fișierul modelului '{MODEL_PATH}' nu a fost găsit. "
            "Aplicația rulează interfața completă, dar pentru predicții reale trebuie adăugat modelul antrenat."
        )

    model.eval()
    return model, device, model_loaded, model_message

model, device, model_loaded, model_message = load_model()

# =========================================================
# HELPERS
# =========================================================
def normalize_to_unit(img_array):
    img_array = img_array.astype(np.float32)
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
    return np.clip(img_array, 0.0, 1.0)


def preprocess_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert("L")
    original = np.array(img)
    original = normalize_to_unit(original)

    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    resized = np.array(img_resized)
    resized = normalize_to_unit(resized)

    tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return original, resized, tensor


def predict_mask(input_tensor, threshold=0.5):
    with torch.no_grad():
        pred_prob = model(input_tensor).cpu().squeeze().numpy()
    pred_mask = (pred_prob > threshold).astype(np.uint8)
    return pred_prob, pred_mask


def colorize_mask(mask, color="green"):
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)

    if color == "green":
        rgb[mask == 1] = [0.0, 1.0, 0.0]
    elif color == "red":
        rgb[mask == 1] = [1.0, 0.0, 0.0]
    elif color == "yellow":
        rgb[mask == 1] = [1.0, 1.0, 0.0]

    return rgb


def make_overlay(gray_img, mask, color="red", alpha=0.55):
    gray_rgb = np.stack([gray_img, gray_img, gray_img], axis=-1)
    color_layer = np.zeros_like(gray_rgb)

    if color == "red":
        color_layer[mask == 1] = [1.0, 0.0, 0.0]
    elif color == "green":
        color_layer[mask == 1] = [0.0, 1.0, 0.0]
    elif color == "yellow":
        color_layer[mask == 1] = [1.0, 1.0, 0.0]

    overlay = gray_rgb.copy()
    overlay[mask == 1] = (1 - alpha) * gray_rgb[mask == 1] + alpha * color_layer[mask == 1]
    return np.clip(overlay, 0.0, 1.0)


def resize_mask_to_original(mask, original_shape):
    pil_mask = Image.fromarray((mask * 255).astype(np.uint8))
    resized_mask = pil_mask.resize((original_shape[1], original_shape[0]))
    resized_mask = (np.array(resized_mask) > 127).astype(np.uint8)
    return resized_mask


def estimate_clinical_risk(age, smoker, hypertension, cholesterol, diabetes, bmi, systolic_bp):
    score = 0

    if age >= 65:
        score += 3
    elif age >= 55:
        score += 2
    elif age >= 45:
        score += 1

    if smoker == "Da":
        score += 2
    if hypertension == "Da":
        score += 2
    if cholesterol == "Da":
        score += 2
    if diabetes == "Da":
        score += 2
    if bmi >= 30:
        score += 1
    if systolic_bp >= 140:
        score += 1

    if score <= 2:
        label = "Scăzut"
        color = "🟢"
    elif score <= 5:
        label = "Moderat"
        color = "🟠"
    else:
        label = "Crescut"
        color = "🔴"

    return score, label, color


def estimate_image_flags(mask, prob_map):
    area_ratio = float(mask.mean()) * 100.0
    active_pixels = prob_map[pred_mask == 1]
    confidence = float(active_pixels.mean() * 100.0) if active_pixels.size > 0 else 0.0

    flags = []
    if area_ratio < 1:
        flags.append("arie segmentată foarte mică")
    elif area_ratio > 35:
        flags.append("arie segmentată extinsă")

    if confidence < 20:
        flags.append("încredere globală redusă a modelului")
    elif confidence > 70:
        flags.append("semnal segmentat bine definit")

    if len(flags) == 0:
        flags.append("fără indicatori imagistici simpli ieșiți din intervalele euristice")

    return area_ratio, confidence, flags


def image_download_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# =========================================================
# HEADER
# =========================================================
st.title("🩺 Carotid Ultrasound Risk Assistant")
st.write(
    "Aplicație Streamlit pentru analiză asistată a imaginilor ecografice carotidiene, "
    "segmentare Attention U-Net și integrarea unor factori clinici simpli pentru estimare orientativă de risc."
)

if model_loaded:
    st.success(model_message)
else:
    st.warning(model_message)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Date pacient")

    age = st.number_input("Vârstă", min_value=18, max_value=100, value=54)
    sex = st.selectbox("Sex", ["Feminin", "Masculin"])
    smoker = st.selectbox("Fumător", ["Nu", "Da"])
    hypertension = st.selectbox("Hipertensiune", ["Nu", "Da"])
    cholesterol = st.selectbox("Colesterol crescut", ["Nu", "Da"])
    diabetes = st.selectbox("Diabet", ["Nu", "Da"])
    bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
    systolic_bp = st.number_input("TA sistolică", min_value=80, max_value=250, value=125)

    st.divider()
    st.header("Setări analiză")
    threshold = st.slider("Prag segmentare", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    alpha = st.slider("Transparență overlay", min_value=0.10, max_value=0.90, value=0.55, step=0.05)

    st.divider()
    st.markdown("**Legendă culori**")
    st.markdown(
        '<span class="legend-box" style="background:#ff0000;"></span>Predicție model',
        unsafe_allow_html=True
    )
    st.markdown(
        '<span class="legend-box" style="background:#00aa00;"></span>Mască de referință (opțional, dacă există)',
        unsafe_allow_html=True
    )

# =========================================================
# INPUTS
# =========================================================
tab1, tab2 = st.tabs(["Analiză imagine", "Rezumat clinic"])

with tab1:
    st.subheader("Încărcare imagine")
    uploaded_image = st.file_uploader(
        "Încarcă o imagine ecografică carotidiană",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )

    uploaded_mask = st.file_uploader(
        "Încarcă opțional o mască reală pentru comparație",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )

    if uploaded_image is not None:
        original_img, resized_img, input_tensor = preprocess_uploaded_image(uploaded_image)

        if model_loaded:
            pred_prob, pred_mask = predict_mask(input_tensor, threshold=threshold)
        else:
            pred_prob = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
            pred_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        pred_overlay = make_overlay(resized_img, pred_mask, color="red", alpha=alpha)

        true_mask_resized = None
        true_overlay = None
        comparison_overlay = None

        if uploaded_mask is not None:
            true_mask_img = Image.open(uploaded_mask).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
            true_mask_resized = (np.array(true_mask_img) > 127).astype(np.uint8)
            true_overlay = make_overlay(resized_img, true_mask_resized, color="green", alpha=alpha)

            comparison_overlay = np.stack([resized_img, resized_img, resized_img], axis=-1)
            comparison_overlay[(true_mask_resized == 1) & (pred_mask == 0)] = [0.0, 1.0, 0.0]
            comparison_overlay[(true_mask_resized == 0) & (pred_mask == 1)] = [1.0, 0.0, 0.0]
            comparison_overlay[(true_mask_resized == 1) & (pred_mask == 1)] = [1.0, 1.0, 0.0]

        area_ratio, confidence, flags = estimate_image_flags(pred_mask, pred_prob)

        st.subheader("Rezultate vizuale")

        if uploaded_mask is not None:
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown("**Imagine originală**")
                st.image(resized_img, clamp=True, use_container_width=True)

            with c2:
                st.markdown("**Mască reală**")
                st.image(colorize_mask(true_mask_resized, color="green"), clamp=True, use_container_width=True)

            with c3:
                st.markdown("**Predicție model**")
                st.image(colorize_mask(pred_mask, color="red"), clamp=True, use_container_width=True)

            with c4:
                st.markdown("**Suprapunere comparație**")
                st.image(comparison_overlay, clamp=True, use_container_width=True)

            st.caption("Verde = mască reală, roșu = predicție, galben = suprapunere corectă.")
        else:
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Imagine originală**")
                st.image(resized_img, clamp=True, use_container_width=True)

            with c2:
                st.markdown("**Hartă probabilitate**")
                fig_prob, ax_prob = plt.subplots(figsize=(4, 4))
                ax_prob.imshow(pred_prob, cmap="magma")
                ax_prob.axis("off")
                st.pyplot(fig_prob)

            with c3:
                st.markdown("**Overlay predicție**")
                st.image(pred_overlay, clamp=True, use_container_width=True)

            st.caption("Roșu = zona segmentată de model.")

        st.subheader("Indicatori imagistici rapizi")
        m1, m2, m3 = st.columns(3)
        m1.metric("Arie segmentată", f"{area_ratio:.2f}%")
        m2.metric("Încredere medie model", f"{confidence:.2f}%")
        m3.metric("Prag utilizat", f"{threshold:.2f}")

        st.markdown("**Observații euristice:**")
        for flag in flags:
            st.write(f"- {flag}")

        summary_text = (
            f"Analiza imaginii indică o arie segmentată de {area_ratio:.2f}% "
            f"și o încredere medie a modelului de {confidence:.2f}%. "
            f"Aceste rezultate trebuie corelate cu contextul clinic și cu interpretarea specialistului."
        )
        st.text_area("Interpretare asistată", value=summary_text, height=120)

        export_fig, export_ax = plt.subplots(figsize=(5, 5))
        export_ax.imshow(pred_overlay)
        export_ax.axis("off")
        export_ax.set_title("Predicție carotidă - overlay")
        overlay_png = image_download_bytes(export_fig)

        st.download_button(
            "Descarcă overlay PNG",
            data=overlay_png,
            file_name="carotid_overlay_prediction.png",
            mime="image/png"
        )

    else:
        st.info("Încarcă o imagine pentru a activa analiza.")

with tab2:
    st.subheader("Rezumat clinic orientativ")

    risk_score, risk_label, risk_color = estimate_clinical_risk(
        age=age,
        smoker=smoker,
        hypertension=hypertension,
        cholesterol=cholesterol,
        diabetes=diabetes,
        bmi=bmi,
        systolic_bp=systolic_bp
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Scor clinic simplificat", risk_score)
    c2.metric("Categorie risc", f"{risk_color} {risk_label}")
    c3.metric("Sex", sex)

    st.markdown("**Factori introduși:**")
    st.write(
        f"- Vârstă: {age} ani\n"
        f"- Fumător: {smoker}\n"
        f"- Hipertensiune: {hypertension}\n"
        f"- Colesterol crescut: {cholesterol}\n"
        f"- Diabet: {diabetes}\n"
        f"- BMI: {bmi:.1f}\n"
        f"- TA sistolică: {systolic_bp}"
    )


    

