import re
import time

import pandas as pd
import pdfplumber
import plotly.graph_objects as go
import streamlit as st
from transformers import pipeline

# ============================================================
# 1. KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="MALAQBI - SDGs Center Unsulbar",
    page_icon="🌊",
    layout="wide",
)

# ============================================================
# 2. DATA 17 SDGs (label bilingual + warna resmi PBB)
#    Label ini juga dipakai sebagai "kalimat kandidat" oleh model zero-shot,
#    jadi jangan diubah tanpa alasan karena akan memengaruhi hasil.
# ============================================================
SDG_DATA = [
    (1, "SDG 1: No Poverty (Tanpa Kemiskinan)", "#E5243B"),
    (2, "SDG 2: Zero Hunger (Ketahanan Pangan)", "#DDA63A"),
    (3, "SDG 3: Good Health (Kesehatan & Kesejahteraan)", "#4C9F38"),
    (4, "SDG 4: Quality Education (Pendidikan)", "#C5192D"),
    (5, "SDG 5: Gender Equality (Kesetaraan Gender)", "#FF3A21"),
    (6, "SDG 6: Clean Water (Air Bersih & Sanitasi)", "#26BDE2"),
    (7, "SDG 7: Clean Energy (Energi Terbarukan)", "#FCC30B"),
    (8, "SDG 8: Economic Growth (Pekerjaan & Ekonomi)", "#A21942"),
    (9, "SDG 9: Industry & Innovation (Inovasi & Infrastruktur)", "#FD6925"),
    (10, "SDG 10: Reduced Inequalities (Mengurangi Ketimpangan)", "#DD1367"),
    (11, "SDG 11: Sustainable Cities (Kota & Pemukiman Berkelanjutan)", "#FD9D24"),
    (12, "SDG 12: Responsible Consumption (Konsumsi & Produksi)", "#BF8B2E"),
    (13, "SDG 13: Climate Action (Aksi Iklim)", "#3F7E44"),
    (14, "SDG 14: Life Below Water (Ekosistem Laut & Perikanan)", "#0A97D9"),
    (15, "SDG 15: Life on Land (Ekosistem Darat & Kehutanan)", "#56C02B"),
    (16, "SDG 16: Peace & Justice (Keadilan & Kelembagaan)", "#00689D"),
    (17, "SDG 17: Partnerships (Kemitraan Global)", "#19486A"),
]
SDG_LABELS = [label for _, label, _ in SDG_DATA]
LABEL_KE_NOMOR = {label: no for no, label, _ in SDG_DATA}
NOMOR_KE_LABEL = {no: label for no, label, _ in SDG_DATA}
NOMOR_KE_WARNA = {no: warna for no, _, warna in SDG_DATA}

# ============================================================
# 3. DAFTAR MODEL AI (Hugging Face Hub)
# ============================================================
MODEL_REGISTRY = {
    "🧠 mDeBERTa-v3 (Zero-Shot, Multibahasa)": {
        "path": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "task": "zero-shot-classification",
        "jenis": "Zero-Shot Classification berbasis Natural Language Inference (NLI)",
        "bahasa": "Multibahasa, termasuk Bahasa Indonesia",
        "cakupan": list(range(1, 18)),
        "catatan": (
            "Setiap SDG dinilai secara terpisah (multi-label), sehingga beberapa SDG "
            "dapat sama-sama memperoleh skor tinggi."
        ),
    },
    "📊 sdgBERT (Fine-Tuned, OSDG-CD)": {
        "path": "sadickam/sdg-classification-bert",
        "task": "text-classification",
        "jenis": "Fine-Tuned BERT, dilatih pada OSDG Community Dataset (sadickam)",
        "bahasa": "Bahasa Inggris",
        "cakupan": list(range(1, 17)),
        "catatan": (
            "Hanya mencakup SDG 1–16; SDG 17 tidak dikenali model ini. "
            "Hasil paling andal untuk abstrak berbahasa Inggris."
        ),
    },
    "🔬 OSDG SDG Classifier (Fine-Tuned)": {
        "path": "jonas/sdg_classifier_osdg",
        "task": "text-classification",
        "jenis": "Fine-Tuned BERT, dilatih pada data komunitas OSDG (jonas)",
        "bahasa": "Bahasa Inggris",
        "cakupan": list(range(1, 16)),
        "catatan": (
            "Hanya mencakup SDG 1–15; SDG 16 dan 17 tidak dikenali model ini. "
            "Hasil paling andal untuk abstrak berbahasa Inggris."
        ),
    },
}

BATAS_KATA = 300  # perkiraan kasar batas baca model (512 token)

# Pengaturan "lebar penuh" yang cocok untuk Streamlit versi lama maupun baru
_VERSI_ST = tuple(int(x) for x in st.__version__.split(".")[:2])
LEBAR_PENUH = {"width": "stretch"} if _VERSI_ST >= (1, 50) else {"use_container_width": True}

# ============================================================
# 4. GAYA TAMPILAN (CSS)
# ============================================================
st.markdown(
    """
    <style>
        .main-header {
            background: linear-gradient(135deg, #003366 0%, #0056b3 100%);
            padding: 1.6rem 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
        .main-header h1 {
            color: white; text-align: center; margin: 0; font-size: 2.2rem;
        }
        .main-header p {
            color: #d1ecf1; text-align: center; margin: 0.4rem 0 0 0; font-size: 1rem;
        }
        .sdg-row {
            margin-bottom: 12px; padding: 10px 12px; border-radius: 6px;
            background: rgba(128, 128, 128, 0.08);
        }
        .sdg-row-head {
            display: flex; justify-content: space-between; gap: 8px;
            font-weight: 600; margin-bottom: 6px;
        }
        .sdg-track {
            background: rgba(128, 128, 128, 0.2); border-radius: 4px; height: 10px; width: 100%;
        }
        .footer {
            text-align: center; opacity: 0.75; padding: 1.5rem 0 0.5rem 0; font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. FUNGSI-FUNGSI
# ============================================================
@st.cache_resource(show_spinner=False, max_entries=1)
def load_classifier(model_path, model_task):
    """Memuat model AI. max_entries=1: hanya satu model disimpan di memori
    agar aplikasi tidak kehabisan RAM di server gratis."""
    return pipeline(task=model_task, model=model_path, device=-1)


def nomor_sdg_dari_label(label):
    """Menerjemahkan label keluaran model fine-tuned (mis. 'sdg3', 'LABEL_2', '3')
    menjadi nomor SDG."""
    teks = str(label).upper()
    cocok = re.search(r"LABEL_(\d+)", teks)
    if cocok:
        return int(cocok.group(1)) + 1  # LABEL_0 = SDG 1
    cocok = re.search(r"\d+", teks)
    if cocok:
        return int(cocok.group())
    return None


def klasifikasi(classifier, task, text):
    """Menghasilkan skor (0-1) untuk SDG 1-17."""
    skor = {no: 0.0 for no in range(1, 18)}

    if task == "zero-shot-classification":
        raw = classifier(
            text,
            SDG_LABELS,
            multi_label=True,
            hypothesis_template="Teks dokumen ini membahas topik tentang {}.",
            truncation=True,
            max_length=512,
        )
        hasil = raw[0] if isinstance(raw, list) else raw
        for label, nilai in zip(hasil["labels"], hasil["scores"]):
            skor[LABEL_KE_NOMOR[label]] = float(nilai)
    else:
        raw = classifier(text, top_k=None, truncation=True, max_length=512)
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]
        for item in raw:
            no = nomor_sdg_dari_label(item["label"])
            if no and 1 <= no <= 17:
                skor[no] = max(skor[no], float(item["score"]))

    return skor


def ambil_teks_pdf():
    """Mengambil teks dari 2 halaman pertama PDF lalu memasukkannya ke kotak teks."""
    file_pdf = st.session_state.get("file_pdf")
    if file_pdf is None:
        return
    try:
        file_pdf.seek(0)
        with pdfplumber.open(file_pdf) as pdf:
            halaman = pdf.pages[:2]
            teks = "\n".join((h.extract_text() or "") for h in halaman).strip()
    except Exception:
        st.session_state["pdf_status"] = "error"
        return

    if teks:
        st.session_state["input_text"] = teks
        st.session_state["pdf_status"] = "ok"
    else:
        st.session_state["pdf_status"] = "kosong"


def tampilkan_hasil(hasil):
    info = MODEL_REGISTRY[hasil["model"]]
    cakupan = set(info["cakupan"])

    df = pd.DataFrame(
        {
            "No": list(range(1, 18)),
            "SDG": [NOMOR_KE_LABEL[n] for n in range(1, 18)],
            "Skor (%)": [hasil["skor"][n] * 100 for n in range(1, 18)],
            "Keterangan": ["" if n in cakupan else "Tidak dicakup model" for n in range(1, 18)],
        }
    )
    df_urut = df.sort_values("Skor (%)", ascending=False).reset_index(drop=True)
    df_dicakup = df_urut[df_urut["No"].isin(cakupan)]

    st.success(f"Analisis selesai dengan **{hasil['model']}** ({hasil['waktu']:.1f} detik).")

    # ---------- Top 3 ----------
    st.markdown("#### 🏆 Top 3 SDG Paling Relevan")
    for _, row in df_dicakup.head(3).iterrows():
        warna = NOMOR_KE_WARNA[row["No"]]
        nilai = row["Skor (%)"]
        st.markdown(
            f"""
            <div class="sdg-row" style="border-left: 5px solid {warna};">
                <div class="sdg-row-head">
                    <span>{row['SDG']}</span>
                    <span style="color: {warna};">{nilai:.2f}%</span>
                </div>
                <div class="sdg-track">
                    <div style="background: {warna}; height: 100%; width: {min(nilai, 100.0)}%; border-radius: 4px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if info["task"] == "zero-shot-classification":
        st.caption(
            "Skor zero-shot dihitung terpisah untuk setiap SDG (0–100%), sehingga jumlahnya tidak harus 100%."
        )
    else:
        st.caption(
            "Skor model fine-tuned dibagi ke SDG yang dicakup model. "
            "Skor antar-model tidak dapat dibandingkan secara langsung."
        )

    # ---------- Grafik ----------
    st.markdown("#### 📈 Visualisasi Distribusi Skor")
    tab_radar, tab_bar = st.tabs(["🕸️ Radar Chart (17 SDGs)", "📊 Bar Chart (5 Besar)"])

    with tab_radar:
        r = df["Skor (%)"].tolist()
        theta = [f"SDG {n}" for n in df["No"]]
        batas_atas = min(100.0, max(max(r) * 1.15, 5.0))
        fig_radar = go.Figure(
            go.Scatterpolar(
                r=r + [r[0]],
                theta=theta + [theta[0]],
                fill="toself",
                fillcolor="rgba(0, 86, 179, 0.25)",
                line=dict(color="#0056b3", width=2),
                hovertemplate="%{theta}: %{r:.2f}%<extra></extra>",
            )
        )
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, batas_atas], ticksuffix="%"),
                angularaxis=dict(rotation=90, direction="clockwise"),
            ),
            showlegend=False,
            height=420,
            margin=dict(l=40, r=40, t=30, b=30),
        )
        st.plotly_chart(fig_radar, **LEBAR_PENUH)

    with tab_bar:
        top_5 = df_dicakup.head(5).iloc[::-1]
        fig_bar = go.Figure(
            go.Bar(
                y=top_5["SDG"],
                x=top_5["Skor (%)"],
                orientation="h",
                marker_color=[NOMOR_KE_WARNA[n] for n in top_5["No"]],
                text=[f"{v:.1f}%" for v in top_5["Skor (%)"]],
                textposition="auto",
                hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
            )
        )
        fig_bar.update_layout(
            xaxis_title="Skor Relevansi (%)",
            yaxis=dict(automargin=True),
            height=380,
            margin=dict(l=10, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_bar, **LEBAR_PENUH)

    # ---------- Tabel lengkap ----------
    with st.expander("🔍 Rincian Skor 17 SDGs"):
        tabel = df_urut[["SDG", "Skor (%)", "Keterangan"]].copy()
        tabel.index = range(1, len(tabel) + 1)
        st.dataframe(tabel.style.format({"Skor (%)": "{:.2f}%"}), **LEBAR_PENUH)

        unduh = df_urut[["No", "SDG", "Skor (%)", "Keterangan"]].copy()
        unduh.insert(0, "Model", info["path"])
        st.download_button(
            "⬇️ Unduh Hasil (CSV)",
            data=unduh.to_csv(index=False).encode("utf-8-sig"),
            file_name="hasil_malaqbi.csv",
            mime="text/csv",
            **LEBAR_PENUH,
        )

    # ---------- Interpretasi ----------
    teratas = df_dicakup.iloc[0]
    st.info(
        f"Teks yang dianalisis paling dekat dengan **{teratas['SDG']}** "
        f"(skor {teratas['Skor (%)']:.1f}%).\n\n"
        "Skor menunjukkan kedekatan makna teks dengan tema setiap SDG, bukan ukuran dampak riil "
        "penelitian. Gunakan hasil ini sebagai bahan pertimbangan, lalu konfirmasi oleh peneliti "
        "atau pakar bidang terkait."
    )


# ============================================================
# 6. HEADER
# ============================================================
st.markdown(
    """
    <div class="main-header">
        <h1>🌊 PLATFORM MALAQBI</h1>
        <p>Mapping Academic Literature And Qualifying Broad Impact — <b>SDGs Center Universitas Sulawesi Barat</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 7. SIDEBAR
# ============================================================
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/"
        "SustainableDevelopmentGoalsLogoAlone.svg/960px-SustainableDevelopmentGoalsLogoAlone.svg.png"
    )
    st.markdown("### 🏛️ Tentang MALAQBI")
    st.info(
        "MALAQBI adalah platform berbasis **Kecerdasan Buatan (AI)** untuk memetakan keterkaitan "
        "karya ilmiah sivitas akademika Unsulbar dengan 17 Tujuan Pembangunan Berkelanjutan (SDGs)."
    )
    st.markdown("### 🤖 Model yang Tersedia")
    st.markdown(
        "- **mDeBERTa-v3**: zero-shot, multibahasa\n"
        "- **sdgBERT**: fine-tuned, Bahasa Inggris\n"
        "- **OSDG SDG Classifier**: fine-tuned, Bahasa Inggris"
    )
    st.caption("Penjelasan lengkap ada di tab **Metodologi**.")

# ============================================================
# 8. ISI UTAMA
# ============================================================
tab_analisis, tab_metodologi = st.tabs(["🔍 Analisis SDGs", "📖 Metodologi"])

with tab_analisis:
    col_input, col_hasil = st.columns([1, 1.3], gap="large")

    # ---------------- KOLOM KIRI: INPUT ----------------
    with col_input:
        st.markdown("### ⚙️ Pilih Model AI")
        nama_model = st.selectbox(
            "Model klasifikasi:",
            options=list(MODEL_REGISTRY.keys()),
            key="model_dipilih",
        )
        info_model = MODEL_REGISTRY[nama_model]
        st.markdown(
            f"**Jenis:** {info_model['jenis']}  \n"
            f"**Bahasa:** {info_model['bahasa']}  \n"
            f"**Cakupan:** SDG {info_model['cakupan'][0]}–{info_model['cakupan'][-1]}"
        )
        if info_model["task"] == "zero-shot-classification":
            st.success(info_model["catatan"])
        else:
            st.warning(info_model["catatan"])

        st.markdown("---")
        st.markdown("### 📥 Input Dokumen")
        tab_teks, tab_pdf = st.tabs(["📝 Teks (Judul & Abstrak)", "📁 Unggah PDF"])

        with tab_teks:
            text = st.text_area(
                "Tempelkan judul dan abstrak penelitian:",
                height=220,
                key="input_text",
                placeholder=(
                    "Contoh:\n\nJudul: Pengaruh Perubahan Iklim terhadap Produksi Padi di Sulawesi Barat"
                    "\n\nAbstrak: Penelitian ini menganalisis dampak perubahan iklim..."
                ),
            )
            jumlah_kata = len(text.split())
            st.caption(f"Jumlah kata: {jumlah_kata}")
            if jumlah_kata > BATAS_KATA:
                st.warning(
                    f"Teks lebih dari {BATAS_KATA} kata. Model hanya membaca bagian awal teks "
                    "(sekitar 512 token), sisanya diabaikan. Sebaiknya sisakan judul dan abstrak saja."
                )

        with tab_pdf:
            st.caption(
                "Unggah artikel PDF, lalu klik **Ambil Teks dari PDF**. Teks dari 2 halaman pertama "
                "akan dipindahkan ke tab **Teks**. Rapikan teksnya di sana sebelum dianalisis."
            )
            st.file_uploader("Unggah manuskrip (PDF)", type=["pdf"], key="file_pdf")
            st.button(
                "📄 Ambil Teks dari PDF",
                on_click=ambil_teks_pdf,
                disabled=st.session_state.get("file_pdf") is None,
            )
            status = st.session_state.get("pdf_status")
            if status == "ok":
                st.success("Teks berhasil diambil. Buka tab **Teks** untuk memeriksa dan merapikannya.")
            elif status == "kosong":
                st.warning("Tidak ada teks yang terbaca. Kemungkinan PDF ini hasil scan (berupa gambar).")
            elif status == "error":
                st.error("PDF gagal dibuka. Kemungkinan file rusak atau terkunci kata sandi.")

        st.markdown("---")
        tombol_analisis = st.button(
            "🚀 Mulai Analisis SDGs", type="primary", **LEBAR_PENUH
        )

    # ---------------- KOLOM KANAN: HASIL ----------------
    with col_hasil:
        st.markdown("### 📊 Hasil Analisis")

        if tombol_analisis:
            if text.strip() == "":
                st.error("⚠️ Mohon masukkan judul dan abstrak terlebih dahulu.")
            else:
                with st.spinner(
                    f"Menyiapkan {nama_model}... (pemuatan pertama dapat memakan waktu beberapa menit)"
                ):
                    classifier = load_classifier(info_model["path"], info_model["task"])
                with st.spinner("Model AI sedang membaca teks..."):
                    mulai = time.perf_counter()
                    skor = klasifikasi(classifier, info_model["task"], text)
                    durasi = time.perf_counter() - mulai
                st.session_state["hasil"] = {"model": nama_model, "skor": skor, "waktu": durasi}

        if "hasil" in st.session_state:
            tampilkan_hasil(st.session_state["hasil"])
        else:
            st.info("👈 Pilih model, masukkan judul dan abstrak, lalu klik **Mulai Analisis SDGs**.")

with tab_metodologi:
    st.markdown(
        """
### 🔬 Metodologi Klasifikasi MALAQBI

MALAQBI memetakan judul dan abstrak karya ilmiah ke dalam 17 Tujuan Pembangunan Berkelanjutan (SDGs)
menggunakan model bahasa (*language model*) dari Hugging Face Hub. Pengguna dapat memilih satu dari tiga model
dengan pendekatan yang berbeda.

#### Model yang digunakan

| Model | Pendekatan | Bahasa | Cakupan SDG |
|---|---|---|---|
| `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | Zero-Shot (NLI) | Multibahasa, termasuk Indonesia | SDG 1–17 |
| `sadickam/sdg-classification-bert` (sdgBERT) | Fine-Tuned BERT (OSDG Community Dataset) | Inggris | SDG 1–16 |
| `jonas/sdg_classifier_osdg` | Fine-Tuned BERT (data komunitas OSDG) | Inggris | SDG 1–15 |

#### 1. Zero-Shot Classification (mDeBERTa-v3)
- Model **tidak dilatih khusus** untuk SDGs. Model menilai apakah teks mendukung sebuah pernyataan,
  sebuah teknik yang disebut *Natural Language Inference* (NLI).
- Untuk setiap SDG, model menguji pernyataan:
  *"Teks dokumen ini membahas topik tentang {nama SDG}."*
- Setiap SDG dinilai terpisah (*multi-label*), sehingga satu penelitian dapat relevan dengan beberapa SDG sekaligus.
- Karena multibahasa, model ini dapat membaca abstrak Bahasa Indonesia secara langsung.

#### 2. Fine-Tuned Classification (sdgBERT dan OSDG SDG Classifier)
- Model BERT yang telah **dilatih ulang** menggunakan teks berlabel SDG dari komunitas OSDG.
- Model langsung memberikan skor untuk setiap SDG yang dikenalnya.
- Kedua model dilatih dengan teks berbahasa Inggris dan **tidak mencakup seluruh 17 SDG**.

#### Alur proses
1. **Input**: judul dan abstrak (diketik, ditempel, atau diambil dari 2 halaman pertama PDF).
2. **Tokenisasi**: teks dipecah menjadi token; model membaca maksimal 512 token pertama.
3. **Inferensi**: model menghitung skor relevansi untuk setiap SDG.
4. **Output**: skor 0–100%, Top 3 SDG, radar chart, bar chart, dan tabel yang dapat diunduh (CSV).

#### Keterbatasan
- Skor menunjukkan **kedekatan makna** teks dengan tema SDG, bukan ukuran dampak riil penelitian.
- Skor dari model yang berbeda **tidak dapat dibandingkan secara langsung** karena cara perhitungannya berbeda.
- Teks yang lebih panjang dari sekitar 512 token akan terpotong.
- Model fine-tuned kurang andal untuk teks berbahasa Indonesia dan tidak dapat mendeteksi SDG di luar cakupannya.
- Hasil klasifikasi sebaiknya dikonfirmasi oleh peneliti atau pakar bidang terkait (*human-in-the-loop*).
        """
    )

# ============================================================
# 9. FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <p><b>MALAQBI</b> — Platform Pemetaan SDGs Berbasis AI</p>
        <p>Dikembangkan oleh SDGs Center Universitas Sulawesi Barat</p>
        <p style="font-size: 0.8rem;">© 2026 · Powered by CLICKTAP.ID</p>
    </div>
    """,
    unsafe_allow_html=True,
)
