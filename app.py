import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.graph_objects as go
import re  # <--- TAMBAHAN BARU: Untuk membaca pola label dari model spesialis

# --- 1. CONFIG HALAMAN TINGKAT TINGGI ---
st.set_page_config(page_title="MALAQBI - SDGs Center Unsulbar", page_icon="🌊", layout="wide")

# --- 2. KODE WARNA & LABEL BILINGUAL (Sesuai Standar PBB) ---
SDG_COLORS = {
    "SDG 1: No Poverty (Tanpa Kemiskinan)": "#E5243B",
    "SDG 2: Zero Hunger (Ketahanan Pangan)": "#DDA63A",
    "SDG 3: Good Health (Kesehatan & Kesejahteraan)": "#4C9F38",
    "SDG 4: Quality Education (Pendidikan)": "#C5192D",
    "SDG 5: Gender Equality (Kesetaraan Gender)": "#FF3A21",
    "SDG 6: Clean Water (Air Bersih & Sanitasi)": "#26BDE2",
    "SDG 7: Clean Energy (Energi Terbarukan)": "#FCC30B",
    "SDG 8: Economic Growth (Pekerjaan & Ekonomi)": "#A21942",
    "SDG 9: Industry & Innovation (Inovasi & Infrastruktur)": "#FD6925",
    "SDG 10: Reduced Inequalities (Mengurangi Ketimpangan)": "#DD1367",
    "SDG 11: Sustainable Cities (Kota & Pemukiman Berkelanjutan)": "#FD9D24",
    "SDG 12: Responsible Consumption (Konsumsi & Produksi)": "#BF8B2E",
    "SDG 13: Climate Action (Aksi Iklim)": "#3F7E44",
    "SDG 14: Life Below Water (Ekosistem Laut & Perikanan)": "#0A97D9",
    "SDG 15: Life on Land (Ekosistem Darat & Kehutanan)": "#56C02B",
    "SDG 16: Peace & Justice (Keadilan & Kelembagaan)": "#00689D",
    "SDG 17: Partnerships (Kemitraan Global)": "#19486A"
}
SDG_LABELS = list(SDG_COLORS.keys())

# --- 3. DAFTAR PILIHAN MODEL AI ASLI (HUGGING FACE HUB) ---
MODEL_REGISTRY = {
    "🧠 mDeBERTa-v3 (Zero-Shot Multilingual)": {
        "path": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "task": "zero-shot-classification"
    },
    "📊 Elsevier SDG Multi-Class (Fine-Tuned)": {
        "path": "sadickam/sdg-classification-bert",
        "task": "text-classification"
    },
    "🔬 OSDG Global Community (Fine-Tuned)": {
        "path": "jonas/sdg_classifier_osdg",
        "task": "text-classification"
    }
}


# --- 4. MESIN AI DINAMIS (MEMBACA TASK BERBEDA) ---
@st.cache_resource(show_spinner=False)
def load_classifier(model_path, model_task):
    return pipeline(
        task=model_task,
        model=model_path,
        device=-1
    )


# --- 5. TAMPILAN HEADER UTAMA ---
st.markdown(
    """
    <div style="background-color: #003366; padding: 20px; border-radius: 10px; margin-bottom: 25px;">
        <h1 style="text-align: center; color: white; margin-bottom: 0px;">PLATFORM MALAQBI</h1>
        <p style="text-align: center; color: #d1ecf1; font-size: 16px; margin-top: 5px;">
            Mapping Academic Literature And Qualifying Broad Impact — <b>SDGs Center Universitas Sulawesi Barat</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/SustainableDevelopmentGoalsLogoAlone.svg/960px-SustainableDevelopmentGoalsLogoAlone.svg.png",
        use_column_width=True)
    st.markdown("### 🏛️ Tentang MALAQBI")
    st.info(
        "MALAQBI adalah sistem berbasis **Kecerdasan Buatan (AI)** untuk mengukur seberapa besar dampak dan "
        "kontribusi karya ilmiah sivitas akademika Unsulbar terhadap 17 pilar Pembangunan Berkelanjutan (SDGs) PBB."
    )
    st.markdown("---")
    st.markdown("**(c) 2024 SDGs Center Unsulbar**")

# --- 6. TATA LETAK WORKSPACE (2 KOLOM UTAMA) ---
col_input, col_dashboard = st.columns([1, 1.3])

with col_input:
    st.markdown("### ⚙️ Pengaturan Mesin AI")

    selected_model_name = st.selectbox(
        "Pilih Model Klasifikasi Asli:",
        options=list(MODEL_REGISTRY.keys()),
        help="Pilih algoritma AI yang ingin digunakan."
    )

    if "mDeBERTa" in selected_model_name:
        st.success("Tipe: **Zero-Shot**. Sangat baik untuk teks Bahasa Indonesia umum dan analisis semantik baru.")
    elif "OSDG" in selected_model_name:
        st.info(
            "Tipe: **Multi-Label Fine-Tuned**. Model terlatih dari komunitas OSDG PBB. Sangat peka terhadap teks publikasi ilmiah global.")
    else:
        st.warning(
            "Tipe: **Multi-Class Fine-Tuned**. Model klasifikasi berbasis kerangka pemetaan institusi. Membagi probabilitas ke seluruh target.")

    st.markdown("---")
    st.markdown("### 📥 Input Dokumen Akademik")

    tab1, tab2 = st.tabs(["📝 Tempel Teks (Judul & Abstrak)", "📁 Unggah File PDF"])

    with tab1:
        text = st.text_area("Tempelkan Judul dan Abstrak Penelitian di sini:", height=200)
    with tab2:
        st.file_uploader("Unggah Manuskrip (Format PDF)", type=["pdf"])

    st.markdown("---")
    submit_button = st.button("🚀 Mulai Analisis Dampak SDGs", type="primary", use_container_width=True)

with col_dashboard:
    st.markdown("### 📊 Dasbor Analisis Kontribusi SDGs")

    if submit_button:
        if text.strip() == "":
            st.error("⚠️ Mohon masukkan judul dan abstrak di kolom sebelah kiri terlebih dahulu.")
        else:
            model_info = MODEL_REGISTRY[selected_model_name]

            with st.spinner(
                    f"Menyiapkan {selected_model_name}... (Pengunduhan pertama mungkin butuh waktu beberapa menit)"):
                classifier = load_classifier(model_info["path"], model_info["task"])

            with st.spinner("Mesin AI sedang membaca teks secara mendalam..."):

                # --- CABANG LOGIKA: ZERO-SHOT VS FINE-TUNED ---
                if model_info["task"] == "zero-shot-classification":
                    # Dokter Umum
                    raw_result = classifier(
                        text,
                        SDG_LABELS,
                        multi_label=True,
                        hypothesis_template="Teks dokumen ini membahas topik tentang {}.",
                        truncation=True,
                        max_length=512
                    )
                    result = raw_result[0] if isinstance(raw_result, list) else raw_result
                    final_labels = result["labels"]
                    final_scores = result["scores"]

                else:
                    # Dokter Spesialis (Aurora / Elsevier)
                    # top_k=None memastikan AI memuntahkan semua 17 nilai, bukan hanya juara 1
                    raw_result = classifier(text, top_k=None, truncation=True, max_length=512)

                    if isinstance(raw_result, list) and isinstance(raw_result[0], list):
                        raw_result = raw_result[0]

                    # Membuat template skor 0.0 untuk ke-17 SDG
                    mapped_scores = {k: 0.0 for k in SDG_LABELS}

                    # Logika Penerjemah Label (Parser)
                    for item in raw_result:
                        lbl = str(item['label']).upper()
                        score = float(item['score'])
                        sdg_num = None

                        # Jika mesin memuntahkan format "LABEL_0"
                        if 'LABEL_' in lbl:
                            match = re.search(r'LABEL_(\d+)', lbl)
                            if match:
                                sdg_num = int(match.group(1)) + 1  # Biasanya LABEL_0 = SDG 1
                        else:
                            # Jika mesin memuntahkan format "SDG_1" atau "1"
                            match = re.search(r'\d+', lbl)
                            if match:
                                sdg_num = int(match.group())

                        # Cocokkan angka yang didapat dengan Kamus UI kita
                        if sdg_num and 1 <= sdg_num <= 17:
                            for key in SDG_LABELS:
                                if f"SDG {sdg_num}:" in key:
                                    mapped_scores[key] = max(mapped_scores[key], score)

                    # Mengurutkan dari tertinggi ke terendah
                    sorted_mapped = sorted(mapped_scores.items(), key=lambda x: x[1], reverse=True)
                    final_labels = [x[0] for x in sorted_mapped]
                    final_scores = [x[1] for x in sorted_mapped]

                # --- MENGGAMBAR DASHBOARD HASIL ---
                df = pd.DataFrame({
                    "SDG": final_labels,
                    "Skor (%)": [score * 100 for score in final_scores]
                })

                top_3 = df.head(3)
                st.success(f"Analisis Selesai Menggunakan **{selected_model_name}**!")
                st.markdown("#### 🏆 Top 3 Pilar Kontribusi Utama:")

                for index, row in top_3.iterrows():
                    sdg_name = row['SDG']
                    score_val = row['Skor (%)']
                    official_color = SDG_COLORS[sdg_name]

                    st.markdown(
                        f"""
                        <div style="margin-bottom: 15px; background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 5px solid {official_color};">
                            <div style="display: flex; justify-content: space-between; font-weight: bold; margin-bottom: 5px;">
                                <span style="color: #333;">{sdg_name}</span>
                                <span style="color: {official_color};">{score_val:.2f}%</span>
                            </div>
                            <div style="background-color: #e9ecef; border-radius: 4px; height: 10px; width: 100%;">
                                <div style="background-color: {official_color}; height: 100%; width: {min(score_val, 100.0)}%; border-radius: 4px;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("---")
                st.markdown("#### 📈 Visualisasi Distribusi Capaian")
                tab_radar, tab_bar = st.tabs(["🕸️ Radar Chart (Holistik)", "📊 Bar Chart (5 Besar)"])

                with tab_radar:
                    radar_df = df.copy()
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=radar_df['Skor (%)'].tolist() + [radar_df['Skor (%)'].iloc[0]],
                        theta=radar_df['SDG'].tolist() + [radar_df['SDG'].iloc[0]],
                        fill='toself',
                        fillcolor='rgba(0, 51, 102, 0.2)',
                        line=dict(color='#003366', width=2)
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, max(df['Skor (%)']) + 5])
                        ),
                        showlegend=False,
                        height=400,
                        margin=dict(l=40, r=40, t=20, b=20)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                with tab_bar:
                    top_5 = df.head(5).iloc[::-1]
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        y=top_5['SDG'],
                        x=top_5['Skor (%)'],
                        orientation='h',
                        marker_color=top_5['SDG'].map(SDG_COLORS).tolist(),
                        text=[f"{val:.1f}%" for val in top_5['Skor (%)']],
                        textposition='inside'
                    ))
                    fig_bar.update_layout(
                        xaxis_title="Persentase Relevansi (%)",
                        height=350,
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                with st.expander("🔍 Lihat Hasil Rincian Data 17 SDGs Lengkap"):
                    st.dataframe(df.style.format({"Skor (%)": "{:.2f}%"}), use_container_width=True)
    else:
        st.info(
            "👈 Silakan pilih model AI, masukkan judul/abstrak riset, lalu klik tombol **Mulai Analisis Dampak SDGs**.")