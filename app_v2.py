import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.graph_objects as go  # <--- TAMBAHAN BARU: Untuk grafik canggih & Radar Chart

# --- 1. CONFIG HALAMAN TINGKAT TINGGI ---
st.set_page_config(page_title="MALAQBI - SDGs Center Unsulbar", page_icon="🌊", layout="wide")

# --- 2. KODE WARNA RESMI UN SDGs (STANDAR GLOBAL PBB) ---
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


# --- 3. MESIN AI (CACHE) ---
@st.cache_resource
def load_classifier():
    return pipeline(
        task="zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device=-1
    )


classifier = load_classifier()

# --- 4. TAMPILAN HEADER UTAMA (SANGAT ELEGAN) ---
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

# Kotak Tentang Aplikasi yang Rapi
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/d/df/Sustainable_Development_Goals.png",
        use_column_width=True)
    st.markdown("### 🏛️ Tentang MALAQBI")
    st.info(
        "MALAQBI adalah sistem berbasis **Kecerdasan Buatan (AI)** untuk mengukur seberapa besar dampak dan "
        "kontribusi karya ilmiah civitas akademika Unsulbar terhadap 17 pilar Pembangunan Berkelanjutan (SDGs) PBB."
    )
    st.markdown("---")
    st.markdown("**Core Engine:** `mDeBERTa-v3 Multilingual NLP`")
    st.markdown("**Methodology:** `Normalized Softmax Distribution`")

# --- 5. TATA LETAK WORKSPACE (2 KOLOM UTAMA) ---
col_input, col_dashboard = st.columns([1, 1.3])

# --- KOLOM KIRI: AREA INPUT DOKUMEN ---
with col_input:
    st.markdown("### 📥 Input Dokumen Akademik")

    tab1, tab2 = st.tabs(["📝 Tempel Teks (Judul & Abstrak)", "📁 Unggah File PDF"])

    with tab1:
        text = st.text_area("Tempelkan Judul dan Abstrak Penelitian di sini:", height=250,
                            placeholder="Contoh: Desain model pendataan sumberdaya perikanan berbasis masyarakat...")

    with tab2:
        st.file_uploader("Unggah Manuskrip (Format PDF)", type=["pdf"])
        st.caption("Catatan: Ekstraksi teks PDF otomatis sedang disiapkan untuk fase integrasi sistem selanjutnya.")

    st.markdown("---")
    submit_button = st.button("🚀 Mulai Analisis Dampak SDGs", type="primary", use_container_width=True)

# --- KOLOM KANAN: DASBOR EXECUTIVE HASIL ANALISIS ---
with col_dashboard:
    st.markdown("### 📊 Dasbor Analisis Kontribusi SDGs")

    if submit_button:
        if text.strip() == "":
            st.error("⚠️ Kolom teks kosong! Mohon masukkan judul dan abstrak di kolom sebelah kiri terlebih dahulu.")
        else:
            with st.spinner("Mengaktifkan Mesin AI mDeBERTa untuk membaca konteks teks..."):

                # Menjalankan Klasifikasi AI
                raw_result = classifier(text, SDG_LABELS, multi_label=False, hypothesis_template="Teks dokumen ini membahas topik tentang {}.")
                result = raw_result[0] if isinstance(raw_result, list) else raw_result

                # Membuat DataFrame Hasil
                df = pd.DataFrame({
                    "SDG": result["labels"],
                    "Skor (%)": [score * 100 for score in result["scores"]]
                })

                # Menyortir data dari tertinggi ke terendah
                df = df.sort_values(by="Skor (%)", ascending=False).reset_index(drop=True)
                top_3 = df.head(3)

                st.success("Analisis Berhasil Diselesaikan!")

                # --- A. HIGHLIGHT TOP 3 DENGAN WARNA RESMI PBB ---
                st.markdown("#### 🏆 Top 3 Pilar Kontribusi Utama:")

                for index, row in top_3.iterrows():
                    sdg_name = row['SDG']
                    score_val = row['Skor (%)']
                    official_color = SDG_COLORS[sdg_name]

                    # Membuat progress bar kustom menggunakan HTML/CSS agar warnanya akurat sesuai standar PBB
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 15px; background: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 5px solid {official_color};">
                            <div style="display: flex; justify-content: space-between; font-weight: bold; margin-bottom: 5px;">
                                <span style="color: #333;">{sdg_name}</span>
                                <span style="color: {official_color};">{score_val:.2f}%</span>
                            </div>
                            <div style="background-color: #e9ecef; border-radius: 4px; height: 10px; width: 100%;">
                                <div style="background-color: {official_color}; height: 100%; width: {min(score_val * 4, 100.0)}%; border-radius: 4px;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("---")

                # --- B. MULTI-VISUALIZATION (TAB UNTUK GRAFIK BERBEDA) ---
                st.markdown("#### 📈 Visualisasi Distribusi Capaian")
                tab_radar, tab_bar = st.tabs(["🕸️ Radar Chart (Holistik)", "📊 Bar Chart (5 Besar)"])

                # 1. Pembuatan Radar Chart (Sangat cocok untuk presentasi pimpinan)
                with tab_radar:
                    # Plotly membutuhkan loop data yang tertutup agar grafiknya menyambung kembali ke awal
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
                            radialaxis=dict(visible=True, range=[0, max(df['Skor (%)']) + 2])
                        ),
                        showlegend=False,
                        height=400,
                        margin=dict(l=40, r=40, t=20, b=20)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
                    st.caption("Grafik Radar menampilkan sebaran kontribusi naskah di seluruh 17 pilar SDGs sekaligus.")

                # 2. Pembuatan Kustom Bar Chart (Warna batang otomatis menyesuaikan standar PBB)
                    # 2. Pembuatan Kustom Bar Chart (Warna batang otomatis menyesuaikan standar PBB)
                with tab_bar:
                    top_5 = df.head(5).iloc[::-1]  # Dibalik agar yang tertinggi ada di atas grafik horizontal

                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        y=top_5['SDG'],
                        x=top_5['Skor (%)'],
                        orientation='h',
                        # --- PERBAIKAN DI SINI: Menggunakan cara standar Pandas yang lebih rapi ---
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

                # --- C. TABEL RINCIAN LENGKAP DI BAGIAN BAWAH ---
                with st.expander("🔍 Lihat Hasil Rincian Data 17 SDGs Lengkap"):
                    st.dataframe(df.style.format({"Skor (%)": "{:.2f}%"}), use_container_width=True)
    else:
        # Tampilan Awal Sebelum Klik Analisis
        st.info(
            "👈 Silakan masukkan judul dan abstrak riset Anda di kolom kiri, lalu klik tombol **Mulai Analisis Dampak SDGs** untuk memunculkan dasbor eksekutif.")