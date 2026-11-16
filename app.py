import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="MALAQBI - SMART Unsulbar",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# WARNA RESMI SDGs (UN Official Colors)
# ============================================
SDG_COLORS = {
    "No Poverty": "#E5243B",
    "Zero Hunger": "#DDA63A",
    "Good Health and Well-being": "#4C9F38",
    "Quality Education": "#C5192D",
    "Gender Equality": "#FF3A21",
    "Clean Water and Sanitation": "#26BDE2",
    "Affordable and Clean Energy": "#FCC30B",
    "Decent Work and Economic Growth": "#A21942",
    "Industry, Innovation and Infrastructure": "#FD6925",
    "Reduced Inequalities": "#DD1367",
    "Sustainable Cities and Communities": "#FD9D24",
    "Responsible Consumption and Production": "#BF8B2E",
    "Climate Action": "#3F7E44",
    "Life Below Water": "#0A97D9",
    "Life on Land": "#56C02B",
    "Peace, Justice and Strong Institutions": "#00689D",
    "Partnerships for the Goals": "#19486A"
}

# ============================================
# CSS CUSTOM STYLING
# ============================================
st.markdown("""
<style>
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #0056b3 0%, #003d82 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-title {
        color: white;
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .subtitle {
        color: #e0e0e0;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Card Styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #0056b3;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    
    /* SDG Badge */
    .sdg-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        margin: 0.3rem;
        font-size: 0.9rem;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #0056b3 0%, #003d82 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# HEADER
# ============================================
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🌊 MALAQBI - SMART UNSULBAR</h1>
    <p class="subtitle">SDG Mapping and Assessment for Research and Tracking</p>
    <p class="subtitle" style="font-size: 0.9rem; margin-top: 0.3rem;">
        Pusat Studi SDGs Center - Universitas Sulawesi Barat
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR - About & Information
# ============================================
with st.sidebar:
    #st.image("https://via.placeholder.com/300x100/0056b3/FFFFFF?text=UNSULBAR", 
    #         use_container_width=True)
    
    st.markdown("### 📋 Tentang MALAQBI")
    st.info("""
    **MALAQBI** (Mapping Academic Literature and Qualifying Broad Impact) 
    adalah platform berbasis AI yang mengklasifikasikan riset akademik ke dalam 
    17 Tujuan Pembangunan Berkelanjutan (SDGs).
    
    Platform ini menggunakan **mDeBERTa-v3**, model AI multibahasa yang dapat 
    memproses teks Bahasa Indonesia secara native.
    """)
    
    st.markdown("### 🎯 Fitur Utama")
    st.markdown("""
    - ✅ Klasifikasi otomatis ke 17 SDGs
    - ✅ Analisis berbasis AI multibahasa
    - ✅ Visualisasi interaktif
    - ✅ Mendukung Bahasa Indonesia
    """)
    
    st.markdown("### 📊 Model AI")
    st.code("mDeBERTa-v3-base-mnli-xnli")
    st.caption("Zero-Shot Classification dengan Natural Language Inference")
    
    st.markdown("---")
    st.markdown("### 👥 Tim Pengembang")
    st.markdown("""
    **Ketua:** Ihsan Arham, S.P., M.Si.  
    **Anggota:** Zulfathri Randhi, S.Pi., M.Si.
    """)

# ============================================
# DAFTAR SDGs
# ============================================
SDG_LABELS = [
    "No Poverty", "Zero Hunger", "Good Health and Well-being",
    "Quality Education", "Gender Equality", "Clean Water and Sanitation",
    "Affordable and Clean Energy", "Decent Work and Economic Growth",
    "Industry, Innovation and Infrastructure", "Reduced Inequalities",
    "Sustainable Cities and Communities", "Responsible Consumption and Production",
    "Climate Action", "Life Below Water", "Life on Land",
    "Peace, Justice and Strong Institutions", "Partnerships for the Goals"
]

# ============================================
# LOAD AI MODEL (dengan caching)
# ============================================
@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline(
        task="zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device=-1
    )

with st.spinner("🤖 Memuat model AI... (pertama kali akan memakan waktu lebih lama)"):
    classifier = load_classifier()

st.success("✅ Model AI siap digunakan!")

# ============================================
# MAIN CONTENT - INPUT AREA
# ============================================
st.markdown("## 📝 Input Dokumen Riset")

# Tabs untuk input
tab_text, tab_pdf, tab_metodologi = st.tabs([
    "📝 Input Teks/Abstrak", 
    "📁 Upload PDF",
    "📖 Metodologi"
])

with tab_text:
    st.markdown("""
    <div class="info-card">
        <strong>📌 Panduan:</strong> Masukkan judul dan abstrak penelitian Anda. 
        Semakin lengkap informasi yang diberikan, semakin akurat hasil klasifikasi.
    </div>
    """, unsafe_allow_html=True)
    
    text = st.text_area(
        "Masukkan Judul dan Abstrak Riset Anda:",
        height=250,
        placeholder="Contoh:\n\nJudul: Pengaruh Perubahan Iklim terhadap Produksi Padi di Sulawesi Barat\n\nAbstrak: Penelitian ini menganalisis dampak perubahan iklim...",
        help="Paste judul dan abstrak penelitian Anda di sini"
    )

with tab_pdf:
    st.markdown("""
    <div class="info-card">
        <strong>🚧 Status:</strong> Fitur upload PDF sedang dalam pengembangan.
        Untuk saat ini, silakan gunakan tab "Input Teks/Abstrak".
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload File PDF (Coming Soon)", 
        type=["pdf"],
        disabled=True
    )

with tab_metodologi:
    st.markdown("""
    ### 🔬 Metodologi Klasifikasi
    
    #### Model: mDeBERTa-v3 (Multilingual)
    Platform MALAQBI menggunakan **DeBERTa-v3** (Decoding-enhanced BERT with 
    disentangled attention) yang telah di-train pada dataset MNLI dan XNLI.
    
    #### Teknik: Zero-Shot Classification
    - **Zero-Shot** artinya model dapat mengklasifikasikan teks tanpa perlu training 
      spesifik untuk SDGs
    - Model memahami **konteks semantik** dari abstrak riset
    - Menggunakan **Natural Language Inference (NLI)** untuk mengevaluasi 
      kesesuaian teks dengan setiap SDG
    
    #### Proses Klasifikasi:
    1. **Input**: Abstrak penelitian dalam Bahasa Indonesia/Inggris
    2. **Encoding**: Teks dikonversi menjadi representasi vektor
    3. **Inference**: Model mengevaluasi kesesuaian dengan 17 SDGs
    4. **Output**: Skor probabilitas untuk setiap SDG (0-100%)
    
    #### Keunggulan:
    - ✅ Mendukung Bahasa Indonesia secara native
    - ✅ Akurasi tinggi (79.8% pada benchmark XNLI)
    - ✅ Tidak perlu dataset training lokal
    - ✅ Open-source dan transparan
    """)

# Tombol Klasifikasi
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    submit_button = st.button("🔍 Klasifikasi Teks", use_container_width=True)

# ============================================
# RESULTS AREA
# ============================================
if submit_button:
    if text.strip() == "":
        st.error("⚠️ **Mohon masukkan teks abstrak terlebih dahulu!**")
    else:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🤖 Menganalisis teks...")
        progress_bar.progress(25)
        
        # Jalankan AI
        result = classifier(text, SDG_LABELS, multi_label=False)
        progress_bar.progress(75)
        
        # Susun DataFrame
        df = pd.DataFrame({
            "SDG": result["labels"],
            "Skor (%)": [score * 100 for score in result["scores"]],
            "Nomor": [SDG_LABELS.index(label) + 1 for label in result["labels"]]
        })
        
        progress_bar.progress(100)
        status_text.text("✅ Analisis selesai!")
        
        st.balloons()
        
        # ============================================
        # HASIL ANALISIS
        # ============================================
        st.markdown("---")
        st.markdown("## 📊 Hasil Analisis SDGs")
        
        # Top 3 SDG Cards
        st.markdown("### 🏆 Top 3 SDG Paling Relevan")
        
        top_3 = df.head(3)
        cols = st.columns(3)
        
        for idx, (col, (_, row)) in enumerate(zip(cols, top_3.iterrows())):
            with col:
                sdg_color = SDG_COLORS[row['SDG']]
                st.markdown(f"""
                <div style="
                    background: {sdg_color};
                    padding: 1.5rem;
                    border-radius: 10px;
                    text-align: center;
                    color: white;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
                ">
                    <h2 style="margin: 0; font-size: 2.5rem;">#{idx + 1}</h2>
                    <h3 style="margin: 0.5rem 0; font-size: 1.1rem;">{row['SDG']}</h3>
                    <p style="margin: 0; font-size: 2rem; font-weight: bold;">
                        {row['Skor (%)']:.1f}%
                    </p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.9;">
                        SDG {row['Nomor']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ============================================
        # VISUALISASI
        # ============================================
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown("### 🕸️ Radar Chart - Semua SDGs")
            
            # Radar Chart dengan Plotly
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=df['Skor (%)'],
                theta=df['Nomor'].apply(lambda x: f"SDG {x}"),
                fill='toself',
                name='Skor SDGs',
                line_color='#0056b3',
                fillcolor='rgba(0, 86, 179, 0.3)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        ticksuffix="%"
                    )
                ),
                showlegend=False,
                height=500,
                margin=dict(l=80, r=80, t=50, b=50)
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with viz_col2:
            st.markdown("### 📊 Bar Chart - Top 10 SDGs")
            
            # Bar Chart dengan warna SDGs
            top_10 = df.head(10).copy()
            top_10['Color'] = top_10['SDG'].map(SDG_COLORS)
            
            fig_bar = go.Figure()
            
            fig_bar.add_trace(go.Bar(
                x=top_10['Skor (%)'],
                y=top_10['SDG'],
                orientation='h',
                marker=dict(
                    color=top_10['Color'],
                    line=dict(color='rgba(0,0,0,0.2)', width=1)
                ),
                text=top_10['Skor (%)'].apply(lambda x: f"{x:.1f}%"),
                textposition='outside'
            ))
            
            fig_bar.update_layout(
                xaxis_title="Skor Relevansi (%)",
                yaxis_title="",
                height=500,
                margin=dict(l=250, r=50, t=50, b=50),
                yaxis=dict(autorange="reversed")
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # ============================================
        # TABEL DETAIL
        # ============================================
        st.markdown("---")
        st.markdown("### 📋 Rincian Lengkap 17 SDGs")
        
        # Format tabel
        df_display = df.copy()
        df_display['Skor (%)'] = df_display['Skor (%)'].apply(lambda x: f"{x:.2f}%")
        df_display['SDG'] = df_display.apply(
            lambda row: f"SDG {row['Nomor']}: {row['SDG']}", axis=1
        )
        df_display = df_display[['SDG', 'Skor (%)']]
        df_display.index = range(1, len(df_display) + 1)
        
        st.dataframe(
            df_display,
            use_container_width=True,
            height=400
        )
        
        # ============================================
        # INTERPRETASI
        # ============================================
        st.markdown("---")
        st.markdown("### 💡 Interpretasi Hasil")
        
        top_sdg = top_3.iloc[0]
        
        st.info(f"""
        **Riset Anda paling relevan dengan SDG {top_sdg['Nomor']}: {top_sdg['SDG']}** 
        dengan skor kepercayaan **{top_sdg['Skor (%)']:.1f}%**.
        
        Ini menunjukkan bahwa penelitian Anda memiliki kontribusi signifikan terhadap 
        pencapaian tujuan pembangunan berkelanjutan tersebut.
        
        **Rekomendasi:**
        - Sertakan temuan ini dalam publikasi Anda
        - Gunakan sebagai dasar untuk aplikasi riset berkelanjutan
        - Pertimbangkan kolaborasi dengan peneliti di bidang terkait
        """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <p><strong>MALAQBI v2.0</strong> - Platform Pemetaan SDGs Berbasis AI</p>
    <p>Dikembangkan oleh SDGs Center Universitas Sulawesi Barat</p>
    <p style="font-size: 0.85rem; color: #999;">
        © 2026 Universitas Sulawesi Barat | Model: mDeBERTa-v3 | 
        Lisensi: Open Source
    </p>
</div>
""", unsafe_allow_html=True)
