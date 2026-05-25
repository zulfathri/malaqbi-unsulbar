import streamlit as st
from transformers import pipeline
import pandas as pd

# --- 1. PENGATURAN HALAMAN DASAR ---
st.set_page_config(page_title="MALAQBI-Unsulbar", page_icon="🌊", layout="wide")

# --- 2. HEADER & ABOUT SECTION (Meniru Web Pertamina) ---
st.markdown("<h1 style='text-align: center; color: #0056b3;'>MALAQBI UNSULBAR</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #555555;'>SDG Mapping and Assessment for Research and Tracking</h4>",
            unsafe_allow_html=True)
st.markdown("---")

with st.expander("ℹ️ About MALAQBI Unsulbar", expanded=True):
    st.write("""
    **Platform ini memungkinkan Anda untuk mengklasifikasikan artikel/riset ke dalam Sustainable Development Goals (SDGs).** MALAQBI Unsulbar memanfaatkan kecerdasan buatan (AI) multibahasa yang dapat mendeteksi relevansi teks berdasarkan kerangka kerja SDGs global. 
    Aplikasi ini dirancang khusus untuk memetakan kontribusi riset sivitas akademika Universitas Sulawesi Barat terhadap target keberlanjutan.
    """)

# --- 3. DAFTAR SDGs ---
SDG_LABELS = [
    "No Poverty", "Zero Hunger", "Good Health and Well-being",
    "Quality Education", "Gender Equality", "Clean Water and Sanitation",
    "Affordable and Clean Energy", "Decent Work and Economic Growth",
    "Industry, Innovation and Infrastructure", "Reduced Inequalities",
    "Sustainable Cities and Communities", "Responsible Consumption and Production",
    "Climate Action", "Life Below Water", "Life on Land",
    "Peace, Justice and Strong Institutions", "Partnerships for the Goals"
]


# --- 4. MESIN AI ---
@st.cache_resource
def load_classifier():
    return pipeline(
        task="zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device=-1
    )


classifier = load_classifier()

# --- 5. TATA LETAK UTAMA (Membagi Layar) ---
col_input, col_result = st.columns([1.2, 1])  # Kolom kiri sedikit lebih lebar

# --- KOLOM KIRI: AREA INPUT & PENGATURAN ---
with col_input:
    st.subheader("📥 Input Dokumen")

    # Membuat Tab meniru web Pertamina
    tab1, tab2 = st.tabs(["📝 Paste Text / Abstract", "📁 Upload PDF"])

    with tab1:
        text = st.text_area("Masukkan judul dan abstrak riset Anda di sini:", height=200)

    with tab2:
        st.file_uploader("Upload Your Manuscript (PDF only)", type=["pdf"])
        st.info(
            "Fitur pembacaan PDF otomatis sedang dalam tahap pengembangan. Silakan gunakan tab 'Paste Text' untuk saat ini.")

    # Kotak Informasi Model (Meniru desain web Pertamina)
    st.markdown("---")
    st.markdown("**⚙️ Model Klasifikasi SDG**")
    st.info(
        "**mDeBERTa-v3 Multi-Label Model**\n\nMengklasifikasikan teks ke dalam 17 tujuan SDG. Mendukung pemrosesan Bahasa Indonesia secara native.")

    # Tombol Eksekusi
    submit_button = st.button("🔍 Classify Text", type="primary", use_container_width=True)

# --- KOLOM KANAN: AREA HASIL (DASHBOARD) ---
with col_result:
    st.subheader("📊 Hasil Analisis")

    if submit_button:
        if text.strip() == "":
            st.error("⚠️ Mohon masukkan teks abstrak terlebih dahulu pada tab di sebelah kiri.")
        else:
            with st.spinner("🤖 Mengklasifikasikan teks Anda..."):

                # Menjalankan AI
                result = classifier(text, SDG_LABELS, multi_label=False)

                # Menyusun DataFrame
                df = pd.DataFrame({
                    "SDG": result["labels"],
                    "Skor (%)": [score * 100 for score in result["scores"]]
                })

                # Mengambil Top 3 untuk di-highlight
                top_3 = df.head(3)

                st.success("Analisis Selesai!")
                st.markdown("**🏆 Top 3 SDG Paling Relevan:**")

                # Menampilkan Top 3 dengan Progress Bar yang cantik
                for index, row in top_3.iterrows():
                    # Menulis label dan skor
                    st.write(f"**{row['SDG']}** ({row['Skor (%)']:.1f}%)")
                    # Membuat progress bar (nilai harus dibagi 100 karena progress bar Streamlit menerima angka 0.0 - 1.0)
                    progress_val = min(row['Skor (%)'] / 100.0, 1.0)
                    # Jika skor bagi kue kecil, kita kali 2 hanya untuk tampilan visual progress bar agar terlihat lebih panjang
                    st.progress(min(progress_val * 3, 1.0))

                st.markdown("---")

                # Grafik Batang untuk 5 Besar
                st.markdown("**📈 Visualisasi 5 Besar**")
                st.bar_chart(data=df.head(5).set_index("SDG"))

                # Menu Lipat untuk semua skor
                with st.expander("Lihat Rincian Seluruh 17 SDG"):
                    # Menampilkan tabel data yang rapi
                    st.dataframe(df.style.format({"Skor (%)": "{:.2f}%"}), use_container_width=True)
    else:
        # Tampilan kosong sebelum tombol diklik
        st.info("👈 Masukkan teks dan klik tombol **Classify Text** untuk melihat hasil pemetaan di sini.")