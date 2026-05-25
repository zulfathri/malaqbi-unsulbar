# 🌊 MALAQBI: Mapping Academic Literature And Qualifying Broad Impact

![MALAQBI Banner](https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/SustainableDevelopmentGoalsLogoAlone.svg/960px-SustainableDevelopmentGoalsLogoAlone.svg.png)

**MALAQBI** adalah platform dasbor interaktif berbasis Kecerdasan Buatan (AI) yang dikembangkan oleh **SDGs Center Universitas Sulawesi Barat**. Platform ini berfungsi untuk memetakan dan mengklasifikasikan kontribusi literatur akademik (abstrak publikasi, skripsi, tesis) ke dalam 17 pilar *Sustainable Development Goals* (SDGs) Perserikatan Bangsa-Bangsa.

Nama "Malaqbi" diadaptasi dari filosofi lokal Sulawesi Barat yang bermakna mulia atau baik, merepresentasikan visi universitas dalam mengawal pembangunan berkelanjutan yang berdampak positif bagi masyarakat luas.

## ✨ Fitur Utama
* **Multi-Model AI Architecture:** Mendukung berbagai model bahasa (*Pre-trained Language Models*) dari Hugging Face:
  * `mDeBERTa-v3` (Zero-Shot Multilingual) untuk pemahaman semantik teks lokal.
  * `OSDG Global Community` (Fine-Tuned) berbasis standar komunitas SDGs PBB.
  * `Elsevier SDG Multi-Class` (Fine-Tuned) berbasis pemetaan institusi Elsevier.
* **Interactive Executive Dashboard:** Visualisasi data otomatis menggunakan Radar Chart dan Bar Chart (*Plotly*) dengan palet warna resmi PBB.
* **Bilingual Support:** Mampu memproses teks berbahasa Indonesia maupun Inggris.

## 🛠️ Teknologi yang Digunakan
* **Frontend/UI:** [Streamlit](https://streamlit.io/)
* **AI/Machine Learning:** [Transformers (Hugging Face)](https://huggingface.co/), PyTorch
* **Data Analisis & Visualisasi:** Pandas, Plotly, Matplotlib, Seaborn
* **Pemrosesan Dokumen:** PyPDF2, pdfplumber

## 🚀 Cara Menjalankan di Komputer Lokal (Localhost)
1. Kloning repositori ini ke komputer Anda:
   ```bash
   git clone [https://github.com/username-anda/malaqbi-unsulbar.git](https://github.com/username-anda/malaqbi-unsulbar.git)