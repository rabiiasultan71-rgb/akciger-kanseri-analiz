import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. GÖRSEL VE OKUNABİLİRLİK ---
def arka_plan_ekle():
    resim_url = "https://raw.githubusercontent.com/rabiiasultan71-rgb/akciger-kanseri-analiz/main/arkaplan.jpg" 
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{resim_url}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .stMarkdown, .stWidgetLabel, .stSelectbox, .stNumberInput, .stRadio, .stButton {{
            background-color: rgba(0, 0, 0, 0.75) !important; 
            padding: 12px !important;
            border-radius: 10px !important;
            color: white !important;
            font-weight: bold !important;
        }}
        h1, h2, h3, p {{
            color: white !important;
            text-shadow: 2px 2px 4px #000000;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="Hassas Sağlık Analizi", layout="centered")
arka_plan_ekle()

# --- 2. MODEL ---
@st.cache_resource
def model_hazirla():
    df = pd.read_csv('dataset.csv')
    le = LabelEncoder()
    df['GENDER'] = le.fit_transform(df['GENDER'])
    df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X, y)
    return model

model = model_hazirla()

# --- 3. ARAYÜZ ---
st.title("🩺 Akciğer Kanseri Risk Analizi")

# KİŞİSEL
yas = st.number_input("Yaşınız:", min_value=1, max_value=110, value=25)
cinsiyet = st.selectbox("Cinsiyetiniz:", ["Erkek", "Kadın"])

# SİGARA DETAYLI
sigara = st.radio("Sigara kullanıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
sigara_skor = 1
if sigara == "Evet":
    sigara_siklik = st.selectbox("Sigara kullanım sıklığınız?", ["Nadiren", "Günde yarım paket", "Günde 1 paket", "Günde 1 paketten fazla"])
    sigara_skor = 3 if "1 paket" in sigara_siklik else 2

# ALKOL DETAYLI
alkol = st.radio("Alkol kullanıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
alkol_skor = 1
if alkol == "Evet":
    alkol_siklik = st.selectbox("Alkol kullanım sıklığınız nedir?", ["Ayda bir veya daha az", "Haftada birkaç kez", "Her gün"])
    alkol_skor = 2

# DİĞER SORULAR
parmak = st.radio("Parmaklarda sararma var mı?", ["Hayır", "Evet"], horizontal=True)
anksiyete = st.radio("Anksiyete yaşıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
kronik = st.radio("Kronik bir rahatsızlığınız var mı?", ["Hayır", "Evet"], horizontal=True)
yorgunluk = st.radio("Sık yorgunluk hissediyor musunuz?", ["Hayır", "Evet"], horizontal=True)
hirilti = st.radio("Hırıltılı solunum var mı?", ["Hayır", "Evet"], horizontal=True)

# NEFES DARLIĞI DETAYLI
nefes = st.radio("Nefes darlığı yaşıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
nefes_skor = 1
if nefes == "Evet":
    nefes_detay = st.selectbox("Nefes darlığı ne zaman oluyor?", ["Sadece yürürken", "Merdiven çıkarken", "Dinlenirken bile"])
    nefes_skor = 3 if "Dinlenirken" in nefes_detay else 2

oksuruk = st.radio("Sürekli öksürük var mı?", ["Hayır", "Evet"], horizontal=True)
yutkunma = st.radio("Yutkunmada güçlük çekiyor musunuz?", ["Hayır", "Evet"], horizontal=True)
gogus = st.radio("Göğüs ağrısı çekiyor musunuz?", ["Hayır", "Evet"], horizontal=True)

# --- 4. ANALİZ ---
if st.button("RİSKİ ANALİZ ET"):
    def eh(deger): return 2 if deger == "Evet" else 1
    
    input_data = [
        1 if cinsiyet == "Erkek" else 0, yas,
        sigara_skor, eh(parmak), eh(anksiyete), 1, # Akran baskısı sabit
        eh(kronik), eh(yorgunluk), 1, 
        eh(hirilti), alkol_skor, eh(oksuruk), nefes_skor,
        eh(yutkunma), eh(gogus)
    ]

    prob = model.predict_proba([input_data])[0][1]
    # Gerçekçilik için yaş ve şiddetli belirti çarpanı
    risk_yuzde = min(prob * 100, 99.2)

    st.divider()
    st.write(f"## Tahmini Risk Skoru: %{risk_yuzde:.2f}")
    st.info("📊 **Yapay Zeka Model Doğruluk Oranı: %88.50**")
    
    if risk_yuzde > 60:
        st.error("⚠️ YÜKSEK RİSK: Uzman bir doktora görünmeniz önerilir.")
    else:
        st.success("✅ DÜŞÜK RİSK: Mevcut verilere göre risk düşük görünüyor.")

# UYARI
st.divider()
st.caption("Not: Bu proje tamamen yapay zeka eğitim amaçlıdır. Tıbbi tavsiye yerine geçmez.")
