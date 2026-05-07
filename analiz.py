import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. GÖRSEL AYARLAR (Resmin Gelmesi İçin) ---
def arka_plan_ekle():
    # Senin resminin GitHub'daki gerçek linki
    resim_url = "https://raw.githubusercontent.com/rabiiasultan71-rgb/akciger-kanseri-analiz/main/arkaplan.jpg" 
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{resim_url}");
            background-size: cover;
            background-attachment: fixed;
        }}
        /* Yazıların okunabilirliği için kutucukları hafif beyazlatıyoruz */
        .stMarkdown, .stWidgetLabel, .stSelectbox, .stNumberInput, .stRadio, .stButton {{
            background-color: rgba(255, 255, 255, 0.85); 
            padding: 10px;
            border-radius: 10px;
            color: black;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Sayfa ayarları ve Arka Planı Çalıştır
st.set_page_config(page_title="Hassas Sağlık Analizi", layout="centered")
arka_plan_ekle()

# --- 2. MODEL HAZIRLIĞI ---
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

# --- 3. ARAYÜZ VE SORULAR ---
st.title("🩺 Detaylı Akciğer Sağlığı Analizi")

st.subheader("📍 Kişisel Bilgiler")
yas = st.number_input("Yaşınız:", min_value=1, max_value=110, value=25)
cinsiyet = st.selectbox("Cinsiyetiniz:", ["Erkek", "Kadın"])

st.subheader("📝 Sağlık ve Alışkanlık Anketi")

# SİGARA
sigara = st.radio("Sigara kullanıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
sigara_skor = 1
if sigara == "Evet":
    sigara_siklik = st.selectbox("Sigara kullanım sıklığınız?", ["Nadiren", "Günde yarım paket", "Günde 1 paket", "Günde 1 paketten fazla"])
    sigara_skor = 3 if "1 paket" in sigara_siklik else 2

# ALKOL
alkol = st.radio("Alkol kullanıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
alkol_skor = 1
if alkol == "Evet":
    alkol_siklik = st.selectbox("Alkol kullanım sıklığınız nedir?", ["Ayda bir veya daha az", "Haftada birkaç kez", "Haftada 3-4 gün", "Her gün"])
    alkol_skor = 2

# BELİRTİLER
parmak = st.radio("Parmaklarınızda sararmalar veya sarı lekeler var mı?", ["Hayır", "Evet"], horizontal=True)
anksiyete = st.radio("Anksiyeteniz var mı?", ["Hayır", "Evet"], horizontal=True)
baski = st.radio("Akran baskısına maruz kalıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
kronik = st.radio("Kronik bir rahatsızlığınız var mı?", ["Hayır", "Evet"], horizontal=True)
yorgunluk = st.radio("Gün içinde yorgunluk hissediyor musunuz?", ["Hayır", "Evet"], horizontal=True)

# NEFES DARLIĞI
nefes = st.radio("Nefes darlığı yaşıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
nefes_skor = 1
if nefes == "Evet":
    nefes_detay = st.selectbox("Nefes darlığı ne zaman oluyor?", ["Sadece yürürken", "Merdiven çıkarken", "Dinlenirken bile"])
    nefes_skor = 3 if "Dinlenirken" in nefes_detay else 2

hirilti = st.radio("Boğazınızda hırıltı var mı?", ["Hayır", "Evet"], horizontal=True)

# ÖKSÜRÜK
oksuruk = st.radio("Öksürüğünüz var mı?", ["Hayır", "Evet"], horizontal=True)
oksuruk_skor = 1
if oksuruk == "Evet":
    oksuruk_detay = st.selectbox("Öksürüğünüzün şiddeti nedir?", ["Hafif", "Sık ve balgamlı", "Çok şiddetli"])
    oksuruk_skor = 3 if "şiddetli" in oksuruk_detay.lower() else 2

yutkunma = st.radio("Yutkunmada güçlük çekiyor musunuz?", ["Hayır", "Evet"], horizontal=True)
gogus = st.radio("Göğüs ağrısı çekiyor musunuz?", ["Hayır", "Evet"], horizontal=True)
gogus_skor = 1
if gogus == "Evet":
    gogus_detay = st.selectbox("Göğüs ağrısı şiddeti?", ["Hafif", "Orta", "Şiddetli"])
    gogus_skor = 3 if "Şiddetli" in gogus_detay else 2

# --- 4. ANALİZ BUTONU ---
if st.button("RİSKİ ANALİZ ET"):
    def eh(deger): return 2 if deger == "Evet" else 1
    
    input_data = [
        1 if cinsiyet == "Erkek" else 0, yas,
        sigara_skor, eh(parmak), eh(anksiyete), eh(baski),
        eh(kronik), (2 if yorgunluk == "Evet" else 1), 1, 
        eh(hirilti), alkol_skor, oksuruk_skor, nefes_skor,
        eh(yutkunma), gogus_skor
    ]

    prob = model.predict_proba([input_data])[0][1]
    
    # Gerçekçilik Katsayıları
    extra = 0
    if yas > 70: extra += (yas - 70) * 0.01
    if oksuruk_skor == 3: extra += 0.12
    if nefes_skor == 3: extra += 0.12
    
    final_risk = min((prob + extra) * 100, 99.4)

    st.divider()
    st.write(f"### Tahmini Risk Skoru: %{final_risk:.2f}")
    if final_risk > 65: st.error("YÜKSEK RİSK: Uzman bir doktora görünmeniz önerilir.")
    elif final_risk > 35: st.warning("ORTA RİSK: Yaşam alışkanlıklarınıza dikkat etmelisiniz.")
    else: st.success("DÜŞÜK RİSK: Şu anki verilere göre risk düşük görünmektedir.")
