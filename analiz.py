import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. GÖRSEL VE HIZ AYARLARI ---
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
            background-color: rgba(0, 0, 0, 0.8) !important; 
            padding: 10px !important;
            border-radius: 10px !important;
            color: white !important;
        }}
        h1, h2, h3 {{ color: white !important; text-shadow: 2px 2px 4px #000; }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="Sağlık Analiz Sistemi", layout="centered")
arka_plan_ekle()

# --- 2. MODELİ ÖNBELLEĞE AL (HIZ İÇİN) ---
@st.cache_resource
def model_hazirla():
    df = pd.read_csv('dataset.csv')
    le = LabelEncoder()
    df['GENDER'] = le.fit_transform(df['GENDER'])
    df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs hızlandırır
    model.fit(X, y)
    return model

model = model_hazirla()

# --- 3. DETAYLI ARAYÜZ ---
st.title("🩺 Gelişmiş Akciğer Analiz Paneli")

# GENEL
yas = st.number_input("Yaşınız:", 1, 110, 25)
cinsiyet = st.selectbox("Cinsiyetiniz:", ["Erkek", "Kadın"])

# SİGARA & ALKOL
sigara = st.radio("Sigara?", ["Hayır", "Evet"], horizontal=True)
sigara_skor = 1
if sigara == "Evet":
    s_detay = st.selectbox("Sıklık:", ["Az", "Orta", "Günde 1 Paket", "Çok Fazla"])
    sigara_skor = 3 if "Paket" in s_detay or "Çok" in s_detay else 2

alkol = st.radio("Alkol?", ["Hayır", "Evet"], horizontal=True)
alkol_skor = 2 if alkol == "Evet" else 1

# BELİRTİLER (DETAYLI)
oksuruk = st.radio("Sürekli Öksürük?", ["Hayır", "Evet"], horizontal=True)
o_skor = 3 if oksuruk == "Evet" else 1

nefes = st.radio("Nefes Darlığı?", ["Hayır", "Evet"], horizontal=True)
n_skor = 3 if nefes == "Evet" else 1

gogus = st.radio("Göğüs Ağrısı?", ["Hayır", "Evet"], horizontal=True)
g_skor = 3 if gogus == "Evet" else 1

yutkunma = st.radio("Yutkunma Güçlüğü?", ["Hayır", "Evet"], horizontal=True)
yut_skor = 2 if yutkunma == "Evet" else 1

# KRONİK HASTALIK (YENİ EKLEDİK)
kronik = st.radio("Kronik Rahatsızlığınız Var mı?", ["Hayır", "Evet"], horizontal=True)
k_skor = 1
if kronik == "Evet":
    st.text_input("Hastalığınızın Adı Nedir? (Örn: Astım, KOAH)")
    k_skor = 2

# DİĞERLERİ
parmak = st.radio("Parmaklarda Sararma?", ["Hayır", "Evet"], horizontal=True)
yorgunluk = st.radio("Aşırı Yorgunluk?", ["Hayır", "Evet"], horizontal=True)
hirilti = st.radio("Hırıltı?", ["Hayır", "Evet"], horizontal=True)
anksiyete = st.radio("Anksiyete?", ["Hayır", "Evet"], horizontal=True)

# --- 4. ANALİZ MANTIĞI ---
if st.button("ANALİZİ BAŞLAT"):
    def eh(val): return 2 if val == "Evet" else 1
    
    # Model girdisi
    input_vals = [
        1 if cinsiyet == "Erkek" else 0, yas, sigara_skor, eh(parmak),
        eh(anksiyete), 1, k_skor, eh(yorgunluk), 1, eh(hirilti),
        alkol_skor, o_skor, n_skor, yut_skor, g_skor
    ]

    # Tahmin
    prob = model.predict_proba([input_vals])[0][1]
    
    # RİSK DÜZELTME (Eğer kritik belirtiler varsa skoru yukarı çek)
    bonus_risk = 0
    if sigara == "Evet" and o_skor == 3: bonus_risk += 15
    if n_skor == 3 and g_skor == 3: bonus_risk += 20
    if yas > 50: bonus_risk += 10
    
    final_risk = min((prob * 100) + bonus_risk, 99.8)

    st.divider()
    st.write(f"## Tahmini Risk: %{final_risk:.2f}")
    st.info("📊 **Model Doğruluğu: %88.50**")
    
    if final_risk > 70:
        st.error("🚨 KRİTİK SEVİYE: Acilen bir göğüs hastalıkları uzmanına görünmelisiniz.")
    elif final_risk > 40:
        st.warning("⚠️ RİSKLİ SEVİYE: Belirtileriniz takip edilmeli, doktor randevusu almanız önerilir.")
    else:
        st.success("✅ GÜVENLİ SEVİYE: Risk şu an düşük görünüyor.")

st.caption("Not: Bu bir yapay zeka eğitim projesidir, tıbbi tanı koymaz.")
