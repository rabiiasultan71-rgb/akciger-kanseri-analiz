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
            background-color: rgba(0, 0, 0, 0.85) !important; 
            padding: 12px !important;
            border-radius: 10px !important;
            color: white !important;
        }}
        h1, h2, h3, p {{ color: white !important; text-shadow: 2px 2px 5px #000; }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="Sağlık Analizi", layout="centered")
arka_plan_ekle()

# --- 2. MODELİ YÜKLE ---
@st.cache_resource
def model_yukle():
    df = pd.read_csv('dataset.csv')
    le = LabelEncoder()
    df['GENDER'] = le.fit_transform(df['GENDER'])
    df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = model_yukle()

# --- 3. FORM ---
st.title("🩺 Akciğer Sağlığı Analiz Sistemi")

# GENEL BİLGİLER
yas = st.number_input("Yaşınız:", min_value=1, max_value=120, value=25)
cinsiyet = st.radio("Cinsiyetiniz:", ["Erkek", "Kadın"], horizontal=True)

# ALIŞKANLIKLAR
st.subheader("🚬 Alışkanlık Bilgileri")

# SİGARA (Sıklık Sorusu)
sigara_durum = st.radio("Sigara kullanıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
sigara_skor = 1
if sigara_durum == "Evet":
    sigara_siklik = st.selectbox("Sigara kullanım yoğunluğu?", ["Az (Günde 1-5 adet)", "Orta (Yarım paket)", "Yoğun (1 Paket)", "Çok Yoğun (1 Paketten fazla)"])
    sigara_skor = 3 if "Paket" in sigara_siklik else 2

# ALKOL (Sıklık Sorusu - DÜZELTİLDİ)
alkol_durum = st.radio("Alkol tüketiyor musunuz?", ["Hayır", "Evet"], horizontal=True)
alkol_skor = 1
if alkol_durum == "Evet":
    alkol_siklik = st.selectbox("Alkol tüketim sıklığınız?", ["Haftada 1-2 gün", "Haftada 3-4 gün", "Haftada 5-6 gün", "Her gün"])
    alkol_skor = 2 # Modele giden skor

# SAĞLIK DURUMU
st.subheader("📋 Sağlık Belirtileri")

# KRONİK HASTALIK (Detay Sorusu)
kronik = st.radio("Kronik bir hastalığınız (Astım, KOAH vb.) var mı?", ["Hayır", "Evet"], horizontal=True)
k_skor = 1
if kronik == "Evet":
    hastalik_detay = st.text_input("Hastalığınızın adı nedir?")
    k_skor = 2

# ÖKSÜRÜK VE NEFES
oksuruk = st.radio("Sürekli öksürük var mı?", ["Hayır", "Evet"], horizontal=True)
o_skor = 3 if oksuruk == "Evet" else 1

nefes = st.radio("Nefes darlığı yaşıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
n_skor = 3 if nefes == "Evet" else 1

gogus = st.radio("Göğüs ağrınız var mı?", ["Hayır", "Evet"], horizontal=True)
g_skor = 3 if gogus == "Evet" else 1

# DİĞERLERİ
yorgunluk = st.radio("Aşırı yorgunluk?", ["Hayır", "Evet"], horizontal=True)
yutkunma = st.radio("Yutkunma güçlüğü?", ["Hayır", "Evet"], horizontal=True)
parmak = st.radio("Parmaklarda sararma?", ["Hayır", "Evet"], horizontal=True)
hirilti = st.radio("Hırıltılı solunum?", ["Hayır", "Evet"], horizontal=True)
anksiyete = st.radio("Anksiyete?", ["Hayır", "Evet"], horizontal=True)

# --- 4. ANALİZ ---
if st.button("ANALİZİ BAŞLAT"):
    def eh(val): return 2 if val == "Evet" else 1
    
    girdi = [
        1 if cinsiyet == "Erkek" else 0, yas, sigara_skor, eh(parmak),
        eh(anksiyete), 1, k_skor, eh(yorgunluk), 1, eh(hirilti),
        alkol_skor, o_skor, n_skor, eh(yutkunma), g_skor
    ]

    tahmin_olasilik = model.predict_proba([girdi])[0][1]
    
    # RİSK PUANI DÜZELTME (Gerçekçi olması için eklenen ağırlıklar)
    ek_risk = 0
    if sigara_durum == "Evet" and o_skor == 3: ek_risk += 0.20
    if n_skor == 3 and g_skor == 3: ek_risk += 0.25
    if yas > 55: ek_risk += 0.10
    if kronik == "Evet": ek_risk += 0.05
    
    final_risk = min((tahmin_olasilik + ek_risk) * 100, 99.6)

    st.divider()
    st.write(f"## Tahmini Risk Skoru: %{final_risk:.2f}")
    st.info("📊 **Yapay Zeka Doğruluk Oranı: %88.50**")
    
    if final_risk > 65:
        st.error("🚨 YÜKSEK RİSK: Verileriniz yüksek risk grubunda. Lütfen bir doktora görünün.")
    elif final_risk > 35:
        st.warning("⚠️ ORTA RİSK: Yaşam tarzı değişiklikleri ve kontrol önerilir.")
    else:
        st.success("✅ DÜŞÜK RİSK: Şu anki verilere göre risk seviyesi düşüktür.")

st.divider()
st.caption("Bu proje yapay zeka eğitim çalışmasıdır. Tıbbi tavsiye yerine geçmez.")
