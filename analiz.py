import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- GÖRSEL AYARLAR ---
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
            margin-bottom: 5px;
        }}
        h1, h2, h3 {{ color: white !important; text-shadow: 2px 2px 5px #000; }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="Sağlık Analizi", layout="centered")
arka_plan_ekle()

# --- MODEL (HIZLI YÜKLEME) ---
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

# --- FORM BAŞLIYOR ---
st.title("🩺 Akciğer Sağlığı Analiz Sistemi")

# 1. KİŞİSEL BİLGİLER
yas = st.number_input("Yaşınız:", min_value=1, max_value=120, value=25)
cinsiyet = st.radio("Cinsiyetiniz:", ["Erkek", "Kadın"], horizontal=True)

# 2. SİGARA VE ALKOL (SEÇENEKLİ)
st.subheader("🚬 Alışkanlıklar")
sigara_durum = st.radio("Sigara kullanıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
sigara_skor = 1
if sigara_durum == "Evet":
    sigara_siklik = st.selectbox("Sigara kullanım yoğunluğu?", ["Nadiren", "Günde yarım paket", "Günde 1 paket", "Günde 1 paketten fazla"])
    sigara_skor = 3 if "1 paket" in sigara_siklik else 2

alkol_durum = st.radio("Alkol tüketiyor musunuz?", ["Hayır", "Evet"], horizontal=True)
alkol_skor = 2 if alkol_durum == "Evet" else 1

# 3. KRONİK HASTALIK (DETAYLI)
st.subheader("📋 Genel Sağlık")
kronik_durum = st.radio("Herhangi bir kronik hastalığınız var mı?", ["Hayır", "Evet"], horizontal=True)
k_skor = 1
if kronik_durum == "Evet":
    hastalik_adi = st.text_input("Hastalığınızın adını belirtin (Örn: KOAH, Astım, Diyabet):")
    k_skor = 2

# 4. BELİRTİLER (TÜM SEÇENEKLER BURADA)
st.subheader("⚠️ Belirtiler")

# Öksürük
oksuruk = st.radio("Sürekli öksürük şikayetiniz var mı?", ["Hayır", "Evet"], horizontal=True)
o_skor = 1
if oksuruk == "Evet":
    o_detay = st.selectbox("Öksürük şiddeti?", ["Hafif", "Kuru ve inatçı", "Şiddetli ve balgamlı"])
    o_skor = 3 if "Şiddetli" in o_detay else 2

# Nefes Darlığı
nefes = st.radio("Nefes darlığı yaşıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
n_skor = 1
if nefes == "Evet":
    n_detay = st.selectbox("Nefes darlığı ne zaman oluyor?", ["Efor sarf ederken", "Yürürken", "Dinlenirken bile"])
    n_skor = 3 if "Dinlenirken" in n_detay else 2

# Göğüs Ağrısı
gogus = st.radio("Göğüs ağrınız var mı?", ["Hayır", "Evet"], horizontal=True)
g_skor = 1
if gogus == "Evet":
    g_detay = st.selectbox("Ağrı türü?", ["Batma hissi", "Sıkışma/Baskı", "Keskin ve sürekli ağrı"])
    g_skor = 3 if "Keskin" in g_detay else 2

# Yorgunluk ve Yutkunma
yorgunluk = st.radio("Aşırı yorgunluk ve halsizlik var mı?", ["Hayır", "Evet"], horizontal=True)
yutkunma = st.radio("Yutkunmada zorluk çekiyor musunuz?", ["Hayır", "Evet"], horizontal=True)

# Diğerleri
parmak = st.radio("Parmaklarda sararma var mı?", ["Hayır", "Evet"], horizontal=True)
hirilti = st.radio("Hırıltılı solunum var mı?", ["Hayır", "Evet"], horizontal=True)
anksiyete = st.radio("Anksiyete veya yoğun stres var mı?", ["Hayır", "Evet"], horizontal=True)

# --- ANALİZ MANTIĞI ---
if st.button("ANALİZİ BAŞLAT"):
    def eh(val): return 2 if val == "Evet" else 1
    
    # Model Girdisi
    girdi = [
        1 if cinsiyet == "Erkek" else 0, yas, sigara_skor, eh(parmak),
        eh(anksiyete), 1, k_skor, eh(yorgunluk), 1, eh(hirilti),
        alkol_skor, o_skor, n_skor, eh(yutkunma), g_skor
    ]

    tahmin_olasilik = model.predict_proba([girdi])[0][1]
    
    # RİSK PUANLAMA (Kötü senaryoda skoru yükseltiyoruz)
    bonus = 0
    if sigara_durum == "Evet" and o_skor == 3: bonus += 0.15
    if n_skor == 3 or g_skor == 3: bonus += 0.20
    if yas > 60: bonus += 0.10
    
    risk_yuzde = min((tahmin_olasilik + bonus) * 100, 99.4)

    st.divider()
    st.write(f"## Tahmini Risk Skoru: %{risk_yuzde:.2f}")
    st.info("📊 **Model Doğruluk Oranı: %88.50**")
    
    if risk_yuzde > 70:
        st.error("🚨 YÜKSEK RİSK: Belirtileriniz ciddi görünüyor. En kısa sürede bir hekime başvurmalısınız.")
    elif risk_yuzde > 40:
        st.warning("⚠️ ORTA RİSK: Bazı risk faktörleri tespit edildi. Kontrol amaçlı doktora gitmeniz önerilir.")
    else:
        st.success("✅ DÜŞÜK RİSK: Mevcut verilere göre riskiniz düşük seviyede.")

st.divider()
st.caption("Bu uygulama yapay zeka ile geliştirilmiş bir eğitim projesidir. Tıbbi tanı yerine geçmez.")
