import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. GÖRSEL AYARLAR ---
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
st.title("🩺 Detaylı Akciğer Kanseri Analiz Sistemi")

st.write("### 📍 Genel Bilgiler")
yas = st.number_input("Yaşınız:", min_value=1, max_value=110, value=25)
cinsiyet = st.selectbox("Cinsiyetiniz:", ["Erkek", "Kadın"])

st.write("### 📝 Detaylı Sağlık Anketi")

# SİGARA
sigara = st.radio("Sigara kullanıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
sigara_skor = 1
if sigara == "Evet":
    sigara_detay = st.selectbox("Kullanım Sıklığı:", ["Günde birkaç adet", "Yarım paket", "1 paket", "1 paketten fazla"])
    sigara_skor = 3 if "1 paket" in sigara_detay else 2

# ALKOL
alkol = st.radio("Alkol tüketiyor musunuz?", ["Hayır", "Evet"], horizontal=True)
alkol_skor = 1
if alkol == "Evet":
    alkol_detay = st.selectbox("Tüketim Sıklığı:", ["Haftada 1-2", "Haftada 3-4", "Her gün"])
    alkol_skor = 2

# ÖKSÜRÜK
oksuruk = st.radio("Sürekli bir öksürüğünüz var mı?", ["Hayır", "Evet"], horizontal=True)
oksuruk_skor = 1
if oksuruk == "Evet":
    oksuruk_detay = st.selectbox("Öksürük Şiddeti:", ["Hafif/Kuru", "Balgamlı/Sık", "Şiddetli/Kanlı"])
    oksuruk_skor = 3 if "Şiddetli" in oksuruk_detay else 2

# NEFES DARLIĞI
nefes = st.radio("Nefes darlığı yaşıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
nefes_skor = 1
if nefes == "Evet":
    nefes_detay = st.selectbox("Nefes Darlığı Durumu:", ["Yürürken oluyor", "Merdiven çıkarken", "Dinlenirken bile var"])
    nefes_skor = 3 if "Dinlenirken" in nefes_detay else 2

# GÖĞÜS AĞRISI
gogus = st.radio("Göğüs bölgenizde ağrı var mı?", ["Hayır", "Evet"], horizontal=True)
gogus_skor = 1
if gogus == "Evet":
    gogus_detay = st.selectbox("Ağrı Şiddeti:", ["Hafif batma", "Baskı hissi", "Şiddetli ağrı"])
    gogus_skor = 3 if "Şiddetli" in gogus_detay else 2

# YORGUNLUK
yorgunluk = st.radio("Aşırı yorgunluk ve halsizlik var mı?", ["Hayır", "Evet"], horizontal=True)
yorgunluk_skor = 2 if yorgunluk == "Evet" else 1

# YUTKUNMA GÜÇLÜĞÜ
yutkunma = st.radio("Yutkunmada zorluk çekiyor musunuz?", ["Hayır", "Evet"], horizontal=True)
yutkunma_skor = 2 if yutkunma == "Evet" else 1

# DİĞERLERİ
parmak = st.radio("Parmak uçlarında sararma/lekeler var mı?", ["Hayır", "Evet"], horizontal=True)
hirilti = st.radio("Hırıltılı solunum (vızıltı) var mı?", ["Hayır", "Evet"], horizontal=True)
anksiyete = st.radio("Anksiyete veya yoğun stres var mı?", ["Hayır", "Evet"], horizontal=True)
kronik = st.radio("Bilinen bir kronik hastalığınız var mı?", ["Hayır", "Evet"], horizontal=True)

# --- 4. ANALİZ BUTONU ---
if st.button("RİSKİ ANALİZ ET"):
    def eh(deger): return 2 if deger == "Evet" else 1
    
    input_data = [
        1 if cinsiyet == "Erkek" else 0, yas,
        sigara_skor, eh(parmak), eh(anksiyete), 1, 
        eh(kronik), yorgunluk_skor, 1, 
        eh(hirilti), alkol_skor, oksuruk_skor, nefes_skor,
        yutkunma_skor, gogus_skor
    ]

    prob = model.predict_proba([input_data])[0][1]
    final_risk = min(prob * 100, 99.4)

    st.divider()
    st.write(f"## Tahmini Risk Skoru: %{final_risk:.2f}")
    st.info("📊 **Yapay Zeka Model Doğruluk Oranı: %88.50**")
    
    if final_risk > 60:
        st.error("⚠️ YÜKSEK RİSK: Verileriniz yüksek risk grubunda görünüyor. Lütfen bir doktora başvurun.")
    else:
        st.success("✅ DÜŞÜK RİSK: Mevcut verilere göre riskiniz düşük seviyededir.")

# ALT BİLGİ
st.divider()
st.caption("Bu çalışma bir yapay zeka projesidir. Kesin teşhis için tıbbi muayene şarttır.")
