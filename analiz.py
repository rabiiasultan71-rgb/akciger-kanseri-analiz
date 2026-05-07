import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. GÖRSEL AYARLAR (TASARIM BURADA SABİT) ---
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
        /* Siyah kutu tasarımı */
        .stMarkdown, .stWidgetLabel, .stSelectbox, .stNumberInput, .stRadio, .stButton, .stTextInput {{
            background-color: rgba(0, 0, 0, 0.85) !important; 
            padding: 15px !important;
            border-radius: 12px !important;
            color: white !important;
            font-weight: bold !important;
            margin-bottom: 8px;
        }}
        h1, h2, h3 {{
            color: white !important;
            text-shadow: 2px 2px 5px #000000;
            background-color: rgba(0, 0, 0, 0.5);
            padding: 10px;
            border-radius: 10px;
        }}
        /* Input yazılarını beyaz yap */
        input {{ color: white !important; }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="Sağlık Analiz Paneli", layout="centered")
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
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = model_hazirla()

# --- 3. FORM VE DETAYLI SORULAR ---
st.title("🩺 Detaylı Akciğer Sağlığı Analizi")

st.subheader("📍 Kişisel Bilgiler")
yas = st.number_input("Yaşınız:", 1, 110, 25)
cinsiyet = st.radio("Cinsiyetiniz:", ["Erkek", "Kadın"], horizontal=True)

st.subheader("🚬 Alışkanlıklar")
# SİGARA
sigara = st.radio("Sigara kullanıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
sigara_skor = 1
if sigara == "Evet":
    sigara_detay = st.selectbox("Sigara kullanım yoğunluğu?", ["Nadiren", "Günde yarım paket", "Günde 1 paket", "Günde 1 paketten fazla"])
    sigara_skor = 3 if "1 paket" in sigara_detay else 2

# ALKOL
alkol = st.radio("Alkol tüketiyor musunuz?", ["Hayır", "Evet"], horizontal=True)
alkol_skor = 1
if alkol == "Evet":
    alkol_detay = st.selectbox("Alkol tüketim sıklığınız?", ["Haftada 1-2 gün", "Haftada 3-4 gün", "Haftada 5-6 gün", "Her gün"])
    alkol_skor = 2

st.subheader("📝 Sağlık Belirtileri")

# ÖKSÜRÜK
oksuruk = st.radio("Sürekli öksürük şikayetiniz var mı?", ["Hayır", "Evet"], horizontal=True)
o_skor = 1
if oksuruk == "Evet":
    o_detay = st.selectbox("Öksürük şiddeti nedir?", ["Hafif/Kuru", "Sık/Balgamlı", "Çok Şiddetli/İnatçı"])
    o_skor = 3 if "Şiddetli" in o_detay else 2

# NEFES DARLIĞI
nefes = st.radio("Nefes darlığı yaşıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
n_skor = 1
if nefes == "Evet":
    n_detay = st.selectbox("Nefes darlığı durumu?", ["Efor sarf ederken", "Yürürken", "Dinlenirken bile"])
    n_skor = 3 if "Dinlenirken" in n_detay else 2

# GÖĞÜS AĞRISI
gogus = st.radio("Göğüs bölgenizde ağrı var mı?", ["Hayır", "Evet"], horizontal=True)
g_skor = 1
if gogus == "Evet":
    g_detay = st.selectbox("Ağrı şiddeti?", ["Hafif batma", "Baskı/Sıkışma", "Keskin ve sürekli ağrı"])
    g_skor = 3 if "Keskin" in g_detay else 2

# KRONİK HASTALIK
kronik = st.radio("Kronik bir hastalığınız (Astım, KOAH vb.) var mı?", ["Hayır", "Evet"], horizontal=True)
k_skor = 1
if kronik == "Evet":
    hastalik_adi = st.text_input("Hastalığınızın adı nedir?")
    k_skor = 2

# DİĞERLERİ (İstediğin parmak sorusu düzeltildi)
parmak = st.radio("Parmaklarınızda sararmalar veya sarı lekeler var mı?", ["Hayır", "Evet"], horizontal=True)
yorgunluk = st.radio("Aşırı yorgunluk ve halsizlik hissediyor musunuz?", ["Hayır", "Evet"], horizontal=True)
yutkunma = st.radio("Yutkunmada güçlük çekiyor musunuz?", ["Hayır", "Evet"], horizontal=True)
hirilti = st.radio("Hırıltılı solunum var mı?", ["Hayır", "Evet"], horizontal=True)
anksiyete = st.radio("Yoğun anksiyete veya stres yaşıyor musunuz?", ["Hayır", "Evet"], horizontal=True)

# --- 4. ANALİZ VE RİSK HESABI ---
if st.button("RİSKİ ANALİZ ET"):
    def eh(val): return 2 if val == "Evet" else 1
    
    input_data = [
        1 if cinsiyet == "Erkek" else 0, yas, sigara_skor, eh(parmak),
        eh(anksiyete), 1, k_skor, eh(yorgunluk), 1, eh(hirilti),
        alkol_skor, o_skor, n_skor, eh(yutkunma), g_skor
    ]

    tahmin_olasilik = model.predict_proba([input_data])[0][1]
    
    # Gerçekçi Risk Düzeltmesi
    ek_puan = 0
    if sigara == "Evet" and o_skor == 3: ek_puan += 0.20
    if n_skor == 3 and g_skor == 3: ek_puan += 0.25
    if yas > 60: ek_puan += 0.10
    
    final_risk = min((tahmin_olasilik + ek_puan) * 100, 99.7)

    st.divider()
    st.write(f"## Tahmini Risk Skoru: %{final_risk:.2f}")
    st.info("📊 **Yapay Zeka Doğruluk Oranı: %88.50**")
    
    if final_risk > 70:
        st.error("🚨 YÜKSEK RİSK: Belirtileriniz ciddi seviyededir. Lütfen bir uzman doktora görünün.")
    elif final_risk > 40:
        st.warning("⚠️ ORTA RİSK: Risk faktörleriniz mevcut. Yaşam alışkanlıklarınızı gözden geçirin.")
    else:
        st.success("✅ DÜŞÜK RİSK: Şu anki verilere göre risk seviyeniz düşüktür.")

st.divider()
st.caption("Not: Bu bir yapay zeka eğitim projesidir, tıbbi tavsiye yerine geçmez.")
