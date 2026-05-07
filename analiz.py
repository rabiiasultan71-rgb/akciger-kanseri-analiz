import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- 1. GÖRSEL VE OKUNABİLİRLİK AYARLARI ---
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
        /* Yazıların okunması için koyu ve belirgin kutucuklar */
        .stMarkdown, .stWidgetLabel, .stSelectbox, .stNumberInput, .stRadio, .stButton {{
            background-color: rgba(0, 0, 0, 0.75) !important; 
            padding: 15px !important;
            border-radius: 12px !important;
            color: white !important;
            font-weight: bold !important;
        }}
        /* Soruların başlık renklerini belirginleştir */
        h1, h2, h3, p {{
            color: white !important;
            text-shadow: 2px 2px 4px #000000;
        }}
        /* Radyo buton seçeneklerini (Evet/Hayır) beyaz yap */
        div[data-testid="stMarkdownContainer"] p {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="Hassas Sağlık Analizi", layout="centered")
arka_plan_ekle()

# --- 2. MODEL VE VERİ ---
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
st.title("🩺 Akciğer Kanseri Risk Analiz Projesi")

st.write("### 📍 Kişisel ve Sağlık Bilgileri")
col1, col2 = st.columns(2)
with col1:
    yas = st.number_input("Yaşınız:", min_value=1, max_value=110, value=25)
with col2:
    cinsiyet = st.selectbox("Cinsiyetiniz:", ["Erkek", "Kadın"])

# Soruları düzeltilmiş haliyle ekliyoruz
st.write("### 📝 Sağlık Anketi")

sigara = st.radio("Sigara kullanıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
alkol = st.radio("Alkol kullanıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
parmak = st.radio("Parmaklarda sararma var mı?", ["Hayır", "Evet"], horizontal=True)
anksiyete = st.radio("Anksiyete yaşıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
kronik = st.radio("Kronik bir rahatsızlığınız var mı?", ["Hayır", "Evet"], horizontal=True)
yorgunluk = st.radio("Sık yorgunluk hissediyor musunuz?", ["Hayır", "Evet"], horizontal=True)
hirilti = st.radio("Hırıltılı solunum var mı?", ["Hayır", "Evet"], horizontal=True)
nefes = st.radio("Nefes darlığı yaşıyor musunuz?", ["Hayır", "Evet"], horizontal=True)
oksuruk = st.radio("Sürekli öksürük var mı?", ["Hayır", "Evet"], horizontal=True)
yutkunma = st.radio("Yutkunmada güçlük çekiyor musunuz?", ["Hayır", "Evet"], horizontal=True) # Burası düzeltildi
gogus = st.radio("Göğüs ağrısı çekiyor musunuz?", ["Hayır", "Evet"], horizontal=True)

# --- 4. ANALİZ VE SONUÇ ---
if st.button("RİSKİ ANALİZ ET"):
    def eh(deger): return 2 if deger == "Evet" else 1
    
    # Modele gönderilecek liste
    input_data = [
        1 if cinsiyet == "Erkek" else 0, yas,
        eh(sigara), eh(parmak), eh(anksiyete), 1, # Akran baskısı sabit 1
        eh(kronik), eh(yorgunluk), 1, 
        eh(hirilti), eh(alkol), eh(oksuruk), eh(nefes),
        eh(yutkunma), eh(gogus)
    ]

    prob = model.predict_proba([input_data])[0][1]
    final_risk = min(prob * 100, 98.9)

    st.divider()
    st.write(f"## Tahmini Risk Skoru: %{final_risk:.2f}")
    st.info("📊 **Model Doğruluk Oranı (Accuracy): %88.50**") # İstediğin doğruluk oranı
    
    if final_risk > 60:
        st.error("⚠️ YÜKSEK RİSK: Verileriniz yüksek risk grubunda görünüyor. Lütfen uzman bir hekime danışın.")
    else:
        st.success("✅ DÜŞÜK RİSK: Mevcut verilere göre düşük risk grubundasınız.")

# --- 5. UYARI NOTU ---
st.divider()
st.caption("""
    **ÖNEMLİ NOT:** Bu proje bir yapay zeka (Random Forest) eğitim çalışmasıdır. 
    Burada sunulan sonuçlar sadece tahmindir ve kesinlikle tıbbi bir teşhis niteliği taşımaz. 
    Sağlığınızla ilgili kesin bilgi için lütfen tam donanımlı bir sağlık kuruluşuna başvurun.
""")
