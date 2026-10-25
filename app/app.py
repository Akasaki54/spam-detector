import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
import re
from datetime import datetime
from pathlib import Path

# Resolve paths relative to this file
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA  = DATA_DIR / "raw_data.csv"
TEST_DATA = DATA_DIR / "test_cases.csv"

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="SpamGuard · Detektor Spam",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* ── Background ── */
.stApp { background: #0f1117; }

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1100px; }

/* ── Header ── */
.sg-header {
    text-align: center;
    padding: 3rem 0 2rem;
}
.sg-logo {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.sg-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
}
.sg-subtitle {
    color: #64748b;
    font-size: 1rem;
    margin-top: 0.4rem;
}

/* ── Card ── */
.sg-card {
    background: #1e2130;
    border: 1px solid #2d3147;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.25rem;
}

/* ── Result badges ── */
.result-spam {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-ham {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
}
.result-label {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 0;
}
.result-sub {
    font-size: 0.9rem;
    opacity: 0.8;
    margin-top: 0.3rem;
}

/* ── Probability bar ── */
.prob-row { display: flex; align-items: center; gap: 0.75rem; margin: 0.4rem 0; }
.prob-label { color: #94a3b8; font-size: 0.85rem; width: 130px; flex-shrink: 0; }
.prob-bar-bg { flex: 1; background: #2d3147; border-radius: 100px; height: 8px; }
.prob-bar-fill { height: 8px; border-radius: 100px; }
.prob-value { color: #e2e8f0; font-size: 0.85rem; font-weight: 600; width: 48px; text-align: right; flex-shrink: 0; }

/* ── Chip ── */
.chip {
    display: inline-block;
    background: #2d3147;
    border: 1px solid #3d4168;
    color: #a5b4fc;
    font-size: 0.78rem;
    padding: 0.2rem 0.65rem;
    border-radius: 100px;
    margin: 0.15rem;
}

/* ── Metric card ── */
.metric-mini {
    background: #1e2130;
    border: 1px solid #2d3147;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-mini .val { font-size: 1.5rem; font-weight: 700; color: #a78bfa; }
.metric-mini .lbl { font-size: 0.78rem; color: #64748b; margin-top: 0.1rem; }

/* ── Table overrides ── */
.dataframe { border: none !important; }
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid #2d3147;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.25rem;
    font-weight: 500;
    font-size: 0.9rem;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: #1e2130 !important;
    color: #a78bfa !important;
    border-bottom: 2px solid #a78bfa;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    font-size: 0.9rem;
    transition: opacity 0.2s, transform 0.2s;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* ── Text area ── */
.stTextArea textarea {
    background: #151722 !important;
    border: 1px solid #2d3147 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
    resize: vertical;
}
.stTextArea textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

/* ── Radio ── */
.stRadio [data-baseweb="radio"] { color: #94a3b8 !important; }

/* ── Info / warning ── */
.sg-info {
    background: #172033;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    color: #93c5fd;
    font-size: 0.88rem;
    margin: 0.75rem 0;
}
.sg-warn {
    background: #1f1510;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    color: #fcd34d;
    font-size: 0.88rem;
    margin: 0.75rem 0;
}

/* ── Section title ── */
.sg-section {
    color: #cbd5e1;
    font-size: 1.05rem;
    font-weight: 600;
    margin: 1.5rem 0 0.75rem;
}
.sg-section span { color: #a78bfa; }

/* ── Divider ── */
.sg-divider { border: none; border-top: 1px solid #2d3147; margin: 1.5rem 0; }

/* ── Matplotlib background ── */
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  MODEL TRAINING (cached)
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Melatih model…")
def load_and_train():
    try:
        df = pd.read_csv(RAW_DATA)
        df = df[['pesan', 'label']].dropna()
    except Exception:
        # Fallback inline dataset
        pesan = [
            "Menang hadiah 10 juta! Klik link ini sekarang juga gratis!",
            "Halo bro, malam ini kita makan di kantornya Andi ya.",
            "Verifikasi rekening Anda melalui tautan berikut sekarang juga.",
            "Meeting jam 3 siang di ruang rapat ya, jangan telat.",
            "PROMO TERBATAS! Beli 1 Gratis 2. Hubungi 08123456789",
            "Pak, draf laporan yang kemarin sudah saya email, tolong dicek.",
            "SELAMAT! Anda terpilih mendapatkan iPhone 15. Klaim sekarang!",
            "Bu, dokumen yang ibu minta sudah saya kirim ke email.",
            "Dapatkan uang cepat! Transfer sekarang juga ke rekening ini.",
            "Jangan lupa besok ada presentasi proyek di ruang direksi.",
            "KLIK DI SINI untuk hadiah spesial hanya hari ini!!!",
            "Pak direktur, laporan keuangan Q3 sudah saya siapkan.",
            "Promo gila-gilaan! Diskon 90% khusus member baru!",
            "Rek, tugas kelompok deadline besok ya, jangan lupa kumpul.",
            "Anda mendapat undian 500 juta! Balas SMS ini sekarang!",
            "Halo, bisakah kita reschedule rapat besok ke hari Kamis?",
            "FREE IPHONE! Klik tautan ini untuk klaim hadiah Anda sekarang!",
            "Sudah lihat jadwal UAS terbaru? Ada perubahan dari kemarin.",
            "PINJAMAN CEPAT tanpa agunan cair 1 jam! Bunga 0%! Daftar!",
            "Terima kasih sudah datang ke acara kami, materi sudah dikirim.",
            "Investasi crypto untung 500% per bulan, gabung grup kami!",
            "Bu, saya izin tidak masuk besok karena demam, sudah ke dokter.",
            "DARURAT! Akun bank Anda diblokir, verifikasi: http://palsu.com",
            "Kak, bisa minta notulen rapat minggu lalu? Saya tidak hadir.",
            "Daftar member VIP! Bayar Rp 250rb, hasilkan Rp 2juta/hari!",
            "Selamat ulang tahun! Semoga panjang umur dan selalu sehat.",
            "GIVEAWAY! Kirim data KTP Anda ke DM kami untuk ikut undian.",
            "Nanti kumpul di kosan Dito jam 7 buat ngerjain tugas bareng.",
            "Transfer Rp 500rb ke rekening ini atau data Anda kami sebar!",
            "Konfirmasi reservasi meja 4 orang untuk malam ini jam 19.00.",
        ] * 10
        label = [
            "spam","ham","spam","ham","spam","ham","spam","ham","spam","ham",
            "spam","ham","spam","ham","spam","ham","spam","ham","spam","ham",
            "spam","ham","spam","ham","spam","ham","spam","ham","spam","ham",
        ] * 10
        df = pd.DataFrame({"pesan": pesan, "label": label})

    df['target'] = df['label'].map({'spam': 1, 'ham': 0})
    df = df.dropna(subset=['target'])

    X_train, X_test, y_train, y_test = train_test_split(
        df['pesan'], df['target'].astype(int),
        test_size=0.25, random_state=42, stratify=df['target']
    )

    # Word model
    vec_w = CountVectorizer(min_df=1)
    nb_w = MultinomialNB(alpha=0.5)
    nb_w.fit(vec_w.fit_transform(X_train), y_train)

    # Char model
    vec_c = CountVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1)
    nb_c = MultinomialNB(alpha=0.5)
    nb_c.fit(vec_c.fit_transform(X_train), y_train)

    pred_w = nb_w.predict(vec_w.transform(X_test))
    pred_c = nb_c.predict(vec_c.transform(X_test))

    return {
        'nb_w': nb_w, 'vec_w': vec_w,
        'nb_c': nb_c, 'vec_c': vec_c,
        'X_test': X_test, 'y_test': y_test,
        'pred_w': pred_w, 'pred_c': pred_c,
        'n_train': len(X_train), 'n_test': len(X_test),
        'vocab_size': len(vec_w.vocabulary_),
    }


@st.cache_data
def load_test_cases():
    try:
        return pd.read_csv(TEST_DATA)
    except Exception:
        return pd.DataFrame()


models = load_and_train()
test_df = load_test_cases()

# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'pending_input' not in st.session_state:
    st.session_state.pending_input = ""


# ─────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="sg-header">
  <div class="sg-logo">🛡️</div>
  <h1 class="sg-title">SpamGuard</h1>
  <p class="sg-subtitle">Deteksi pesan spam dengan Naive Bayes · Word-level &amp; Character n-gram</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  Periksa Pesan",
    "📊  Performa Model",
    "🧪  Test Cases",
    "ℹ️  Tentang",
])


# ══════════════════════════════════════════
#  TAB 1 · CHECK MESSAGE
# ══════════════════════════════════════════
with tab1:
    col_inp, col_res = st.columns([1, 1], gap="large")

    with col_inp:
        st.markdown('<p class="sg-section"><span>①</span> Masukkan Pesan</p>', unsafe_allow_html=True)

        # Read pending value from sample buttons BEFORE creating the widget
        _prefill = st.session_state.pop("pending_input", "")

        pesan_input = st.text_area(
            label="pesan",
            label_visibility="collapsed",
            height=160,
            placeholder="Ketik atau tempel pesan yang ingin diperiksa…",
            value=_prefill,
        )

        c1, c2 = st.columns([3, 1])
        with c1:
            run_btn = st.button("🔎  Analisis Sekarang", use_container_width=True)
        with c2:
            if st.button("↺ Reset", use_container_width=True):
                st.session_state.pending_input = ""
                st.rerun()

        # Quick test samples — write to pending_input (not the widget key)
        st.markdown('<p class="sg-section"><span>②</span> Atau coba contoh</p>', unsafe_allow_html=True)
        samples = {
            "💸 Penipuan Hadiah": "SELAMAT! Anda memenangkan hadiah Rp 50 juta. Klik tautan ini untuk klaim sekarang!",
            "🎣 Phishing Bank": "Akun bank Anda diblokir! Verifikasi sekarang: http://bca-palsu.com/login",
            "💬 Pesan Normal": "Halo, meeting besok jam 9 pagi di ruang rapat lantai 3, jangan lupa bawa laptop.",
            "📚 Akademik": "Kak, bisa bantu jelaskan materi pertemuan kemarin? Saya tidak bisa hadir.",
        }
        for label_s, text_s in samples.items():
            if st.button(label_s, key=f"sample_{label_s}", use_container_width=True):
                st.session_state.pending_input = text_s
                st.rerun()

    with col_res:
        st.markdown('<p class="sg-section"><span>③</span> Hasil Analisis</p>', unsafe_allow_html=True)

        txt = pesan_input

        if txt.strip():
            prob_w = models['nb_w'].predict_proba(models['vec_w'].transform([txt]))[0]
            prob_c = models['nb_c'].predict_proba(models['vec_c'].transform([txt]))[0]

            spam_score = (prob_w[1] + prob_c[1]) / 2
            is_spam = spam_score >= 0.5
            confidence = spam_score * 100 if is_spam else (1 - spam_score) * 100

            if is_spam:
                st.markdown(f"""
                <div class="result-spam">
                  <p class="result-label" style="color:#fca5a5">🚫 SPAM TERDETEKSI</p>
                  <p class="result-sub" style="color:#fca5a5">Keyakinan: {confidence:.0f}%</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-ham">
                  <p class="result-label" style="color:#6ee7b7">✅ PESAN AMAN</p>
                  <p class="result-sub" style="color:#6ee7b7">Keyakinan: {confidence:.0f}%</p>
                </div>""", unsafe_allow_html=True)

            st.markdown("<hr class='sg-divider'>", unsafe_allow_html=True)

            # Probability bars
            def prob_bar(label, spam_pct, color):
                return f"""
                <div class="prob-row">
                  <span class="prob-label">{label}</span>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{spam_pct:.0f}%;background:{color}"></div>
                  </div>
                  <span class="prob-value">{spam_pct:.0f}%</span>
                </div>"""

            color_w = "#ef4444" if prob_w[1] >= 0.5 else "#10b981"
            color_c = "#ef4444" if prob_c[1] >= 0.5 else "#10b981"

            st.markdown(
                f"<p style='color:#64748b;font-size:.82rem;margin-bottom:.4rem'>Probabilitas Spam per Model</p>"
                + prob_bar("Model Kata", prob_w[1]*100, color_w)
                + prob_bar("Model Karakter", prob_c[1]*100, color_c),
                unsafe_allow_html=True
            )

            # Trigger words
            words = re.findall(r'\w+', txt.lower())
            triggers = [w for w in words if w in models['vec_w'].vocabulary_][:8]
            if triggers:
                chips = "".join(f'<span class="chip">{w}</span>' for w in triggers)
                st.markdown(
                    f"<p style='color:#64748b;font-size:.82rem;margin:.75rem 0 .3rem'>Kata yang dikenali model</p>{chips}",
                    unsafe_allow_html=True
                )

            if is_spam:
                st.markdown("""<div class="sg-warn">
                    ⚠️ <b>Saran:</b> Abaikan pesan ini. Jangan klik tautan, balas, atau berikan data pribadi.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="sg-info">
                    ℹ️ Pesan tampak aman. Tetap waspada terhadap konten yang mencurigakan.
                </div>""", unsafe_allow_html=True)

            # Save to history
            st.session_state.history.append({
                'Waktu': datetime.now().strftime("%H:%M:%S"),
                'Pesan': txt[:80] + ('…' if len(txt) > 80 else ''),
                'Hasil': '🚫 SPAM' if is_spam else '✅ AMAN',
                'Keyakinan': f"{confidence:.0f}%",
            })

        elif not run_btn:
            st.markdown("""
            <div style="border:1px dashed #2d3147;border-radius:14px;padding:3rem 2rem;text-align:center;color:#334155">
              <p style="font-size:2rem;margin:0">🔍</p>
              <p style="margin:.5rem 0 0">Hasil analisis akan muncul di sini</p>
            </div>""", unsafe_allow_html=True)

    # History strip
    if st.session_state.history:
        st.markdown("<hr class='sg-divider'>", unsafe_allow_html=True)

        h_col1, h_col2 = st.columns([3, 1])
        with h_col1:
            st.markdown('<p class="sg-section"><span>④</span> Riwayat Sesi Ini</p>', unsafe_allow_html=True)
        with h_col2:
            if st.button("🗑️ Hapus Riwayat", use_container_width=True, key="clear_hist"):
                st.session_state.history = []
                st.rerun()

        df_hist = pd.DataFrame(st.session_state.history[::-1])
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        # CSV download
        import io as _io
        _csv_buf = _io.StringIO()
        df_hist.to_csv(_csv_buf, index=False)
        st.download_button(
            label="⬇️ Unduh Riwayat sebagai CSV",
            data=_csv_buf.getvalue().encode("utf-8"),
            file_name=f"riwayat_spam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_hist_csv",
        )


# ══════════════════════════════════════════
#  TAB 2 · MODEL PERFORMANCE
# ══════════════════════════════════════════
with tab2:
    # Top metrics
    rw = classification_report(models['y_test'], models['pred_w'],
                               target_names=['Normal', 'Spam'], output_dict=True, zero_division=0)
    rc = classification_report(models['y_test'], models['pred_c'],
                               target_names=['Normal', 'Spam'], output_dict=True, zero_division=0)

    st.markdown('<p class="sg-section"><span>①</span> Ringkasan Performa</p>', unsafe_allow_html=True)

    cols = st.columns(5)
    stats = [
        ("Data Latih", f"{models['n_train']:,}", "#a78bfa"),
        ("Data Uji", f"{models['n_test']:,}", "#60a5fa"),
        ("Vocab Size", f"{models['vocab_size']:,}", "#34d399"),
        ("F1 (Kata)", f"{rw['Spam']['f1-score']*100:.1f}%", "#fb923c"),
        ("F1 (Karakter)", f"{rc['Spam']['f1-score']*100:.1f}%", "#f472b6"),
    ]
    for col, (lbl, val, color) in zip(cols, stats):
        col.markdown(f"""
        <div class="metric-mini">
          <div class="val" style="color:{color}">{val}</div>
          <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='sg-divider'>", unsafe_allow_html=True)

    # Confusion matrices side by side
    st.markdown('<p class="sg-section"><span>②</span> Confusion Matrix</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    def draw_cm(ax, cm, title):
        ax.set_facecolor('#1e2130')
        ax.figure.set_facecolor('#1e2130')
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Spam'], color='#94a3b8', fontsize=10)
        ax.set_yticklabels(['Normal', 'Spam'], color='#94a3b8', fontsize=10)
        ax.set_xlabel('Prediksi', color='#64748b', fontsize=9)
        ax.set_ylabel('Aktual', color='#64748b', fontsize=9)
        ax.set_title(title, color='#e2e8f0', fontsize=11, pad=10, fontweight='600')
        ax.tick_params(colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2d3147')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white', fontsize=14, fontweight='700')

    for col, (pred, title, report) in zip(
        [c1, c2],
        [
            (models['pred_w'], "Model Berbasis Kata", rw),
            (models['pred_c'], "Model Berbasis Karakter", rc),
        ]
    ):
        with col:
            cm = confusion_matrix(models['y_test'], pred)
            fig, ax = plt.subplots(figsize=(4, 3.2))
            draw_cm(ax, cm, title)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # mini metrics under each
            m1, m2, m3 = st.columns(3)
            for mc, (mname, mval) in zip(
                [m1, m2, m3],
                [("Precision", report['Spam']['precision']),
                 ("Recall", report['Spam']['recall']),
                 ("F1-Score", report['Spam']['f1-score'])]
            ):
                mc.markdown(f"""
                <div class="metric-mini" style="padding:.6rem .8rem">
                  <div class="val" style="font-size:1.1rem;color:#a78bfa">{mval*100:.1f}%</div>
                  <div class="lbl">{mname}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='sg-divider'>", unsafe_allow_html=True)

    # Error analysis
    st.markdown('<p class="sg-section"><span>③</span> Analisis Kesalahan (Model Terbaik)</p>', unsafe_allow_html=True)
    best_pred = models['pred_w'] if rw['Spam']['f1-score'] >= rc['Spam']['f1-score'] else models['pred_c']
    cm_b = confusion_matrix(models['y_test'], best_pred)
    fp, fn = cm_b[0, 1], cm_b[1, 0]

    e1, e2 = st.columns(2)
    with e1:
        st.markdown(f"""
        <div class="sg-card">
          <p style="color:#fbbf24;font-weight:600;margin:0 0 .5rem">⚠️ False Positive: {fp} pesan</p>
          <p style="color:#94a3b8;font-size:.9rem;margin:0">Pesan normal yang salah ditandai SPAM.<br>
          Dampak: pengguna kehilangan pesan penting.</p>
        </div>""", unsafe_allow_html=True)
    with e2:
        st.markdown(f"""
        <div class="sg-card">
          <p style="color:#f87171;font-weight:600;margin:0 0 .5rem">🚨 False Negative: {fn} pesan</p>
          <p style="color:#94a3b8;font-size:.9rem;margin:0">Spam yang lolos sebagai NORMAL.<br>
          Dampak: inbox pengguna terkontaminasi.</p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════
#  TAB 3 · TEST CASES
# ══════════════════════════════════════════
with tab3:
    if test_df.empty:
        st.markdown("""<div class="sg-warn">
            ⚠️ File test_cases.csv tidak ditemukan di folder <code>data/</code>.
        </div>""", unsafe_allow_html=True)
    else:
        # Stats
        n_spam = (test_df['label_manual'] == 'spam').sum()
        n_ham  = (test_df['label_manual'] == 'ham').sum()
        cats   = test_df[test_df['label_manual']=='spam']['kategori_spam'].nunique()

        st.markdown('<p class="sg-section"><span>①</span> Statistik Dataset Uji</p>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        for mc, (lbl, val, color) in zip(
            [m1, m2, m3, m4],
            [("Total Test Case", len(test_df), "#a78bfa"),
             ("Spam", n_spam, "#f87171"),
             ("Ham", n_ham, "#34d399"),
             ("Kategori Spam", cats, "#60a5fa")]
        ):
            mc.markdown(f"""
            <div class="metric-mini">
              <div class="val" style="color:{color}">{val}</div>
              <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='sg-divider'>", unsafe_allow_html=True)

        # Run model on all test cases
        st.markdown('<p class="sg-section"><span>②</span> Evaluasi Batch</p>', unsafe_allow_html=True)

        if st.button("▶  Jalankan Evaluasi pada Semua Test Case", use_container_width=False):
            with st.spinner("Mengevaluasi…"):
                msgs = test_df['pesan'].tolist()
                pw = models['nb_w'].predict_proba(models['vec_w'].transform(msgs))
                pc = models['nb_c'].predict_proba(models['vec_c'].transform(msgs))
                scores = (pw[:, 1] + pc[:, 1]) / 2
                preds  = (scores >= 0.5).astype(int)
                truth  = test_df['label_manual'].map({'spam': 1, 'ham': 0}).values

                test_df['prediksi'] = preds
                test_df['label_pred'] = test_df['prediksi'].map({1: '🚫 SPAM', 0: '✅ HAM'})
                test_df['label_truth'] = test_df['label_manual'].map({'spam': '🚫 SPAM', 'ham': '✅ HAM'})
                test_df['score_spam'] = (scores * 100).round(1)
                test_df['benar'] = (preds == truth)

                acc = test_df['benar'].mean() * 100
                st.markdown(f"""
                <div class="sg-card" style="text-align:center">
                  <p style="color:#34d399;font-size:1.8rem;font-weight:700;margin:0">{acc:.1f}%</p>
                  <p style="color:#64748b;margin:.2rem 0 0">Akurasi pada {len(test_df)} test case</p>
                </div>""", unsafe_allow_html=True)

                # Show mistakes
                wrong = test_df[~test_df['benar']]
                if len(wrong) > 0:
                    st.markdown(f'<p class="sg-section"><span>③</span> Kesalahan Prediksi ({len(wrong)} pesan)</p>',
                                unsafe_allow_html=True)
                    st.dataframe(
                        wrong[['id','pesan','label_truth','label_pred','score_spam','kategori_spam']],
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.success("🎉 Semua test case diprediksi dengan benar!")

        st.markdown("<hr class='sg-divider'>", unsafe_allow_html=True)

        # Browse & filter
        st.markdown('<p class="sg-section"><span>③</span> Jelajahi Test Case</p>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            filter_label = st.selectbox("Filter label:", ["Semua", "Spam", "Ham"], key="tc_filter")
        with c2:
            cats_list = ["Semua"] + sorted(test_df['kategori_spam'].dropna().unique().tolist())
            filter_cat = st.selectbox("Filter kategori:", cats_list, key="tc_cat")

        df_view = test_df.copy()
        if filter_label != "Semua":
            df_view = df_view[df_view['label_manual'] == filter_label.lower()]
        if filter_cat != "Semua":
            df_view = df_view[df_view['kategori_spam'] == filter_cat]

        st.markdown(f"<p style='color:#64748b;font-size:.85rem'>{len(df_view)} data ditampilkan</p>",
                    unsafe_allow_html=True)
        st.dataframe(
            df_view[['id','pesan','label_manual','kategori_spam','catatan']],
            use_container_width=True, hide_index=True
        )

        # Category distribution chart
        if filter_label in ["Semua", "Spam"]:
            st.markdown("<hr class='sg-divider'>", unsafe_allow_html=True)
            st.markdown('<p class="sg-section"><span>④</span> Distribusi Kategori Spam</p>',
                        unsafe_allow_html=True)
            cat_counts = (
                test_df[test_df['label_manual']=='spam']['kategori_spam']
                .value_counts()
                .head(15)
            )
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#1e2130')
            ax.set_facecolor('#1e2130')
            bars = ax.barh(cat_counts.index[::-1], cat_counts.values[::-1],
                           color='#7c3aed', alpha=0.85, height=0.6)
            ax.tick_params(colors='#94a3b8', labelsize=9)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.xaxis.set_tick_params(colors='#64748b')
            ax.set_xlabel("Jumlah", color='#64748b', fontsize=9)
            for bar, val in zip(bars, cat_counts.values[::-1]):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        str(val), va='center', color='#a78bfa', fontsize=9, fontweight='600')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ══════════════════════════════════════════
#  TAB 4 · ABOUT
# ══════════════════════════════════════════
with tab4:
    st.markdown('<p class="sg-section"><span>①</span> Tentang SpamGuard</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sg-card">
      <p style="color:#cbd5e1;line-height:1.8;margin:0">
        <b style="color:#a78bfa">SpamGuard</b> adalah sistem deteksi spam berbasis
        <b>Multinomial Naive Bayes</b> yang membandingkan dua pendekatan ekstraksi fitur:
        <b>word-level</b> (bag-of-words) dan <b>character n-gram (3–5)</b>.
        Sistem voting sederhana menggabungkan kedua model untuk keputusan akhir.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sg-section"><span>②</span> Perbandingan Model</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="sg-card">
          <p style="color:#60a5fa;font-weight:600;margin:0 0 .75rem">📝 Model Berbasis Kata</p>
          <p style="color:#94a3b8;font-size:.88rem;line-height:1.7;margin:0">
            ✓ Cepat &amp; mudah diinterpretasi<br>
            ✓ Cocok untuk spam dengan kata kunci eksplisit<br>
            ✗ Rentan terhadap typo &amp; variasi ejaan<br>
            ✗ Tidak menangkap pola sub-kata
          </p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="sg-card">
          <p style="color:#f472b6;font-weight:600;margin:0 0 .75rem">🔤 Model Berbasis Karakter</p>
          <p style="color:#94a3b8;font-size:.88rem;line-height:1.7;margin:0">
            ✓ Tangguh terhadap typo &amp; obfuscation<br>
            ✓ Menangkap pola karakter spam (!!!,  ALL CAPS)<br>
            ✗ Dimensi fitur sangat besar<br>
            ✗ Lebih lambat saat inferensi
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="sg-section"><span>③</span> Metrik Evaluasi</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sg-card">
      <table style="width:100%;border-collapse:collapse;color:#cbd5e1;font-size:.88rem">
        <thead>
          <tr style="border-bottom:1px solid #2d3147">
            <th style="text-align:left;padding:.5rem .75rem;color:#a78bfa">Metrik</th>
            <th style="text-align:left;padding:.5rem .75rem;color:#a78bfa">Definisi</th>
            <th style="text-align:left;padding:.5rem .75rem;color:#a78bfa">Dampak jika rendah</th>
          </tr>
        </thead>
        <tbody>
          <tr style="border-bottom:1px solid #1e2130">
            <td style="padding:.5rem .75rem;font-weight:600;color:#60a5fa">Precision</td>
            <td style="padding:.5rem .75rem">Dari yang diprediksi spam, berapa yang benar spam</td>
            <td style="padding:.5rem .75rem;color:#fbbf24">Banyak pesan normal masuk folder spam</td>
          </tr>
          <tr style="border-bottom:1px solid #1e2130">
            <td style="padding:.5rem .75rem;font-weight:600;color:#60a5fa">Recall</td>
            <td style="padding:.5rem .75rem">Dari semua spam, berapa yang berhasil ditangkap</td>
            <td style="padding:.5rem .75rem;color:#fbbf24">Banyak spam lolos ke inbox</td>
          </tr>
          <tr>
            <td style="padding:.5rem .75rem;font-weight:600;color:#60a5fa">F1-Score</td>
            <td style="padding:.5rem .75rem">Rata-rata harmonis Precision dan Recall</td>
            <td style="padding:.5rem .75rem;color:#fbbf24">Model tidak seimbang antara dua metrik</td>
          </tr>
        </tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sg-section"><span>④</span> Disclaimer</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sg-warn">
      ⚠️ <b>Disclaimer:</b> Aplikasi ini adalah alat bantu berbasis probabilitas.
      Hasil prediksi <b>bukan alat bukti hukum</b> dan memiliki tingkat kesalahan.
      Selalu verifikasi informasi penting melalui saluran resmi.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sg-section"><span>⑤</span> Stack Teknologi</p>', unsafe_allow_html=True)
    chips = ["Streamlit", "scikit-learn", "Multinomial Naive Bayes", "CountVectorizer",
             "Word n-gram", "Char n-gram (3–5)", "Python 3.11"]
    st.markdown("".join(f'<span class="chip">{c}</span>' for c in chips), unsafe_allow_html=True)