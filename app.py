import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

from database_helper import init_db, insert_history, fetch_history, verify_user
from model_logic import (
    HyperParams,
    download_data,
    slice_1000,
    train_and_evaluate,
    predict_next_day,
    save_artifacts,
    load_artifacts,
    clear_artifacts,
    predict_close_series,
    predict_train_test_series,
    get_dataset,
)

# =========================
# 1) PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Sistem Prediksi Saham BUMN (Fokus BBRI)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_db()

# =========================
# 2) SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "role" not in st.session_state:
    st.session_state["role"] = ""

if "hp" not in st.session_state:
    arts = load_artifacts()
    hp0: HyperParams = arts["hp"]
    st.session_state["hp"] = {
        "window": int(hp0.window),
        "epochs": int(hp0.epochs),
        "hidden_size": int(hp0.hidden_size),
        "lr": float(hp0.lr),
        "batch_size": int(hp0.batch_size),
    }

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if "last_metrics" not in st.session_state:
    st.session_state["last_metrics"] = None

if "last_train_losses" not in st.session_state:
    st.session_state["last_train_losses"] = None

if "show_charts" not in st.session_state:
    st.session_state["show_charts"] = False

if "rw_selected_id" not in st.session_state:
    st.session_state["rw_selected_id"] = None

if "rw_prev_page" not in st.session_state:
    st.session_state["rw_prev_page"] = None

# Tracker emiten & mode data — untuk deteksi perubahan konteks
if "active_kode_saham" not in st.session_state:
    st.session_state["active_kode_saham"] = None

if "active_mode_data" not in st.session_state:
    st.session_state["active_mode_data"] = None

# =========================
# 3) CSS (DESAIN ANDA)
# =========================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@48,700,1,0" />

<style>
    html, body, p, label, div, h1, h2, h3, h4, input, button, select, span {
        font-family: 'Plus Jakarta Sans', sans-serif !important; 
        color: #1E293B;
    }
    .stApp { background-color: #F8FAFC; }

    p, label, div, input, button { font-size: 1.55rem !important; line-height: 1.6 !important; }

    .material-symbols-rounded, [data-testid="stIconMaterial"] {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal;
        line-height: 1;
        direction: ltr;
        white-space: nowrap;
        text-transform: none;
    }

    h1 {
        color: #0F172A; 
        font-weight: 800; 
        font-size: 4.2rem !important;
        margin-bottom: 18px;
        display: flex; align-items: center; gap: 16px; 
    }
    h1 .material-symbols-rounded { font-size: 5rem !important; color: #2563EB; }

    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {
        font-size: 1.55rem !important;
        height: 4.8rem !important; 
        border-radius: 12px;
        border: 2px solid #E2E8F0;
        background-color: #FFFFFF;
        padding-left: 15px;
    }

    /* Primary buttons */
    .stButton > button {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1.7rem !important;
        height: 4.8rem !important;
        font-weight: 800;
        border-radius: 12px;
        border: none;
        background: #2563EB;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        color: #FFFFFF !important;
        margin-top: 6px;
        transition: all 0.2s ease !important; 
    }
    
    .stButton > button * { color: #FFFFFF !important; fill: #FFFFFF !important; }


    .stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8, #4338CA) !important;
        box-shadow: 0 6px 12px -1px rgba(37,99,235,0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* Pagination nav buttons — abu-abu */
    div.rw-nav-btn .stButton > button {
        background: #64748B !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.10) !important;
        color: #FFFFFF !important;
    }
    div.rw-nav-btn .stButton > button:disabled {
        background: #E2E8F0 !important;
        color: #94A3B8 !important;
        cursor: not-allowed !important;
        box-shadow: none !important;
    }

    /* Gear button */
    div[data-testid="stPopover"] > button {
        height: 4rem !important; width: 4rem !important; padding: 0 !important;
        background: #F1F5F9 !important; color: #334155 !important;
        box-shadow: none !important; border: 1px solid #CBD5E1 !important;
        display: flex; align-items: center; justify-content: center; margin-top: 0 !important;
    }

    /* Big popover modal */
    div[data-testid="stPopoverBody"] {
        position: fixed !important;
        top: 50% !important; left: 50% !important;
        transform: translate(-50%, -50%) !important;
        width: 650px !important;
        padding: 38px !important;
        max-height: 85vh !important; 
        border: 4px solid #2563EB !important;
        box-shadow: 0 0 250px rgba(0,0,0,0.6) !important; 
        border-radius: 25px !important;
        z-index: 999999 !important; 
        background-color: white !important;
    }
    div[data-testid="stPopoverBody"] label p {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 10px !important;
    }
    div[data-testid="stPopoverBody"] > div[data-testid="stPopoverArrow"] { display: none !important; }

    /* Cards */
    .step-card {
        background: white; border-radius: 16px; padding: 30px; height: 100%;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.05);
        border: 2px solid #F1F5F9;
    }
    .step-card .material-symbols-rounded { font-size: 5.6rem !important; margin-bottom: 14px; display: block; }

    .metric-card {
        background-color: white; padding: 28px; border-radius: 16px; text-align: center;
        border: 2px solid #F1F5F9; box-shadow: 0 4px 10px rgba(15, 23, 42, 0.05);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { 
        width: 420px !important; background-color: white !important; border-right: 1px solid #E2E8F0; 
    }

    .welcome-banner {
        background: linear-gradient(135deg, #2563EB, #4F46E5); color: white;
        padding: 54px; border-radius: 22px; margin-bottom: 22px;
        box-shadow: 0 10px 18px rgba(37, 99, 235, 0.20);
    }

    /* Clean table */
    .clean_table { width: 100%; border-collapse: separate; border-spacing: 0 6px; margin-top: 12px; }
    .clean_table thead tr th { font-size: 1.7rem !important; font-weight: 900; color: #64748B; padding: 14px 14px; border-bottom: 3px solid #E2E8F0; }
    .clean_table tbody tr td { font-size: 1.5rem !important; padding: 14px 14px; background-color: #FFFFFF; border-top: 1px solid #F1F5F9; border-bottom: 1px solid #F1F5F9; font-weight: 600; color: #334155; }
    .clean_table tbody tr td:first-child { border-top-left-radius: 12px; border-bottom-left-radius: 12px; border-left: 1px solid #F1F5F9; color: #2563EB; font-weight: 900; }
    .clean_table tbody tr td:last-child { border-top-right-radius: 12px; border-bottom-right-radius: 12px; border-right: 1px solid #F1F5F9; }
</style>
""", unsafe_allow_html=True)

# =========================
# 4) UI HELPERS
# =========================
def custom_metric_card(label, value, delta="", color_text="#2563EB"):
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="color:#64748B; margin:0; font-size: 1.5rem; font-weight: 900;">{label}</h4>
        <h2 style="font-size: 3.6rem; margin: 14px 0; color:#0F172A; font-weight: 900;">{value}</h2>
        <p style="color:{color_text}; font-weight: 900; margin:0; font-size: 1.35rem;">{delta}</p>
    </div>
    """, unsafe_allow_html=True)

def draw_step_card(icon_name, title, text, color_border):
    st.markdown(f"""
    <div class="step-card" style="border-left: 10px solid {color_border};">
        <div style="color: {color_border};">
            <span class="material-symbols-rounded">{icon_name}</span>
        </div>
        <div style="font-size: 2rem; font-weight: 900; color: {color_border}; margin-bottom: 10px;">{title}</div>
        <p style="color: #64748B; font-size: 1.45rem; line-height: 1.6; font-weight: 700;">{text}</p>
    </div>
    """, unsafe_allow_html=True)

def df_tail_for_plot(kode_saham: str, mode_data: str) -> pd.DataFrame:
    df = get_dataset(kode_saham, mode_data)

    df_plot = df[["Date", "Close"]].copy().tail(90).reset_index(drop=True)
    res = st.session_state.get("last_result")

    pred_series = [np.nan] * len(df_plot)
    if res and res.get("kode_saham") == kode_saham:
        pred_series[-1] = float(res["next_close_pred"])
    df_plot["Prediksi (Next)"] = pred_series
    return df_plot

# =========================
# 5) PAGES
# =========================
def halaman_login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='justify-content:center;'><span class='material-symbols-rounded'>lock</span> AKUN SISTEM</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748B; font-weight:700;'>Login untuk mengakses aplikasi. Hubungi administrator untuk mendapatkan akun.</p>", unsafe_allow_html=True)
        st.divider()

        user = st.text_input("Username", placeholder="admin", key="login_user")
        pw = st.text_input("Password", type="password", placeholder="admin", key="login_pw")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("MASUK", icon=":material/rocket_launch:", type="primary", use_container_width=True, key="btn_login"):
            ok, role, msg = verify_user(user, pw)
            if ok:
                st.session_state["logged_in"] = True
                st.session_state["username"] = user
                st.session_state["role"] = role
                st.toast("✅ Login berhasil!", icon="✅")
                st.rerun()
            else:
                st.error(msg)

def halaman_beranda():

    st.markdown(f"""
    <div class="welcome-banner">
        <h1 style="color:white; margin:0; font-size: 4.3rem !important; justify-content:center;">
            Selamat Datang, {st.session_state['username']}! 👋
        </h1>
        <p style="font-size: 1.9rem !important; margin-top: 12px; color:#DBEAFE; font-weight: 700;">
            Sistem Prediksi Harga Saham Bank BUMN berbasis <b>LSTM</b> (Penelitian difokuskan pada <b>BBRI</b>)
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: draw_step_card("touch_app", "1. Menu", "Buka menu <b>'Prediksi Saham'</b> di sidebar.", "#F59E0B")
    with c2: draw_step_card("account_balance", "2. Saham", "Pilih bank dari dropdown.", "#2563EB")
    with c3: draw_step_card("bolt", "3. Proses", "Klik <b>Latih Ulang</b> (jika ubah parameter) lalu <b>Prediksi</b>.", "#10B981")
    with c4: draw_step_card("history", "4. Riwayat", "Cek hasil tersimpan di menu <b>Riwayat</b>.", "#EF4444")

def halaman_prediksi(kode_saham: str, mode_data: str):
    hp = st.session_state["hp"]

    # ── Deteksi perubahan emiten atau mode data ──────────────────────────────
    prev_saham = st.session_state.get("active_kode_saham")
    prev_mode  = st.session_state.get("active_mode_data")
    context_changed = (
        (prev_saham is not None and prev_saham != kode_saham) or
        (prev_mode  is not None and prev_mode  != mode_data)
    )
    if context_changed:
        model_exists = os.path.exists("model_state.pt")
        # Hapus model lama dari disk agar tidak bocor ke konteks baru
        clear_artifacts()
        # Reset session state yang berkaitan dengan prediksi
        st.session_state["last_result"]      = None
        st.session_state["last_metrics"]     = None
        st.session_state["last_train_losses"] = None
        st.session_state["show_charts"]      = False
        # Peringatan hanya muncul jika model lama memang ada
        if model_exists:
            st.toast(
                f"⚠️ Emiten/mode data berubah → model lama dihapus. "
                f"Silakan **Latih Ulang Model** untuk {kode_saham} ({mode_data}).",
                icon="🗑️"
            )

    # Perbarui tracker konteks aktif
    st.session_state["active_kode_saham"] = kode_saham
    st.session_state["active_mode_data"]  = mode_data
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown(
        f"<h1><span class='material-symbols-rounded'>analytics</span> Prediksi Saham: "
        f"<span style='color:#2563EB'>{kode_saham}</span></h1>",
        unsafe_allow_html=True
    )
    st.caption(f"🔧 Parameter aktif: Window={hp['window']} | Epochs={hp['epochs']} | Units={hp['hidden_size']} | LR={hp['lr']} | Batch={hp['batch_size']}")

    st.info("Gunakan **Latih Ulang Model** hanya saat hyperparameter diubah. Tombol **Prediksi** tidak retrain (lebih cepat).")

    b1, b2 = st.columns([1, 1])
    with b1:
        if st.session_state.get("role") != "admin":
            st.info("Fitur **Latih Ulang Model** hanya untuk **Admin** agar aman saat dipakai bersamaan (1 localhost).")
        else:
            if st.button("🔄 Latih Ulang Model (retrain)", use_container_width=True):
                with st.spinner("Download data → preprocessing → training LSTM → evaluasi ..."):
                    try:
                        df = get_dataset(kode_saham, mode_data)

                        hp_obj = HyperParams(
                            window=int(hp["window"]),
                            epochs=int(hp["epochs"]),
                            hidden_size=int(hp["hidden_size"]),
                            lr=float(hp["lr"]),
                            batch_size=int(hp["batch_size"]),
                        )
                        out = train_and_evaluate(df, hp_obj, shuffle=(mode_data != "research_first_1000"))

                        # simpan metrik & train_losses terakhir di session
                        st.session_state["last_metrics"]      = out["metrics"]
                        st.session_state["last_train_losses"] = out["train_losses"]

                        save_artifacts(
                            model=out["model"],
                            scaler=out["scaler"],
                            hp=out["hp"],
                            metrics={
                                "mse_test"  : out["metrics"]["mse_test"],
                                "rmse_test" : out["metrics"]["rmse_test"],
                                "mse_train" : out["metrics"]["mse_train"],
                                "rmse_train": out["metrics"]["rmse_train"],
                            },
                            out_dir=".",
                            model_path="model_state.pt",
                            scaler_path="scaler.joblib",
                            config_path="config.json",
                            metrics_path="metrics.json",
                        )
                        st.success(f"Training selesai. RMSE test = {out['metrics']['rmse_test']:.2f}")
                        # sembunyikan grafik sampai user klik Prediksi lagi
                        st.session_state["show_charts"] = False
                        st.session_state["last_result"] = None

                    except (ConnectionError, OSError, TimeoutError, Exception) as _e:
                        _msg = str(_e).lower()
                        if any(k in _msg for k in ["kosong", "empty", "network", "connection", "timeout", "internet", "resolve", "errno", "failed to fetch", "remotedisconnected"]):
                            st.warning("⚠️ Gagal mengunduh data. Pastikan koneksi internet Anda aktif, lalu coba lagi.")
                        else:
                            st.error(f"Terjadi kesalahan: {_e}")


    with b2:
        if st.button("⚡ Prediksi (tanpa retrain)", use_container_width=True):
            with st.spinner("Memuat model → mengambil data → prediksi 1 hari ..."):
                arts = load_artifacts()
                if arts["scaler"] is None:
                    st.error("Scaler belum ditemukan. Jalankan dulu **Latih Ulang Model** (atau pastikan scaler.joblib ada).")
                    return

                model = arts["model"]
                scaler = arts["scaler"]
                hp_loaded: HyperParams = arts["hp"]

                try:
                    df = get_dataset(kode_saham, mode_data)
                except Exception:
                    st.warning("⚠️ Gagal mengunduh data. Pastikan koneksi internet Anda aktif, lalu coba lagi.")
                    st.stop()

                pred = predict_next_day(model, scaler, df, window=int(hp_loaded.window))
                metrics = arts["metrics"] or {}
                # jika baru retrain, metrik session lebih up-to-date
                if st.session_state.get("last_metrics") is not None:
                    metrics = {**metrics, **st.session_state["last_metrics"]}

                st.session_state["last_result"] = {
                    "kode_saham"     : kode_saham,
                    "mode_data"      : mode_data,
                    "last_close"     : pred["last_close"],
                    "next_close_pred": pred["next_close_pred"],
                    "next_date"      : pred["next_date"],
                    "rmse"           : metrics.get("rmse_test", None),
                    "mse"            : metrics.get("mse_test", None),
                }

                import json as _json
                _hp_cfg = _json.dumps({
                    "window"     : int(hp.get("window", 60)),
                    "epochs"     : int(hp.get("epochs", 50)),
                    "hidden_size": int(hp.get("hidden_size", 64)),
                    "lr"         : float(hp.get("lr", 0.001)),
                    "batch_size" : int(hp.get("batch_size", 32)),
                })
                insert_history(
                    username=st.session_state.get('username',''),
                    kode_saham=kode_saham,
                    mse_score=float(metrics.get("mse_test", 0.0)),
                    rmse_score=float(metrics.get("rmse_test", 0.0)),
                    tanggal_prediksi=str(pred["next_date"]),
                    harga_prediksi_final=float(pred["next_close_pred"]),
                    hp_config=_hp_cfg,
                )

                st.toast("✅ Prediksi tersimpan ke riwayat!", icon="📌")
                st.session_state["show_charts"] = True

    res = st.session_state.get("last_result")
    res_mode = res.get("mode_data", None) if res else None
    if (not res
            or res.get("kode_saham") != kode_saham
            or res_mode != mode_data
            or not st.session_state.get("show_charts", False)):
        st.markdown("<br>", unsafe_allow_html=True)
        if res and (res.get("kode_saham") != kode_saham or res_mode != mode_data):
            st.info("Mode data atau emiten berubah. Klik **Prediksi** atau **Latih Ulang** untuk memperbarui grafik.")
        else:
            st.info("Klik **Prediksi** untuk menampilkan ringkasan & grafik.")
        return

    # Gunakan mode_data dari saat prediksi terakhir, bukan sidebar aktif
    render_mode = res.get("mode_data", mode_data)

    # Load artifacts & data DULU sebelum render apapun
    arts      = load_artifacts()
    model     = arts["model"]
    scaler    = arts["scaler"]
    hp_loaded : HyperParams = arts["hp"]

    try:
        df_full = get_dataset(kode_saham, render_mode)
    except Exception:
        st.warning("⚠️ Gagal memuat data grafik. Pastikan koneksi internet aktif, lalu klik **Prediksi** kembali.")
        st.session_state["show_charts"] = False
        st.session_state["last_result"] = None
        return

    # Semua data siap — render kartu & grafik
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: custom_metric_card("Harga Terakhir", f"Rp {res['last_close']:,.0f}".replace(",", "."), "Data Aktual", "#334155")
    with c2: custom_metric_card("Prediksi Berikutnya", f"Rp {res['next_close_pred']:,.0f}".replace(",", "."), f"Tanggal: {res['next_date']}", "#10B981")
    with c3: custom_metric_card("RMSE (Test)", "-" if res["rmse"] is None else f"{res['rmse']:.2f}", "Semakin kecil semakin baik", "#2563EB")
    with c4: custom_metric_card("MSE (Test)", "-" if res["mse"] is None else f"{res['mse']:.2f}", "Semakin kecil semakin baik", "#2563EB")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## Grafik Harga Aktual vs Prediksi")

    df_series = predict_close_series(
        model=model,
        scaler=scaler,
        df=df_full,
        window=int(hp_loaded.window),
        n_points=120,
    )

    # 2) Garis prediksi besok (dotted extension)
    next_date = pd.to_datetime(res["next_date"])
    last_date = pd.to_datetime(df_series["Date"].iloc[-1])
    last_pred = float(df_series["Pred"].iloc[-1])
    next_pred = float(res["next_close_pred"])

    fig = px.line(
        df_series,
        x="Date",
        y=["Actual", "Pred"],
        template="plotly_white",
        labels={"value": "Harga", "variable": ""}
    )
    fig.update_traces(line={"width": 3})
    fig.update_layout(
        height  = 560,
        legend  = dict(
            orientation = "h",
            y           = 1.1,
            font        = dict(size=20),
        ),
        xaxis   = dict(
            title       = dict(text="Date", font=dict(size=20)),
            tickfont    = dict(size=18),
        ),
        yaxis   = dict(
            title       = dict(text="Harga (Rp)", font=dict(size=20)),
            tickfont    = dict(size=18),
        ),
        font    = dict(size=19),
    )

    # Tambah garis dotted dari prediksi terakhir -> prediksi besok
    fig.add_trace(
        go.Scatter(
            x=[last_date, next_date],
            y=[last_pred, next_pred],
            mode="lines+markers",
            name="Prediksi Besok",
            line=dict(width=3, dash="dot"),
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── GRAFIK AKTUAL vs PREDIKSI TRAIN & TEST ───────────────
    st.markdown("## Grafik Harga Aktual vs Prediksi Train & Test")

    df_full_seg = predict_train_test_series(
        model      = model,
        scaler     = scaler,
        df         = df_full,
        window     = int(hp_loaded.window),
        test_ratio = 0.2,
    )

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x    = df_full_seg["Date"],
        y    = df_full_seg["Actual"],
        mode = "lines", name = "Actual",
        line = dict(color="#2563EB", width=3),
    ))

    df_train_only = df_full_seg[df_full_seg["Segment"] == "Train"]
    fig2.add_trace(go.Scatter(
        x    = df_train_only["Date"],
        y    = df_train_only["Pred_Train"],
        mode = "lines", name = "Pred Train (80%)",
        line = dict(color="#10B981", width=3),
    ))

    df_test_only = df_full_seg[df_full_seg["Segment"] == "Test"]
    fig2.add_trace(go.Scatter(
        x    = df_test_only["Date"],
        y    = df_test_only["Pred_Test"],
        mode = "lines", name = "Pred Test (20%)",
        line = dict(color="#ff4800", width=3),
    ))

    n_total       = len(df_full)
    split_idx     = int(n_total * 0.8)
    split_date_dt = pd.to_datetime(df_full["Date"].iloc[split_idx])
    split_date_str= split_date_dt.strftime("%d %b %Y")

    fig2.add_shape(
        type="line", xref="x", yref="paper",
        x0=split_date_dt, x1=split_date_dt, y0=0, y1=1,
        line=dict(color="#94A3B8", width=2, dash="dash"),
    )
    fig2.add_annotation(
        x=split_date_dt, yref="paper", y=1.02,
        text=f"Split Train|Test ({split_date_str})",
        showarrow=False,
        font=dict(size=18, color="#64748B"),
        xanchor="left",
    )

    metrics_now = arts.get("metrics") or {}
    if st.session_state.get("last_metrics"):
        metrics_now = {**metrics_now, **st.session_state["last_metrics"]}
    rmse_train_val = metrics_now.get("rmse_train", None)
    rmse_test_val  = metrics_now.get("rmse_test",  None)
    if rmse_train_val and rmse_test_val:
        gap_val = rmse_test_val / rmse_train_val
        fig2.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.97,
            text=f"RMSE Train: {rmse_train_val:.2f} | RMSE Test: {rmse_test_val:.2f} | GAP: {gap_val:.4f}",
            showarrow=False,
            font=dict(size=19, color="#0F172A"),
            bgcolor="#F1F5F9",
            bordercolor="#CBD5E1",
            borderwidth=1,
            borderpad=8,
            align="left",
        )

    fig2.update_traces(line={"width": 3})
    fig2.update_layout(
        template  = "plotly_white",
        height    = 560,
        xaxis     = dict(title=dict(text="Date", font=dict(size=20)), tickfont=dict(size=18)),
        yaxis     = dict(title=dict(text="Harga (Rp)", font=dict(size=20)), tickfont=dict(size=18)),
        legend    = dict(orientation="h", y=1.1, font=dict(size=20)),
        hovermode = "x unified",
        font      = dict(size=19),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── INFO PEMBAGIAN DATA 80:20 ─────────────────────────────
    st.markdown("## Pembagian Data (80% Train / 20% Test)")

    try:
        df_info = get_dataset(kode_saham, render_mode)
    except Exception:
        df_info = df_full.copy()
    total   = len(df_info)
    n_train = int(total * 0.8)
    n_test  = total - n_train
    window  = int(hp_loaded.window)
    n_train_seq = n_train - window
    n_test_seq  = n_test  - window

    date_min       = df_info["Date"].min().strftime("%d %b %Y")
    date_split     = df_info["Date"].iloc[n_train - 1].strftime("%d %b %Y")
    date_max       = df_info["Date"].max().strftime("%d %b %Y")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 8px solid #2563EB;">
            <h4 style="color:#64748B; margin:0; font-size:1.4rem; font-weight:900;">Total Dataset</h4>
            <h2 style="font-size:3.2rem; margin:10px 0; color:#0F172A; font-weight:900;">{total:,} baris</h2>
            <p style="color:#64748B; margin:0; font-size:1.3rem; font-weight:700;">
                {date_min} &nbsp;→&nbsp; {date_max}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 8px solid #10B981;">
            <h4 style="color:#64748B; margin:0; font-size:1.4rem; font-weight:900;">Data Latih (Train 80%)</h4>
            <h2 style="font-size:3.2rem; margin:10px 0; color:#10B981; font-weight:900;">{n_train:,} baris</h2>
            <p style="color:#64748B; margin:0; font-size:1.3rem; font-weight:700;">
                {date_min} &nbsp;→&nbsp; {date_split}<br>
                Sekuens: {n_train_seq:,} sampel × {window} hari × 5 fitur
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 8px solid #ff7f0e;">
            <h4 style="color:#64748B; margin:0; font-size:1.4rem; font-weight:900;">Data Uji (Test 20%)</h4>
            <h2 style="font-size:3.2rem; margin:10px 0; color:#ff7f0e; font-weight:900;">{n_test:,} baris</h2>
            <p style="color:#64748B; margin:0; font-size:1.3rem; font-weight:700;">
                {date_split} &nbsp;→&nbsp; {date_max}<br>
                Sekuens: {n_test_seq:,} sampel × {window} hari × 5 fitur
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Progress bar visual pembagian data
    pct_train = n_train / total * 100
    pct_test  = n_test  / total * 100
    st.markdown(f"""
    <div style="margin-top:10px;">
        <p style="font-size:1.4rem; font-weight:800; color:#475569; margin-bottom:8px;">Proporsi Pembagian Data</p>
        <div style="display:flex; border-radius:12px; overflow:hidden; height:42px; box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <div style="width:{pct_train:.1f}%; background:#2563EB; display:flex; align-items:center; justify-content:center;">
                <span style="color:white; font-weight:900; font-size:1.5rem;">Train {pct_train:.0f}% ({n_train:,} baris)</span>
            </div>
            <div style="width:{pct_test:.1f}%; background:#ff7f0e; display:flex; align-items:center; justify-content:center;">
                <span style="color:white; font-weight:900; font-size:1.5rem;">Test {pct_test:.0f}% ({n_test:,} baris)</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

def halaman_riwayat():
    import json as _json
    st.markdown("<h1><span class='material-symbols-rounded'>history</span> Riwayat Prediksi</h1>", unsafe_allow_html=True)

    df = fetch_history(
        username=st.session_state.get('username',''),
        role=st.session_state.get('role','user'),
        limit=500,
        show_all_if_admin=False
    )
    if df.empty:
        st.warning("Belum ada riwayat. Jalankan **Prediksi** terlebih dahulu.")
        return

    # ── FILTER SECTION ───────────────────────────────────────
    st.markdown("### 🔍 Filter & Urutkan")
    fc1, fc2 = st.columns(2)

    with fc1:
        saham_opts = ["Semua"] + sorted(df["kode_saham"].unique().tolist())
        filter_saham = st.selectbox("Kode Saham", saham_opts, key="rw_saham")

    with fc2:
        sort_by = st.selectbox(
            "Urutkan berdasarkan",
            ["Terbaru", "MSE & RMSE Terendah"],
            key="rw_sort"
        )

    per_page = 10

    st.markdown("<br>", unsafe_allow_html=True)

    # Terapkan filter saham
    df_f = df.copy()
    if filter_saham != "Semua":
        df_f = df_f[df_f["kode_saham"] == filter_saham]

    # Terapkan urutan
    if sort_by == "MSE & RMSE Terendah":
        df_f = df_f.sort_values(["mse_score", "rmse_score"], ascending=[True, True]).reset_index(drop=True)
    else:
        df_f = df_f.sort_values("waktu_eksekusi", ascending=False).reset_index(drop=True)

    total_rows = len(df_f)
    total_pages = max(1, (total_rows + per_page - 1) // per_page)

    # ── INFO BEST SCORE ───────────────────────────────────────
    if not df_f.empty:
        best_mse_row  = df_f.loc[df_f["mse_score"].idxmin()]
        best_rmse_row = df_f.loc[df_f["rmse_score"].idxmin()]
        bi1, bi2 = st.columns(2)
        with bi1:
            st.markdown(f"""
            <div class="metric-card" style="border-left:6px solid #10B981;">
                <p style="color:#64748B;font-size:1.4rem;margin:0;font-weight:700;">MSE Terendah</p>
                <h2 style="font-size:2.8rem;color:#10B981;margin:6px 0;font-weight:900;">{best_mse_row['mse_score']:.2f}</h2>
                <p style="color:#64748B;font-size:1.3rem;margin:0;">
                    ID #{int(best_mse_row['id'])} &nbsp;|&nbsp; {best_mse_row['kode_saham']} &nbsp;|&nbsp;
                    {pd.to_datetime(best_mse_row['waktu_eksekusi']).strftime('%d-%m %H:%M')}
                </p>
            </div>
            """, unsafe_allow_html=True)
        with bi2:
            st.markdown(f"""
            <div class="metric-card" style="border-left:6px solid #2563EB;">
                <p style="color:#64748B;font-size:1.4rem;margin:0;font-weight:700;">RMSE Terendah</p>
                <h2 style="font-size:2.8rem;color:#2563EB;margin:6px 0;font-weight:900;">{best_rmse_row['rmse_score']:.2f}</h2>
                <p style="color:#64748B;font-size:1.3rem;margin:0;">
                    ID #{int(best_rmse_row['id'])} &nbsp;|&nbsp; {best_rmse_row['kode_saham']} &nbsp;|&nbsp;
                    {pd.to_datetime(best_rmse_row['waktu_eksekusi']).strftime('%d-%m %H:%M')}
                </p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── PAGINATION ────────────────────────────────────────────
    if "rw_page" not in st.session_state:
        st.session_state["rw_page"] = 1
    # Reset ke hal 1 jika filter berubah
    if st.session_state.get("rw_last_filter") != (filter_saham, sort_by, per_page):
        st.session_state["rw_page"] = 1
        st.session_state["rw_last_filter"] = (filter_saham, sort_by, per_page)

    page     = st.session_state["rw_page"]
    page     = max(1, min(page, total_pages))
    start_i  = (page - 1) * per_page
    end_i    = start_i + per_page
    df_page  = df_f.iloc[start_i:end_i].reset_index(drop=True)

    # ── TABEL AGGRID — klik baris langsung ────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    df_tbl = df_page.copy()
    df_tbl["waktu_eksekusi"]       = pd.to_datetime(df_tbl["waktu_eksekusi"]).dt.strftime("%d-%m %H:%M")
    df_tbl["mse_score"]            = df_tbl["mse_score"].round(2)
    df_tbl["rmse_score"]           = df_tbl["rmse_score"].round(2)
    df_tbl["harga_prediksi_final"] = df_tbl["harga_prediksi_final"].apply(lambda x: f"Rp {x:,.0f}".replace(",","."))
    df_tbl = df_tbl.drop(columns=["username","hp_config"], errors="ignore")
    df_tbl = df_tbl.rename(columns={
        "id"                  : "ID",
        "waktu_eksekusi"      : "Waktu Eksekusi",
        "mse_score"           : "MSE Score",
        "rmse_score"          : "RMSE Score",
        "tanggal_prediksi"    : "Tanggal Prediksi",
        "harga_prediksi_final": "Harga Final",
        "kode_saham"          : "Kode Saham",
    })

    gb = GridOptionsBuilder.from_dataframe(df_tbl)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_default_column(
        resizable  = True,
        sortable   = False,
        filter     = False,
        cellStyle  = {
            "fontSize"  : "22px",
            "fontFamily": "Inter, sans-serif",
            "fontWeight": "600",
            "color"     : "#334155",
            "padding"   : "0 14px",
        },
    )
    gb.configure_column("ID",
        width     = 90,
        cellStyle = {"fontSize":"22px","color":"#2563EB","fontWeight":"900","padding":"0 14px"},
    )
    gb.configure_column("Waktu Eksekusi",   width=190)
    gb.configure_column("MSE Score",        width=160)
    gb.configure_column("RMSE Score",       width=160)
    gb.configure_column("Tanggal Prediksi", width=195)
    gb.configure_column("Harga Final",      width=170)
    gb.configure_column("Kode Saham",       width=150)
    gb.configure_grid_options(
        rowHeight              = 62,
        headerHeight           = 62,
        domLayout              = "autoHeight",
        suppressMovableColumns = True,
        suppressCellFocus      = True,
        rowStyle               = {"cursor": "pointer"},
        getRowStyle            = None,
    )
    # Header font besar via CSS inject
    custom_css = {
        ".ag-header-cell-label": {
            "font-size"  : "22px !important",
            "font-weight": "900 !important",
            "color"      : "#64748B !important",
            "font-family": "Inter, sans-serif !important",
        },
        ".ag-row-selected": {
            "background-color": "#EFF6FF !important",
        },
        ".ag-row:hover": {
            "background-color": "#F8FAFC !important",
        },
        ".ag-cell": {
            "border"     : "none !important",
            "line-height": "62px !important",
        },
        ".ag-header": {
            "border-bottom": "3px solid #E2E8F0 !important",
        },
    }
    grid_opts = gb.build()

    grid_resp = AgGrid(
        df_tbl,
        gridOptions              = grid_opts,
        update_mode              = GridUpdateMode.SELECTION_CHANGED,
        data_return_mode         = DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load = True,
        theme                    = "streamlit",
        height                   = None,
        allow_unsafe_jscode      = True,
        custom_css               = custom_css,
        key                      = f"rw_aggrid_{page}_{filter_saham}_{sort_by}",
    )

    # Ambil baris yang diklik — hanya update jika benar ada seleksi baru
    sel_rows_ag = grid_resp.get("selected_rows", None)
    if sel_rows_ag is not None and len(sel_rows_ag) > 0:
        try:
            sel_id_ag = int(sel_rows_ag.iloc[0]["ID"]) if hasattr(sel_rows_ag, "iloc") else int(sel_rows_ag[0]["ID"])
            st.session_state["rw_selected_id"] = sel_id_ag
        except Exception:
            pass
    # Jangan reset rw_selected_id jika tidak ada seleksi — biarkan tetap dari sebelumnya

    sel_id = st.session_state.get("rw_selected_id", None)

    # ── NAVIGASI HALAMAN ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    /* Tombol navigasi — sama seperti tombol prediksi */
    div[data-testid="stHorizontalBlock"] .stButton > button {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1.7rem !important;
        height: 4.8rem !important;
        font-weight: 800;
        border-radius: 12px;
        border: none;
        background: #2563EB;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
        color: #FFFFFF !important;
        margin-top: 6px;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
        background: linear-gradient(135deg, #1D4ED8, #4338CA) !important;
        box-shadow: 0 6px 12px -1px rgba(37,99,235,0.45) !important;
        transform: translateY(-1px) !important;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button:disabled {
        background: #E2E8F0 !important;
        color: #94A3B8 !important;
        box-shadow: none !important;
        transform: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    pn1, pn2, pn3, pn4, pn5 = st.columns([1, 1, 2, 1, 1])
    with pn1:
        if st.button("⏮ Pertama", use_container_width=True,
                     disabled=(page == 1), key="rw_first"):
            st.session_state["rw_page"] = 1
            st.session_state["rw_selected_id"] = None
            st.rerun()
    with pn2:
        if st.button("◀ Prev", use_container_width=True,
                     disabled=(page == 1), key="rw_prev"):
            st.session_state["rw_page"] = page - 1
            st.session_state["rw_selected_id"] = None
            st.rerun()
    with pn3:
        st.markdown(
            f"<p style='text-align:center;font-size:1.6rem;font-weight:800;"
            f"padding-top:10px;color:#475569;'>Halaman {page} / {total_pages}</p>",
            unsafe_allow_html=True
        )
    with pn4:
        if st.button("Next ▶", use_container_width=True,
                     disabled=(page == total_pages), key="rw_next"):
            st.session_state["rw_page"] = page + 1
            st.session_state["rw_selected_id"] = None
            st.rerun()
    with pn5:
        if st.button("Terakhir ⏭", use_container_width=True,
                     disabled=(page == total_pages), key="rw_last"):
            st.session_state["rw_page"] = total_pages
            st.session_state["rw_selected_id"] = None
            st.rerun()

    # ── DETAIL BARIS — hanya tampil jika ID ada di halaman ini ──
    if sel_id is not None and sel_id in df_page["id"].values and sel_id in df_f["id"].values:
        row    = df_f[df_f["id"] == sel_id].iloc[0]
        hp_raw = row.get("hp_config", None)

        st.markdown("<br>", unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)

        with dl1:
            st.markdown(f"""
            <div class="metric-card" style="border-left:6px solid #F59E0B;">
                <p style="font-size:1.5rem;font-weight:900;color:#0F172A;margin:0 0 10px 0;">
                    📋 Detail Riwayat ID #{sel_id}
                </p>
                <table style="width:100%;font-size:1.4rem;border-collapse:collapse;">
                    <tr>
                        <td style="color:#64748B;font-weight:700;padding:5px 12px 5px 0;width:50%;">Waktu Eksekusi</td>
                        <td style="font-weight:800;color:#0F172A;">{pd.to_datetime(row['waktu_eksekusi']).strftime('%d %b %Y %H:%M')}</td>
                    </tr>
                    <tr>
                        <td style="color:#64748B;font-weight:700;padding:5px 12px 5px 0;">Kode Saham</td>
                        <td style="font-weight:800;color:#2563EB;">{row['kode_saham']}</td>
                    </tr>
                    <tr>
                        <td style="color:#64748B;font-weight:700;padding:5px 12px 5px 0;">MSE Score</td>
                        <td style="font-weight:800;color:#EF4444;">{row['mse_score']:.2f}</td>
                    </tr>
                    <tr>
                        <td style="color:#64748B;font-weight:700;padding:5px 12px 5px 0;">RMSE Score</td>
                        <td style="font-weight:800;color:#EF4444;">{row['rmse_score']:.2f}</td>
                    </tr>
                    <tr>
                        <td style="color:#64748B;font-weight:700;padding:5px 12px 5px 0;">Tanggal Prediksi</td>
                        <td style="font-weight:800;color:#0F172A;">{row['tanggal_prediksi']}</td>
                    </tr>
                    <tr>
                        <td style="color:#64748B;font-weight:700;padding:5px 12px 5px 0;">Harga Prediksi Final</td>
                        <td style="font-weight:800;color:#10B981;">Rp {row['harga_prediksi_final']:,.0f}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with dl2:
            if hp_raw:
                try:
                    hp_dict = _json.loads(hp_raw)
                    st.markdown(f"""
                    <div class="metric-card" style="border-left:6px solid #2563EB;">
                        <p style="font-size:1.5rem;font-weight:900;color:#0F172A;margin:0 0 10px 0;">
                            ⚙️ Parameter Training
                        </p>
                        <table style="width:100%;font-size:1.4rem;border-collapse:collapse;">
                            <tr>
                                <td style="color:#64748B;font-weight:700;padding:5px 12px 5px 0;width:50%;">Window Size</td>
                                <td style="font-weight:800;color:#0F172A;">{hp_dict.get('window','-')} hari</td>
                            </tr>
                            <tr>
                                <td style="color:#64748B;font-weight:700;padding:5px 12px 5px 0;">Epochs</td>
                                <td style="font-weight:800;color:#0F172A;">{hp_dict.get('epochs','-')}</td>
                            </tr>
                            <tr>
                                <td style="color:#64748B;font-weight:700;padding:5px 12px 5px 0;">LSTM Units</td>
                                <td style="font-weight:800;color:#0F172A;">{hp_dict.get('hidden_size','-')} neuron</td>
                            </tr>
                            <tr>
                                <td style="color:#64748B;font-weight:700;padding:5px 12px 5px 0;">Learning Rate</td>
                                <td style="font-weight:800;color:#0F172A;">{hp_dict.get('lr','-')}</td>
                            </tr>
                            <tr>
                                <td style="color:#64748B;font-weight:700;padding:5px 12px 5px 0;">Batch Size</td>
                                <td style="font-weight:800;color:#0F172A;">{hp_dict.get('batch_size','-')}</td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception:
                    st.info("Parameter tidak tersedia.")
            else:
                st.info("Parameter training tidak tersedia untuk riwayat lama.")

# =========================
# 6) MAIN
# =========================
def main():
    if not st.session_state["logged_in"]:
        halaman_login()
        return

    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Beranda", "Prediksi Saham", "Riwayat"],
            icons=["grid-fill", "graph-up-arrow", "clock-history"],
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#64748B", "font-size": "32px"},
                "nav-link": {
                    "font-family": "Plus Jakarta Sans",
                    "font-size": "24px",
                    "text-align": "left",
                    "margin": "8px",
                    "padding": "18px",
                    "color": "#334155",
                    "border-radius": "12px",
                    "transition": "all 0.2s ease",
                },
                "nav-link-selected": {"background-color": "#DBEAFE", "color": "#1E40AF", "font-weight": "900"},
            },
        )

        st.markdown("---")

        # Emittent selector (UI requirement from advisor)
        pilihan_saham = st.selectbox(
            "Emiten:",
            ["BBRI", "BMRI", "BBNI", "BBTN"],
            index=0
        )

        # Dataset mode
        if pilihan_saham != "BBRI":
            mode_data = "latest_1000"
            st.radio(
                "Mode Data",
                ["Data Real Time"],
                index=0,
                help="Untuk emiten selain BBRI, aplikasi menggunakan 1000 data terakhir dari Yahoo Finance."
            )
        else:
            _mode_label = st.radio(
                "Mode Data",
                ["Data Lokal", "Data Real Time"],
                index=0,
                help="Data Lokal = dataset statis penelitian (BBRI). Data Real Time = 1000 data terakhir dari Yahoo Finance."
            )
            mode_data = "research_first_1000" if _mode_label == "Data Lokal" else "latest_1000"

        st.markdown("---")

        st.markdown("---")

        # Settings row: title + gear popover
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(
                "<h3 style='color:#475569; font-size: 2rem !important; display:flex; align-items:center; gap:10px; margin: 0;'>"
                "<span class='material-symbols-rounded' style='font-size: 2.4rem !important;'>settings</span> "
                "Konfigurasi</h3>",
                unsafe_allow_html=True
            )
        with c2:
            with st.popover("⚙️", help="Atur Hyperparameter LSTM"):
                st.markdown("<h2 style='text-align:center; color:#2563EB;'>🛠️ Setting Model LSTM</h2>", unsafe_allow_html=True)
                st.divider()

                hp = st.session_state["hp"]
                new_epochs = st.number_input("Epochs (Iterasi)", min_value=10, max_value=200, value=int(hp["epochs"]), step=10)
                new_window = st.number_input("Window Size (Hari)", min_value=30, max_value=120, value=int(hp["window"]), step=5)
                new_units = st.select_slider("LSTM Units (Neuron)", options=[8, 16, 32, 64, 128], value=int(hp["hidden_size"]))
                new_lr = st.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=[0.01, 0.001, 0.0001].index(float(hp["lr"])))
                new_bs = st.selectbox("Batch Size", [16, 32, 64], index=[16, 32, 64].index(int(hp["batch_size"])))

                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("💾 Simpan Pengaturan", type="primary", use_container_width=True):
                    st.session_state["hp"] = {
                        "epochs": int(new_epochs),
                        "window": int(new_window),
                        "hidden_size": int(new_units),
                        "lr": float(new_lr),
                        "batch_size": int(new_bs),
                    }
                    st.toast("✅ Pengaturan disimpan! Jalankan **Latih Ulang Model** untuk menerapkan.", icon="💾")

        st.markdown("<br><hr style='margin: 12px 0; border: none; border-top: 2px solid #CBD5E1;'>", unsafe_allow_html=True)
        if st.button("LOGOUT", icon=":material/logout:", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.session_state["role"] = ""
            st.rerun()

    # Reset selected row jika user pindah dari halaman Riwayat
    if st.session_state.get("rw_prev_page") == "Riwayat" and selected != "Riwayat":
        st.session_state["rw_selected_id"] = None
    st.session_state["rw_prev_page"] = selected

    if selected == "Beranda":
        halaman_beranda()
    elif selected == "Prediksi Saham":
        halaman_prediksi(pilihan_saham, mode_data)
    else:
        halaman_riwayat()

if __name__ == "__main__":
    main()
