import io
import os
import zipfile
import re
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import plotly.graph_objects as go
from scipy.signal import find_peaks
from pyteomics import mzml

# ---------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ---------------------------------------------------------
st.set_page_config(
    page_title="MALDI-TOF STEC & Shigella Diagnostic Tool",
    page_icon="🔬",
    layout="wide"
)

# ---------------------------------------------------------
# AUTENTICACIÓN Y CONTROL DE ACCESO
# ---------------------------------------------------------
credentials = st.secrets["credentials"].to_dict()
cookie_config = st.secrets["cookie"]

authenticator = stauth.Authenticate(
    credentials,
    cookie_config["name"],
    cookie_config["key"],
    cookie_config["expiry_days"]
)

authenticator.login()

if st.session_state["authentication_status"] is False:
    st.error("Usuario o contraseña incorrectos.")
elif st.session_state["authentication_status"] is None:
    st.warning("Por favor, ingrese sus credenciales para acceder al sistema.")
elif st.session_state["authentication_status"]:
    # Barra lateral de sesión
    authenticator.logout("Cerrar Sesión", "sidebar")
    st.sidebar.write(f"Sesión activa: **{st.session_state['name']}**")
    st.sidebar.divider()

    ACCESSORY_BIOMARKERS = [3017, 3083, 3595, 3770, 4012, 4939, 5238, 6037, 6169]

    # ---------------------------------------------------------
    # PREPROCESAMIENTO Y PARSEO MULTIFORMATO
    # ---------------------------------------------------------
    def snip_baseline(y: np.ndarray, iterations: int = 35) -> np.ndarray:
        """Algoritmo SNIP para remoción de fondo químico en MALDI-TOF."""
        spectrum = np.array(y, dtype=np.float64)
        baseline = np.copy(spectrum)
        n = len(spectrum)
        for p in range(1, iterations + 1):
            for i in range(p, n - p):
                temp = (baseline[i - p] + baseline[i + p]) / 2.0
                if temp < baseline[i]:
                    baseline[i] = temp
        return baseline

    def parse_bruker_zip(zip_bytes: bytes):
        """Extrae y calibra espectros binarios Bruker FID/acqus dentro de un ZIP."""
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            fid_files = [f for f in z.namelist() if f.endswith('fid')]
            acqus_files = [f for f in z.namelist() if f.endswith('acqus') or f.endswith('acqu')]
            
            if not fid_files or not acqus_files:
                raise ValueError("El archivo ZIP no contiene una estructura válida de Bruker (fid / acqus).")
            
            acqus_content = z.read(acqus_files[0]).decode('utf-8', errors='ignore')
            
            def get_param(name, default):
                match = re.search(rf'##\${name}=\s*([^\r\n]+)', acqus_content)
                return float(match.group(1)) if match else default

            ml1 = get_param('ML1', 0.0)
            ml2 = get_param('ML2', 0.0)
            ml3 = get_param('ML3', 0.0)
            dw = get_param('DW', 0.5)
            delay = get_param('DELAY', 0.0)

            fid_raw = z.read(fid_files[0])
            intensities = np.frombuffer(fid_raw, dtype='<i4').astype(np.float64)
            
            n_points = len(intensities)
            time_axis = delay + np.arange(n_points) * (dw * 1e-9)
            
            if ml1 != 0 and ml2 != 0:
                mz = ((np.sqrt(ml1**2 + 4 * ml2 * (time_axis * 1e6 - ml3)) - ml1) / (2 * ml2)) ** 2
            else:
                mz = np.linspace(2000, 20000, n_points)

        return mz, intensities

    def load_spectrum_data(uploaded_file):
        """Parsea .mzML, .txt, .csv, .tsv o .zip de Bruker."""
        fname = uploaded_file.name.lower()
        
        if fname.endswith(".mzml"):
            with mzml.read(uploaded_file) as reader:
                for spectrum in reader:
                    if spectrum.get("ms level", 1) == 1:
                        return np.array(spectrum["m/z array"], dtype=np.float64), np.array(spectrum["intensity array"], dtype=np.float64)
            raise ValueError("No se encontraron espectros MS1 en el archivo .mzML.")
            
        elif fname.endswith(".zip"):
            return parse_bruker_zip(uploaded_file.getvalue())
            
        elif fname.endswith((".txt", ".csv", ".tsv")):
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            df = pd.read_csv(io.StringIO(content), sep=None, engine='python', header=None)
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            if df.shape[1] < 2:
                raise ValueError("El archivo tabular requiere al menos 2 columnas [m/z, Intensidad].")
            return df.iloc[:, 0].values.astype(np.float64), df.iloc[:, 1].values.astype(np.float64)
        else:
            raise ValueError(f"Formato no compatible: {fname}")

    def preprocess_spectrum(mz, intensity, min_mz=2000, max_mz=12000, snip_iters=35, prominence=1000):
        sort_idx = np.argsort(mz)
        mz, intensity = mz[sort_idx], intensity[sort_idx]

        mask = (mz >= min_mz) & (mz <= max_mz)
        mz_win, int_win = mz[mask], intensity[mask]

        if len(mz_win) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        baseline = snip_baseline(int_win, iterations=snip_iters)
        corrected_int = np.maximum(0, int_win - baseline)
        peak_idx, _ = find_peaks(corrected_int, distance=8, prominence=prominence)
        
        return mz_win, int_win, corrected_int, peak_idx

    # ---------------------------------------------------------
    # EVALUACIÓN DE CRITERIOS CLÍNICOS
    # ---------------------------------------------------------
    def check_biomarker(mz_peaks, int_peaks, target_mz, tol=10.0):
        if isinstance(target_mz, tuple):
            min_mz, max_mz = target_mz[0] - tol, target_mz[1] + tol
            label = f"{target_mz[0]}-{target_mz[1]} m/z"
        else:
            min_mz, max_mz = target_mz - tol, target_mz + tol
            label = f"{target_mz} m/z"

        mask = (mz_peaks >= min_mz) & (mz_peaks <= max_mz)
        matching_mz = mz_peaks[mask]
        matching_int = int_peaks[mask]

        if len(matching_mz) == 0:
            return {"label": label, "target": target_mz, "detected": False, "obs_mz": None, "intensity": 0.0}

        best_i = np.argmax(matching_int)
        return {"label": label, "target": target_mz, "detected": True, "obs_mz": matching_mz[best_i], "intensity": matching_int[best_i]}

    def classify_sample(mz_peaks, int_peaks, tol=10.0, threshold_9060=2_000_000):
        res_stec_10k = check_biomarker(mz_peaks, int_peaks, (10163, 10168), tol)
        res_stec_5k = check_biomarker(mz_peaks, int_peaks, (5234, 5238), tol)
        res_p9060 = check_biomarker(mz_peaks, int_peaks, 9060, tol)
        
        res_shig_10k = check_biomarker(mz_peaks, int_peaks, (10137, 10142), tol)
        res_shig_5k = check_biomarker(mz_peaks, int_peaks, (5229, 5232), tol)

        acc_results = [check_biomarker(mz_peaks, int_peaks, mz, tol) for mz in ACCESSORY_BIOMARKERS]
        detected_acc = [r for r in acc_results if r["detected"]]
        n_acc = len(detected_acc)

        stec_9060_ok = (not res_p9060["detected"]) or (res_p9060["intensity"] < threshold_9060)
        shig_9060_ok = res_p9060["detected"] and (res_p9060["intensity"] >= threshold_9060)

        is_stec = res_stec_10k["detected"] and res_stec_5k["detected"] and stec_9060_ok and (n_acc >= 3)
        is_shigella = res_shig_10k["detected"] and res_shig_5k["detected"] and shig_9060_ok and (n_acc < 2)

        if is_stec:
            verdict = "STEC E. coli O157:H7 (POSITIVO)"
            status_type = "danger"
            action = "Identificación de stx y serotipificación con antisueros anti-O157 y anti-H7. Derivación de muestra fecal y aislamiento al LNR."
        elif is_shigella:
            verdict = "Shigella spp. u otra E. coli"
            status_type = "warning"
            action = "Identificación de Shigella spp. u otra E. coli: STEC no O157, EPEC, ETEC, EIEC, EAEC, E. coli sin factores de virulencia. Derivación al LNR."
        else:
            verdict = "NO CONCLUYENTE / NEGATIVO"
            status_type = "info"
            action = "Verificar calidad de espectro o contrastar con otro ensayo (p. ej. extracción con ácido fórmico / PCR)."

        return {
            "verdict": verdict, "status_type": status_type, "action": action, "n_acc": n_acc,
            "res_stec_10k": res_stec_10k, "res_stec_5k": res_stec_5k,
            "res_shig_10k": res_shig_10k, "res_shig_5k": res_shig_5k,
            "res_p9060": res_p9060, "acc_results": acc_results
        }

    # ---------------------------------------------------------
    # VISTA STREAMLIT
    # ---------------------------------------------------------
    st.title("🔬 MALDI-TOF STEC O157:H7")

    st.sidebar.header("⚙️ Parámetros de Ajuste")
    tolerance = st.sidebar.slider("Tolerancia (m/z ± Da)", min_value=1.0, max_value=20.0, value=10.0, step=0.5)
    threshold_9060 = st.sidebar.number_input("Corte Pico 9060 m/z (Intensidad)", value=2_000_000, step=100_000, format="%d")
    prominence_val = st.sidebar.slider("Prominencia de Picos", min_value=100, max_value=10000, value=800, step=100)
    snip_iterations = st.sidebar.slider("Iteraciones SNIP (Línea Base)", min_value=10, max_value=60, value=35, step=5)

    uploaded_files = st.file_uploader(
        "Cargar espectros (.mzML, .txt, .csv, .tsv, .zip Bruker)",
        type=["mzml", "txt", "csv", "tsv", "zip"],
        accept_multiple_files=True
    )

    if uploaded_files:
        summary_rows = []
        processed = {}

        for file in uploaded_files:
            try:
                raw_mz, raw_int = load_spectrum_data(file)
                mz_win, int_win, corr_int, p_idx = preprocess_spectrum(
                    raw_mz, raw_int, snip_iters=snip_iterations, prominence=prominence_val
                )
                mz_peaks, int_peaks = mz_win[p_idx], corr_int[p_idx]
                eval_res = classify_sample(mz_peaks, int_peaks, tol=tolerance, threshold_9060=threshold_9060)

                processed[file.name] = {
                    "mz_win": mz_win, "int_win": int_win, "corr_int": corr_int,
                    "mz_peaks": mz_peaks, "int_peaks": int_peaks, "eval": eval_res
                }

                summary_rows.append({
                    "Archivo": file.name,
                    "Diagnóstico": eval_res["verdict"],
                    "Accesorios": f"{eval_res['n_acc']}/9",
                    "10163-10168 (STEC)": "SÍ" if eval_res["res_stec_10k"]["detected"] else "NO",
                    "10137-10142 (Shig)": "SÍ" if eval_res["res_shig_10k"]["detected"] else "NO",
                    "5234-5238 (STEC)": "SÍ" if eval_res["res_stec_5k"]["detected"] else "NO",
                    "5229-5232 (Shig)": "SÍ" if eval_res["res_shig_5k"]["detected"] else "NO",
                    "9060 m/z Int": f"{eval_res['res_p9060']['intensity']:,.0f}" if eval_res['res_p9060']['detected'] else "0",
                    "Acción Clínica": eval_res["action"]
                })
            except Exception as err:
                st.error(f"Error procesando '{file.name}': {str(err)}")

        if processed:
            df_summary = pd.DataFrame(summary_rows)

            # Muestra individual
            sel_sample = st.selectbox("Seleccionar muestra para análisis detallado:", list(processed.keys()))
            s = processed[sel_sample]
            ev = s["eval"]

            if ev["status_type"] == "danger":
                st.error(f"### 🛑 Diagnóstico: {ev['verdict']}")
            elif ev["status_type"] == "warning":
                st.warning(f"### ⚠️ Diagnóstico: {ev['verdict']}")
            else:
                st.info(f"### ℹ️ Diagnóstico: {ev['verdict']}")

            st.markdown(f"**Acción recomendada:** {ev['action']}")

            # Gráfico interactivo
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s["mz_win"], y=s["int_win"], mode='lines', name='Crudo', line=dict(color='#d3d3d3', width=1)))
            fig.add_trace(go.Scatter(x=s["mz_win"], y=s["corr_int"], mode='lines', name='SNIP Corregido', line=dict(color='#1f77b4', width=1.3)))
            fig.add_trace(go.Scatter(x=s["mz_peaks"], y=s["int_peaks"], mode='markers', name='Picos', marker=dict(color='#d62728', size=6, symbol='x')))

            # Regiones diagnósticas coloreadas según detección
            reg_defs = [
                ((10163, 10168), "STEC 10k", ev["res_stec_10k"]["detected"]),
                ((10137, 10142), "Shigella 10k", ev["res_shig_10k"]["detected"]),
                ((5234, 5238), "STEC 5.2k", ev["res_stec_5k"]["detected"]),
                ((5229, 5232), "Shigella 5.2k", ev["res_shig_5k"]["detected"]),
                (9060, "9060 m/z", ev["res_p9060"]["detected"])
            ]
            for target, label, is_det in reg_defs:
                x0 = (target[0] if isinstance(target, tuple) else target) - tolerance
                x1 = (target[1] if isinstance(target, tuple) else target) + tolerance
                col = "rgba(44, 160, 44, 0.2)" if is_det else "rgba(214, 39, 40, 0.15)"
                fig.add_vrect(x0=x0, x1=x1, fillcolor=col, line_width=0, annotation_text=label, annotation_position="top left")

            fig.update_layout(
                title=f"Perfil Espectral MALDI-TOF: {sel_sample}",
                xaxis_title="m/z", yaxis_title="Intensidad",
                height=460, hovermode="x unified", margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Tablas desglosadas
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Biomarcadores Principales")
                st.dataframe(pd.DataFrame([
                    {"Marcador": "STEC [10163-10168]", "Estado": "✅ Detectado" if ev["res_stec_10k"]["detected"] else "❌ Ausente", "m/z Obs": f"{ev['res_stec_10k']['obs_mz']:.1f}" if ev['res_stec_10k']['obs_mz'] else "-", "Intensidad": f"{ev['res_stec_10k']['intensity']:,.0f}"},
                    {"Marcador": "Shigella [10137-10142]", "Estado": "✅ Detectado" if ev["res_shig_10k"]["detected"] else "❌ Ausente", "m/z Obs": f"{ev['res_shig_10k']['obs_mz']:.1f}" if ev['res_shig_10k']['obs_mz'] else "-", "Intensidad": f"{ev['res_shig_10k']['intensity']:,.0f}"},
                    {"Marcador": "STEC [5234-5238]", "Estado": "✅ Detectado" if ev["res_stec_5k"]["detected"] else "❌ Ausente", "m/z Obs": f"{ev['res_stec_5k']['obs_mz']:.1f}" if ev['res_stec_5k']['obs_mz'] else "-", "Intensidad": f"{ev['res_stec_5k']['intensity']:,.0f}"},
                    {"Marcador": "Shigella [5229-5232]", "Estado": "✅ Detectado" if ev["res_shig_5k"]["detected"] else "❌ Ausente", "m/z Obs": f"{ev['res_shig_5k']['obs_mz']:.1f}" if ev['res_shig_5k']['obs_mz'] else "-", "Intensidad": f"{ev['res_shig_5k']['intensity']:,.0f}"},
                    {"Marcador": "Pico 9060 m/z", "Estado": f"{'✅ < 2M' if ev['res_p9060']['intensity'] < threshold_9060 else '⚠️ >= 2M'}", "m/z Obs": f"{ev['res_p9060']['obs_mz']:.1f}" if ev['res_p9060']['obs_mz'] else "-", "Intensidad": f"{ev['res_p9060']['intensity']:,.0f}"}
                ]), use_container_width=True, hide_index=True)

            with c2:
                st.subheader(f"Panel Accesorio ({ev['n_acc']}/9 picos)")
                acc_list = [{"Pico": item["label"], "Detectado": "✅ Sí" if item["detected"] else "❌ No", "m/z Obs": f"{item['obs_mz']:.1f}" if item["detected"] else "-", "Intensidad": f"{item['intensity']:,.0f}" if item["detected"] else "-"} for item in ev["acc_results"]]
                st.dataframe(pd.DataFrame(acc_list), use_container_width=True, hide_index=True)

            # Exportación
            st.divider()
            st.subheader("📋 Consolidado de Muestras")
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
            st.download_button("📥 Descargar Reporte Completo (CSV)", df_summary.to_csv(index=False).encode('utf-8'), "reporte_maldi.csv", "text/csv")
