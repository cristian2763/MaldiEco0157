import io
import re
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks
import streamlit as st
import streamlit_authenticator as stauth
from pyteomics import mzml

# Configuración de página
st.set_page_config(page_title="MALDI-TOF STEC & Shigella", layout="wide")

# =========================================================
# 1. AUTENTICACIÓN Y SEGURIDAD
# =========================================================
credentials = st.secrets["credentials"].to_dict()
cookie_config = st.secrets["cookie"]

authenticator = stauth.Authenticate(
    credentials,
    cookie_config["name"],
    cookie_config["key"],
    cookie_config["expiry_days"]
)

# Renderizar formulario de login
authenticator.login()

# =========================================================
# 2. VALIDACIÓN DE SESIÓN Y LÓGICA DE LA APLICACIÓN
# =========================================================
if st.session_state["authentication_status"] is False:
    st.error("Usuario o contraseña incorrectos.")
elif st.session_state["authentication_status"] is None:
    st.warning("Por favor, ingrese sus credenciales para acceder al sistema.")
elif st.session_state["authentication_status"]:
    # Botón de cerrar sesión en la barra lateral
    authenticator.logout("Cerrar Sesión", "sidebar")
    st.sidebar.write(f"Sesión activa: **{st.session_state['name']}**")
    st.sidebar.divider()

    # --- FUNCIONES DE PARSING Y ANÁLISIS ---
    def parse_bruker_zip(zip_bytes: bytes):
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            fid_files = [f for f in z.namelist() if f.endswith('fid')]
            acqus_files = [f for f in z.namelist() if f.endswith('acqus') or f.endswith('acqu')]
            
            if not fid_files or not acqus_files:
                raise ValueError("El archivo ZIP no contiene una estructura válida de Bruker (fid / acqus).")
            
            acqus_content = z.read(acqus_files[0]).decode('utf-8', errors='ignore')
            
            def get_param(name, default):
                pattern = rf'##\${name}=\s*([^\r\n]+)'
                match = re.search(pattern, acqus_content)
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
        fname = uploaded_file.name.lower()
        if fname.endswith(".mzml"):
            with mzml.read(uploaded_file, use_index=False) as reader:
                for spectrum in reader:
                    if spectrum.get("ms level", 1) == 1:
                        return np.array(spectrum["m/z array"], dtype=np.float64), np.array(spectrum["intensity array"], dtype=np.float64)
            raise ValueError("No se encontraron espectros MS1 en el archivo .mzML.")
        elif fname.endswith(".zip"):
            return parse_bruker_zip(uploaded_file.getvalue())
        elif fname.endswith((".txt", ".csv", ".tsv")):
            sep = "," if fname.endswith(".csv") else ("\t" if fname.endswith(".tsv") else None)
            df = pd.read_csv(uploaded_file, sep=sep, comment="#", header=None)
            df = df.dropna().astype(float)
            return df.iloc[:, 0].values, df.iloc[:, 1].values
        else:
            raise ValueError("Formato de archivo no compatible.")

    # --- INTERFAZ PRINCIPAL DE DIAGNÓSTICO ---
    st.title("MALDI-TOF STEC & Shigella Diagnostic Tool")
    st.write("Cargue un espectro para clasificar los biomarcadores específicos.")

    uploaded_file = st.file_uploader(
        "Cargar archivo de espectrometría (.mzML, .zip Bruker, .txt, .csv, .tsv)",
        type=["mzml", "zip", "txt", "csv", "tsv"]
    )

    if uploaded_file is not None:
        try:
            mz_raw, intensity_raw = load_spectrum_data(uploaded_file)
            
            # Recorte a rango de interés diagnóstico (2000 - 20000 Da)
            mask = (mz_raw >= 2000) & (mz_raw <= 20000)
            mz = mz_raw[mask]
            intensity = intensity_raw[mask]

            # Detección de picos
            peaks, properties = find_peaks(intensity, height=np.max(intensity) * 0.05, distance=10)
            peak_mzs = mz[peaks]
            peak_intensities = intensity[peaks]

            # Lógica diagnóstica
            # 1. Biomarcador STEC (9060 m/z con corte de intensidad < 2M)
            stec_peak_mask = (peak_mzs >= 9050) & (peak_mzs <= 9070)
            stec_intensity_flag = False
            if np.any(stec_peak_mask):
                max_stec_int = np.max(peak_intensities[stec_peak_mask])
                if max_stec_int < 2000000:
                    stec_intensity_flag = True

            # 2. Picos accesorios (rango de detección)
            accessory_targets = [2113, 3030, 4226, 4531, 5094, 9535]
            accessory_count = 0
            for target in accessory_targets:
                if np.any((peak_mzs >= target - 10) & (peak_mzs <= target + 10)):
                    accessory_count += 1

            # Clasificación final
            if stec_intensity_flag and accessory_count >= 3:
                diag_result = "Compatible con STEC (Shiga toxin-producing E. coli)"
                alert_type = st.success
            elif accessory_count < 2:
                diag_result = "Compatible con Shigella spp."
                alert_type = st.warning
            else:
                diag_result = "Perfil no concluyente / E. coli comensal"
                alert_type = st.info

            alert_type(f"**Resultado:** {diag_result}")
            st.metric("Picos accesorios detectados", f"{accessory_count} / {len(accessory_targets)}")

            # Gráfico interactivo
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mz, y=intensity, mode='lines', name='Espectro'))
            fig.add_trace(go.Scatter(x=peak_mzs, y=peak_intensities, mode='markers', name='Picos', marker=dict(color='red', size=6)))
            fig.update_layout(
                title=f"Espectro: {uploaded_file.name}",
                xaxis_title="m/z (Da)",
                yaxis_title="Intensidad Absoluta",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")
