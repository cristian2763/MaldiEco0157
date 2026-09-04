import io
import tempfile
import numpy as np
from pyteomics import mzml
from scipy.signal import find_peaks
import streamlit as st
import streamlit_authenticator as stauth

# Configuración de la página
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

authenticator.login()

# =========================================================
# 2. VALIDACIÓN DE SESIÓN Y LÓGICA ORIGINAL
# =========================================================
if st.session_state["authentication_status"] is False:
    st.error("Usuario o contraseña incorrectos.")
elif st.session_state["authentication_status"] is None:
    st.warning("Por favor, ingrese sus credenciales para acceder al sistema.")
elif st.session_state["authentication_status"]:
    authenticator.logout("Cerrar Sesión", "sidebar")
    st.sidebar.write(f"Sesión activa: **{st.session_state['name']}**")
    st.sidebar.divider()

    # --- TUS FUNCIONES INICIALES EXACTAS ---
    def extract_peaks_from_mzml(file_path, min_intensity=1000):
        """
        Lee un archivo .mzml, promedia los espectros (si hay múltiples) 
        y extrae los centroides de los picos.
        """
        all_mz = []
        all_int = []

        with mzml.read(file_path) as reader:
            for spectrum in reader:
                if spectrum.get("ms level", 1) == 1:
                    all_mz.extend(spectrum["m/z array"])
                    all_int.extend(spectrum["intensity array"])

        if not all_mz:
            raise ValueError("No se encontraron espectros MS1 en el archivo.")

        mz_array = np.array(all_mz)
        int_array = np.array(all_int)

        # Ordenar por m/z
        sort_idx = np.argsort(mz_array)
        mz_array = mz_array[sort_idx]
        int_array = int_array[sort_idx]

        # Detectar picos locales
        peaks_indices, _ = find_peaks(int_array, height=min_intensity, distance=5)
        return mz_array[peaks_indices], int_array[peaks_indices]

    def check_peak(mz_peaks, int_peaks, target_mz, tol=10.0, max_intensity=None):
        """
        Verifica si existe un pico en target_mz ± tol (o dentro de un rango [min, max]).
        """
        if isinstance(target_mz, tuple):
            min_mz, max_mz = target_mz[0] - tol, target_mz[1] + tol
        else:
            min_mz, max_mz = target_mz - tol, target_mz + tol

        mask = (mz_peaks >= min_mz) & (mz_peaks <= max_mz)
        matching_mz = mz_peaks[mask]
        matching_int = int_peaks[mask]

        if len(matching_mz) == 0:
            return False, 0.0, None

        # Pico con mayor intensidad dentro del rango
        best_idx = np.argmax(matching_int)
        best_mz = matching_mz[best_idx]
        best_int = matching_int[best_idx]

        if max_intensity is not None and best_int >= max_intensity:
            return False, best_int, best_mz  # Supera el límite para ser considerado "ausente/bajo"

        return True, best_int, best_mz

    def analyze_biomarkers(file_path, tolerance=10.0):
        mz_peaks, int_peaks = extract_peaks_from_mzml(file_path)

        # Lista de los 9 biomarcadores accesorios
        accessory_biomarkers = [3017, 3083, 3595, 3770, 4012, 4939, 5238, 6037, 6169]

        # Evaluaciones individuales
        stec_10k, int_stec_10k, mz_stec_10k = check_peak(mz_peaks, int_peaks, (10163, 10168), tolerance)
        stec_5k, int_stec_5k, mz_stec_5k = check_peak(mz_peaks, int_peaks, (5234, 5238), tolerance)
        
        # Pico 9060: Ausente o < 2.000.000 para STEC; Presente (>= 2.000.000) para Shigella
        p9060_found, p9060_int, p9060_mz = check_peak(mz_peaks, int_peaks, 9060, tolerance)
        stec_p9060_condition = (not p9060_found) or (p9060_int < 2_000_000)

        shigella_10k, int_shig_10k, mz_shig_10k = check_peak(mz_peaks, int_peaks, (10137, 10142), tolerance)
        shigella_5k, int_shig_5k, mz_shig_5k = check_peak(mz_peaks, int_peaks, (5229, 5232), tolerance)

        # Conteo de picos accesorios
        accessory_found = []
        for mz in accessory_biomarkers:
            found, val_int, val_mz = check_peak(mz_peaks, int_peaks, mz, tolerance)
            if found:
                accessory_found.append((mz, val_mz, val_int))

        acc_count = len(accessory_found)

        # Evaluación de criterios
        is_stec = stec_10k and stec_5k and stec_p9060_condition and (acc_count >= 3)
        is_shigella = shigella_10k and shigella_5k and (p9060_found and p9060_int >= 2_000_000) and (acc_count < 2)

        # Mostrar resultados en pantalla
        st.subheader("Reporte de Análisis")
        st.write(f"• **Biomarcador 10163-10168 m/z:** {'DETECTADO' if stec_10k else 'NO'} (m/z: {mz_stec_10k}, int: {int_stec_10k:,.0f})")
        st.write(f"• **Biomarcador 10137-10142 m/z:** {'DETECTADO' if shigella_10k else 'NO'} (m/z: {mz_shig_10k}, int: {int_shig_10k:,.0f})")
        st.write(f"• **Biomarcador 5234-5238 m/z:** {'DETECTADO' if stec_5k else 'NO'} (m/z: {mz_stec_5k}, int: {int_stec_5k:,.0f})")
        st.write(f"• **Biomarcador 5229-5232 m/z:** {'DETECTADO' if shigella_5k else 'NO'} (m/z: {mz_shig_5k}, int: {int_shig_5k:,.0f})")
        st.write(f"• **Pico 9060 m/z:** {'DETECTADO' if p9060_found else 'NO'} (int: {p9060_int:,.0f} | STEC req: <2M)")
        st.write(f"• **Picos accesorios detectados:** {acc_count}/9 ({[f'{m[0]}m/z' for m in accessory_found]})")
        st.divider()

        if is_stec:
            st.success("### DIAGNÓSTICO: POSITIVO para STEC E. coli O157:H7\n**Confirmación requerida:** Identificación stx, antisueros O157/H7 y derivación a LNR.")
        elif is_shigella:
            st.warning("### DIAGNÓSTICO: Shigella spp. u otra E. coli\n**Confirmación requerida:** Identificación Shigella / E. coli patógena no O157.")
        else:
            st.info("### DIAGNÓSTICO: NO CONCLUYENTE / NEGATIVO para los perfiles especificados.")

    # --- INTERFAZ STREAMLIT ---
    st.title("MALDI-TOF STEC & Shigella Diagnostic Tool")
    tolerance_val = st.sidebar.number_input("Tolerancia (Da)", value=10.0, step=0.5)

    uploaded_file = st.file_uploader("Cargar archivo de espectrometría (.mzml)", type=["mzml"])

    if uploaded_file is not None:
        # Guardar en archivo temporal para que pyteomics lo pueda leer por path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mzml") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            analyze_biomarkers(tmp_path, tolerance=tolerance_val)
        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")
