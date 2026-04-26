import streamlit as st
import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Audio Analyzer Pro", layout="centered")

st.title("🎵 Audio Quality Analyzer")
st.write("Carica il tuo brano (WAV o MP3) per un'analisi tecnica gratuita del mix e mastering.")

uploaded_file = st.file_uploader("Scegli un file audio", type=["wav", "mp3"])

if uploaded_file is not None:
    with st.spinner('Analizzando il brano... attendi...'):
        # 1. Caricamento Audio
        # Leggiamo i dati binari
        audio_bytes = uploaded_file.read()
        data, rate = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=False)
        
        # Convertiamo in formato adatto per pyloudnorm (L, R, samples)
        if len(data.shape) > 1:
            data_pyln = data.T
        else:
            data_pyln = data.reshape(-1, 1)

        # 2. Analisi Loudness (LUFS)
        meter = pyln.Meter(rate) # Standard ITU-R BS.1770-4
        loudness = meter.integrated_loudness(data_pyln)
        
        # 3. Analisi Peak
        true_peak = np.max(np.abs(data))
        true_peak_db = 20 * np.log10(true_peak) if true_peak > 0 else -100

        # 4. Visualizzazione Risultati
        st.subheader("📊 Parametri Rilevati")
        col1, col2 = st.columns(2)
        col1.metric("Loudness (Integrated)", f"{loudness:.2f} LUFS")
        col2.metric("True Peak", f"{true_peak_db:.2f} dB")

        # --- LOGICA DI FEEDBACK ---
        st.divider()
        st.subheader("💡 Feedback e Consigli")
        
        recommendations = []

        # Controllo LUFS (Target Streaming -14)
        if loudness > -10:
            recommendations.append("🔴 **Volume troppo alto:** Il brano è molto compresso. Le piattaforme (Spotify/YT) lo abbasseranno drasticamente. Considera di ridurre il limiter.")
        elif loudness < -16:
            recommendations.append("🟡 **Volume basso:** Il brano potrebbe suonare troppo piano rispetto alla media. Hai spazio per spingere un po' di più il mastering.")
        else:
            recommendations.append("🟢 **Ottimo Volume:** Sei nel range ideale per lo streaming moderno (-14 / -12 LUFS).")

        # Controllo Peak
        if true_peak_db > -0.5:
            recommendations.append("🔴 **Rischio Clipping:** Il picco è troppo vicino allo 0dB. Abbassa l'output del limiter a -1.0 dB per evitare distorsioni dopo la conversione in MP3.")
        else:
            recommendations.append("🟢 **Headroom Corretta:** I picchi sono sotto controllo.")

        for rec in recommendations:
            st.write(rec)

        # 5. Grafico Waveform
        st.subheader("📈 Visualizzazione Onda")
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(data, sr=rate, ax=ax, alpha=0.5)
        ax.set_title("Waveform")
        st.pyplot(fig)

st.info("Nota: Questa app processa i file in memoria. Nessun file viene salvato sui nostri server.")