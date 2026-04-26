import streamlit as st
import librosa
import numpy as np
import pyloudnorm as pyln
import matplotlib.pyplot as plt
import io

# Page Configuration
st.set_page_config(page_title="Audio Analyzer Plus", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎵 Audio Analyzer Plus")
st.write("Professional mix and mastering diagnostic tool. Upload your track to get an instant technical report.")

uploaded_file = st.file_uploader("Upload your track (WAV or MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    with st.spinner('Analyzing audio signal... please wait.'):
        # 1. Load Audio (Stereo is mandatory for correlation)
        audio_bytes = uploaded_file.read()
        data, rate = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=False)
        
        # Ensure data is in (Samples, Channels) format for analysis
        if data.ndim == 1:
            st.warning("⚠️ The uploaded file is Mono. Stereo Correlation analysis cannot be performed.")
            is_stereo = False
            data_stereo = data.reshape(-1, 1)
        else:
            is_stereo = True
            data_stereo = data.T # Transpose to (Samples, 2)

        # 2. Loudness Analysis (LUFS & LRA)
        meter = pyln.Meter(rate)
        integrated_loudness = meter.integrated_loudness(data_stereo)
        lra = meter.loudness_range(data_stereo)

        # 3. Peak and Crest Factor
        peak_linear = np.max(np.abs(data))
        true_peak_db = 20 * np.log10(peak_linear) if peak_linear > 0 else -100
        
        rms_level = np.sqrt(np.mean(data**2))
        crest_factor = peak_linear / rms_level if rms_level > 0 else 0

        # 4. Phase Correlation (only for stereo)
        correlation = 0
        if is_stereo:
            # Calculate Pearson correlation coefficient between L and R
            correlation = np.corrcoef(data[0], data[1])[0, 1]

        # --- UI DISPLAY ---
        st.subheader("📊 Quantitative Analysis")
        
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Loudness", f"{integrated_loudness:.1f} LUFS")
        m_col2.metric("True Peak", f"{true_peak_db:.2f} dBTP")
        m_col3.metric("LU Range (LRA)", f"{lra:.1f} LU")
        m_col4.metric("Crest Factor", f"{crest_factor:.2f}")
        m_col5.metric("Correlation", f"{correlation:.2f}")

        st.divider()

        # --- TECHNICAL FEEDBACK ---
        st.subheader("💡 Technical Feedback & Improvements")
        
        col_fb1, col_fb2 = st.columns(2)

        with col_fb1:
            st.markdown("### 🎚️ Dynamic & Level")
            # Loudness Logic
            if integrated_loudness > -10:
                st.error("**Oversquashed:** Your track is very loud. It will be turned down by streaming platforms. Try to ease up on the Limiter.")
            elif integrated_loudness < -15:
                st.warning("**Low Level:** The track might sound quiet. You have headroom to increase the loudness for a more competitive master.")
            else:
                st.success("**Target Reached:** Your loudness is ideal for modern streaming standards.")

            # Peak Logic
            if true_peak_db > -0.5:
                st.error("**Clipping Risk:** Peaks are too close to 0dB. Lower your Limiter's Ceiling to -1.0 dBTP to avoid inter-sample peaks.")
            
            # LRA Logic
            if lra < 4:
                st.info("**Small LRA:** Very consistent volume, typical of heavy EDM/Pop. If this is Jazz/Rock, you might be over-compressing.")

        with col_fb2:
            st.markdown("### 🧬 Stereo & Phase")
            if is_stereo:
                if correlation < 0:
                    st.error("**Phase Issues:** Negative correlation detected. Your track will lose significant elements (like bass or vocals) when played in Mono.")
                elif correlation < 0.4:
                    st.warning("**Wide/Thin:** Low correlation. The mix is very wide, but check for mono compatibility.")
                else:
                    st.success("**Solid Phase:** Good correlation. The track will sound great even on mono speakers (phones, clubs).")
            else:
                st.info("Mono file: No phase correlation available.")

        # --- VISUALIZATION ---
        st.subheader("📈 Waveform Visualization")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        
        if is_stereo:
            librosa.display.waveshow(data[0], sr=rate, ax=ax, alpha=0.5, label='Left', color='#00d1ff')
            librosa.display.waveshow(data[1], sr=rate, ax=ax, alpha=0.5, label='Right', color='#ff007c')
        else:
            librosa.display.waveshow(data, sr=rate, ax=ax, color='#00d1ff')
        
        ax.legend()
        ax.tick_params(colors='white')
        st.pyplot(fig)

st.caption("Audio Analyzer Plus | Developed for Producers & Engineers | 100% Private Analysis")
