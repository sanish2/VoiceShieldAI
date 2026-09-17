import os
import subprocess
import tempfile
import time

import numpy as np
import soundfile as sf
import streamlit as st

from transformers import pipeline


# =========================================================
# SETTINGS
# =========================================================

MODEL_NAME = "Shanmugapriya6/voice-fake-detector-v1"


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="VoiceShield AI",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ VoiceShield AI")

st.subheader(
    "AI Voice & Deepfake Detection"
)

st.write(
    "Upload a voice recording and VoiceShield AI "
    "will analyze it for signs of AI-generated speech."
)

st.info(
    "For best results, use a clear recording with one speaker "
    "and minimal background noise."
)


# =========================================================
# LOAD AI DETECTOR
# =========================================================

@st.cache_resource
def load_detector():

    detector = pipeline(
        "audio-classification",
        model=MODEL_NAME
    )

    return detector


# =========================================================
# PRELOAD DETECTOR
# =========================================================

startup_status = st.empty()

startup_status.info(
    "🤖 Preparing VoiceShield AI..."
)

detector = load_detector()

startup_status.success(
    "✅ VoiceShield AI is ready"
)


# =========================================================
# CONVERT AUDIO
# =========================================================

def convert_to_wav(uploaded_file):

    extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    with tempfile.NamedTemporaryFile(
        suffix=extension,
        delete=False
    ) as input_file:

        input_file.write(
            uploaded_file.getvalue()
        )

        input_path = input_file.name

    wav_path = input_path + ".wav"

    try:

        subprocess.run(
            [
                "afconvert",
                input_path,
                wav_path,
                "-f",
                "WAVE",
                "-d",
                "LEI16@16000"
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        audio, sample_rate = sf.read(
            wav_path
        )

        # Stereo -> mono
        if audio.ndim > 1:

            audio = np.mean(
                audio,
                axis=1
            )

        audio = audio.astype(
            np.float32
        )

        return audio, sample_rate

    finally:

        if os.path.exists(input_path):
            os.remove(input_path)

        if os.path.exists(wav_path):
            os.remove(wav_path)


# =========================================================
# DETERMINE RESULT
# =========================================================

def get_result(scores):

    label_scores = {
        item["label"]: float(item["score"])
        for item in scores
    }

    # Verified using your recordings:
    # LABEL_0 = REAL
    # LABEL_1 = FAKE

    real_score = label_scores.get(
        "LABEL_0",
        0.0
    )

    fake_score = label_scores.get(
        "LABEL_1",
        0.0
    )

    if fake_score > real_score:

        result = "FAKE"
        confidence = fake_score * 100

    else:

        result = "REAL"
        confidence = real_score * 100

    return (
        result,
        confidence,
        real_score * 100,
        fake_score * 100
    )


# =========================================================
# FILE UPLOAD
# =========================================================

audio_file = st.file_uploader(
    "Upload Voice Recording",
    type=[
        "wav",
        "mp3",
        "m4a"
    ]
)


# =========================================================
# AFTER FILE UPLOAD
# =========================================================

if audio_file is not None:

    st.success(
        "Voice recording uploaded successfully."
    )

    st.audio(
        audio_file
    )

    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    if st.button(
        "🔍 Analyze Voice",
        use_container_width=True
    ):

        try:

            # =================================================
            # PROCESSING DISPLAY
            # =================================================

            st.divider()

            st.subheader(
                "🔄 VoiceShield AI Processing"
            )

            step1 = st.empty()
            step2 = st.empty()
            step3 = st.empty()
            step4 = st.empty()
            step5 = st.empty()

            # =================================================
            # STEP 1 - PREPARE AUDIO
            # =================================================

            step1.write(
                "⏳ Preparing your audio..."
            )

            audio, sample_rate = convert_to_wav(
                audio_file
            )

            duration = (
                len(audio)
                / sample_rate
            )

            step1.write(
                "✅ Audio prepared"
            )

            # =================================================
            # STEP 2 - DETECTOR READY
            # =================================================

            step2.write(
                "⏳ Checking AI voice detector..."
            )

            # Detector was already loaded at startup.
            time.sleep(0.1)

            step2.write(
                "✅ AI voice detector ready"
            )

            # =================================================
            # STEP 3 - ANALYZE COMPLETE RECORDING
            # =================================================

            step3.write(
                "⏳ Analyzing the complete recording..."
            )

            scores = detector(
                {
                    "raw": audio,
                    "sampling_rate": sample_rate
                }
            )

            step3.write(
                "✅ Complete recording analyzed"
            )

            # =================================================
            # STEP 4 - CALCULATE RESULT
            # =================================================

            step4.write(
                "⏳ Calculating detection result..."
            )

            (
                result,
                confidence,
                real_probability,
                fake_probability
            ) = get_result(
                scores
            )

            step4.write(
                "✅ Detection result calculated"
            )

            # =================================================
            # STEP 5 - FINAL REPORT
            # =================================================

            step5.write(
                "⏳ Preparing final report..."
            )

            time.sleep(0.1)

            step5.write(
                "✅ Final report ready"
            )

            # =================================================
            # FINAL REPORT
            # =================================================

            st.divider()

            st.header(
                "🛡️ Voice Analysis Report"
            )

            # =================================================
            # FINAL RESULT
            # =================================================

            if result == "FAKE":

                st.error(
                    "🔴 AI-GENERATED / FAKE VOICE DETECTED"
                )

            else:

                st.success(
                    "🟢 REAL HUMAN VOICE DETECTED"
                )

            # =================================================
            # CONFIDENCE
            # =================================================

            st.metric(
                "Detection Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                min(
                    int(confidence),
                    100
                )
            )

            # =================================================
            # PROBABILITIES
            # =================================================

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "🟢 Real Probability",
                    f"{real_probability:.2f}%"
                )

            with col2:

                st.metric(
                    "🔴 AI/Fake Probability",
                    f"{fake_probability:.2f}%"
                )

            # =================================================
            # RECORDING INFORMATION
            # =================================================

            st.divider()

            st.subheader(
                "🎧 Recording Information"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.write(
                    "**Duration**"
                )

                st.write(
                    f"{duration:.2f} sec"
                )

            with col2:

                st.write(
                    "**Sample Rate**"
                )

                st.write(
                    f"{sample_rate} Hz"
                )

            with col3:

                st.write(
                    "**File Size**"
                )

                st.write(
                    f"{audio_file.size / 1024:.1f} KB"
                )

            # =================================================
            # INFORMATION
            # =================================================

            st.caption(
                "The complete recording is analyzed using "
                "the official VoiceShield AI audio-classification "
                "pipeline."
            )

            # =================================================
            # DISCLAIMER
            # =================================================

            st.warning(
                "⚠️ VoiceShield AI is an experimental prototype. "
                "The displayed percentage is the model's prediction "
                "and is not absolute proof of authenticity."
            )

        except Exception as error:

            st.error(
                f"Analysis failed: {error}"
            )