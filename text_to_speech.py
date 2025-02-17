import pyttsx3
from gtts import gTTS
import os
import time

# Initialize pyttsx3 engine (offline TTS)
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speech speed
tts_engine.setProperty('volume', 1.0)  # Volume level

# Supported languages for gTTS
LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "bn": "Bengali",
    "zh": "Chinese",
    "ja": "Japanese",
    "ru": "Russian",
}

def text_to_speech(text, lang="en", use_online=False):
    """Convert text to speech in the chosen language."""
    if use_online and lang in LANGUAGES:  
        try:
            filename = f"speech_{int(time.time())}.mp3"  # Unique filename
            tts = gTTS(text=text, lang=lang)
            tts.save(filename)  # Save file

            # Play file based on OS
            if os.name == "nt":  # Windows
                os.system(f"start {filename}")
            else:  # Linux/macOS
                os.system(f"mpg321 {filename} || afplay {filename} || play {filename}")

            # Remove file after playing
            time.sleep(2)
            os.remove(filename)

        except Exception as e:
            print(f"Error using gTTS: {e}. Falling back to pyttsx3.")
            tts_engine.say(text)
            tts_engine.runAndWait()
    else:
        # Use offline pyttsx3
        tts_engine.say(text)
        tts_engine.runAndWait()

if __name__ == "__main__":
    print("Select a language:")
    for code, lang in LANGUAGES.items():
        print(f"{code}: {lang}")

    lang_choice = input("Enter language code (default: en): ").strip().lower()
    lang_choice = lang_choice if lang_choice in LANGUAGES else "en"

    print(f"Selected Language: {LANGUAGES[lang_choice]}")

    print("\nEnter text (type 'exit' to stop):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        text_to_speech(user_input, lang=lang_choice, use_online=True)
