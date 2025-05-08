import speech_recognition as sr

def recognize_speech():
    # Inisialisasi recognizer
    recognizer = sr.Recognizer()

    # Gunakan mikrofon sebagai sumber suara
    with sr.Microphone() as source:
        print("Silakan bicara...")
        recognizer.adjust_for_ambient_noise(source)  # Sesuaikan dengan kebisingan sekitar
        audio = recognizer.listen(source)

        # print("Mengenali suara...")

        try:
            # Gunakan Google Web Speech API
            text = recognizer.recognize_google(audio, language="id-ID")  # ganti "id-ID" untuk bahasa Indonesia
            # print("Anda mengatakan:", text)
            return text

        except sr.UnknownValueError:
            print("Google Speech tidak bisa mengenali suara")
        except sr.RequestError as e:
            print(f"Tidak bisa meminta hasil dari Google Speech. Error: {e}")

# Contoh penggunaan
if __name__ == "__main__":
    recognize_speech()