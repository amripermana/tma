import tempfile
import pygame
from gtts import gTTS
import time
import os

def tts_indonesia(text):
    tts = gTTS(text=text, lang='id')

    # Buat file sementara .wav
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as fp:
        temp_filename = fp.name  # Simpan nama file

    tts.save(temp_filename)  # Simpan file TTS sebagai WAV

    try:
        # Inisialisasi pygame
        pygame.mixer.init()
        pygame.mixer.music.load(temp_filename)
        # print("Memainkan suara...")
        pygame.mixer.music.play()

        # Tunggu sampai selesai diputar
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    finally:
        pygame.mixer.quit()  # Pastikan mixer dimatikan
        try:
            os.remove(temp_filename)  # Hapus file sementara
        except:
            print(f"[INFO] File sementara tidak bisa dihapus: {temp_filename}")

# Contoh penggunaan
# if __name__ == "__main__":
#     text = paracetamol"
#     tts_indonesia("Halo, ini adalah uji coba text to speech tanpa error.")