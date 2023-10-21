import http.client, urllib

from gtts import gTTS
from io import BytesIO
import pygame

def send_message(message):

    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",
      urllib.parse.urlencode({
        "token": "apopaey5179h9aepy3smpjguipb9b4",
        "user": "ugp9dvyuib13zb1u2x1sh7dj29jgvw",
        "message": message,
      }), { "Content-type": "application/x-www-form-urlencoded" })
    
    conn.getresponse()

def speak(text, language='en',speed = 1.0):
    mp3_fo = BytesIO()
    tts = gTTS(text, lang=language)
    tts.write_to_fp(mp3_fo)
    mp3_fo.seek(0)
    return mp3_fo

def voice_message(message,speed=1.0):
    pygame.init()
    pygame.mixer.init()
    # sound.seek(0)
    sound = speak(message,speed = speed)
    audio_data = pygame.mixer.Sound(sound)
    audio_length = audio_data.get_length()
    audio_data.play()
    playback_time = audio_length/speed
    pygame.time.wait(int(playback_time * 1000))


def send_all_message(message):
    send_message(message)
    # voice_message(message)
