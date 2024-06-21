from pydub import AudioSegment
import os

for path in os.listdir("./crickets"):
    sound = AudioSegment.from_mp3("./crickets/" + path)
    sound.export("./crickets/" + path[:-3] + "wav", format="wav")