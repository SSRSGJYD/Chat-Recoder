import speech_recognition as sr

# obtain audio from the microphone
r = sr.Recognizer()
harvard = sr.AudioFile('input.wav')
with harvard as source:
    audio = r.record(source)

# recognize speech using Sphinx
try:
    print("Sphinx thinks you said:\n" + r.recognize_sphinx(audio))
except sr.UnknownValueError:
    print("Sphinx could not understand audio")
except sr.RequestError as e:
    print("Sphinx error; {0}".format(e))
