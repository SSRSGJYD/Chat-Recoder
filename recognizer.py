import speech_recognition as sr
from model.DeepSpeech import main


def recognize_by_SpeechRecognizer(file):
    # obtain audio from the microphone
    r = sr.Recognizer()
    harvard = sr.AudioFile(file)
    with harvard as source:
        audio = r.record(source)
    # recognize speech using Sphinx
    try:
        return r.recognize_sphinx(audio)
    except sr.UnknownValueError:
        print("Sphinx could not understand audio")
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))


def recognize_by_DeepSpeech(file):
    text = main(file)
    return text


if __name__ == '__main__':
    print(recognize_by_SpeechRecognizer("C:/ASR/audio/2830-3980-0043.wav"))
    print(recognize_by_DeepSpeech("C:/ASR/audio/2830-3980-0043.wav"))
