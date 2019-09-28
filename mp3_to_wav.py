from pydub import AudioSegment


def trans_mp3_to_wav(filepath):
    song = AudioSegment.from_mp3(filepath)
    song.export("input.wav", format="wav")


if __name__ == '__main__':
    trans_mp3_to_wav('bbc.mp3')
