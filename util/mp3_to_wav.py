from pydub import AudioSegment


def trans_mp3_to_wav(input, output):
    song = AudioSegment.from_mp3(input)
    song.export(output, format="wav")
