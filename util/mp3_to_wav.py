from pydub import AudioSegment


def trans_mp3_to_wav(input, output):
    song = AudioSegment.from_mp3(input)
    song.export(output, format="wav")


if __name__ == "__main__":

    sound = AudioSegment.from_mp3("C:/ASR/audio/dialog.mp3")

    start_time = "7:40"
    stop_time = "8:08"
    # print("time:",start_time,"~",stop_time)
    start_time = (int(start_time.split(':')[0])*60+int(start_time.split(':')[1]))*1000
    stop_time = (int(stop_time.split(':')[0])*60+int(stop_time.split(':')[1]))*1000
    # print("ms:",start_time,"~",stop_time)
    crop_audio = sound[start_time:stop_time]
    crop_audio.set_frame_rate(16000)
    crop_audio.export("C:/ASR/audio/dialog.wav", format="wav")
