import numpy as np
import librosa
import speaker_verification_toolkit.tools as svt
import soundfile
import tqdm


def extract_nonsilence(data, min_segment_duration=1.0, samplerate=16000, segment_length=None, threshold=0.001135):
    '''
    Cut off silence parts from the signal audio data. Doesn't work with signals data affected by environment noise.
    You would consider apply a noise filter before using this silence filter or make sure that environment noise is small enough to be considered as silence.

    :param data: the audio signal data
    :param min_segment_duration: pre-set minimum duration of a segment in second
    :param samplerate: if no segment_length is given, segment_length will be equals samplerate/100 (around 0.01 secs per segment).
    :param segment_length: the number of frames per segment. I.e. for a sample rate SR, a segment length equals SR/100 will represent a chunk containing 0.01 seconds of audio.
    :param threshold: the threshold value. Values less than or equal values will be cut off. The default value was defined at [1] (see the references).
    :returns: the param "data" without silence parts.
    '''
    if segment_length is None:
        segment_length = int(samplerate/100)

    segments = []
    filtered_data = np.array([])

    for index in tqdm.tqdm(range(0, len(data), segment_length)):
        data_slice = data[index : index + segment_length]

        squared_data_slice = np.square(data_slice)
        mean = np.sqrt(np.mean(squared_data_slice))

        if mean > threshold:
            filtered_data = np.append(filtered_data, data_slice)
        elif filtered_data.shape[0] > samplerate * min_segment_duration:
            segments.append(filtered_data)
            filtered_data = np.array([])

    if filtered_data.shape[0] > samplerate * min_segment_duration:
        segments.append(filtered_data)

    return segments


def inter_mfcc_distances(data1, data2, step, bin1=10, bin2=10):
    step1 = min(data1.shape[0] // bin1, step)
    step2 = min(data2.shape[0] // bin2, step)
    mfcc1 = []
    mfcc2 = []
    for i in range(bin1):
        mfcc = svt.mfcc(data1[i*step1: (i+1)*step1])
        mfcc1.append(mfcc)
    for i in range(bin2):
        mfcc = svt.mfcc(data2[i*step2: (i+1)*step2])
        mfcc2.append(mfcc)
    distances = []
    for i in range(bin1):
        for j in range(bin2):
            distance = svt.compute_distance(mfcc1[i], mfcc2[j])
            distances.append(distance)
    return distances


def intra_mfcc_distances(data, step, bin=10):
    step = min(data.shape[0] // bin, step)
    mfccs = []
    for i in range(10):
        mfcc = svt.mfcc(data[i*step: (i+1)*step])
        mfccs.append(mfcc)
    distances = []
    for i in range(bin-1):
        for j in range(i+1, bin):
            distance = svt.compute_distance(mfccs[i], mfccs[j])
            distances.append(distance)
    return distances


class Speaker:
    def __init__(self, audio, sr):
        self.audio = audio
        self.sr = sr
        self.distances = self.intra_mfcc_distances(self.sr // 100)
        # self.distances.sort()
        print('intra:', np.histogram(np.array(self.distances), 5))
        # self.mean = sum(self.distances) / len(self.distances)
        # print(self.mean)
        # print(self.distances[30])


    def intra_mfcc_distances(self, step, bin=10):
        """
        calculate distance of each two mfcc
        :return: a numpy array
        """
        return intra_mfcc_distances(self.audio, step, bin)

    def belong_to(self, segment):
        distances = inter_mfcc_distances(self.audio, segment, self.sr // 100)
        # print('max:', max(distances))
        # print('min:', min(distances))
        print('inter:', np.histogram(np.array(distances), 5))
        # greater_l = list(filter(lambda x:x>200, distances))
        # if len(greater_l) < 10:
        #     return True
        # else:
        #     return False


def segment_by_voice(segments, samplerate=16000, segment_length=None, threshold=200):
    '''
        Cut off silence parts from the signal audio data. Doesn't work with signals data affected by environment noise.
        You would consider apply a noise filter before using this silence filter or make sure that environment noise is small enough to be considered as silence.

        :param data: the audio signal data
        :param samplerate: if no segment_length is given, segment_length will be equals samplerate/100 (around 0.01 secs per segment).
        :param segment_length: the number of frames per segment. I.e. for a sample rate SR, a segment length equals SR/100 will represent a chunk containing 0.01 seconds of audio.
        :param threshold: the threshold value. Values less than or equal values will be cut off. The default value was defined at [1] (see the references).
        :returns: the param "data" without silence parts.
        '''
    if segment_length is None:
        segment_length = int(samplerate / 100)

    voice_segments = []
    last_slice_mfcc = None
    for i, data in enumerate(segments):
        accumulate_data = np.array([])
        for index in range(0, len(data), segment_length):
            data_slice = data[index: index + segment_length]
            mfcc = svt.extract_mfcc(data_slice)
            if last_slice_mfcc is None:
                distance = -1
            else:
                distance = svt.compute_distance(mfcc, last_slice_mfcc)
            last_slice_mfcc = mfcc

            if distance < threshold and distance > 0:
                # same voice
                accumulate_data = np.append(accumulate_data, data_slice)
            else:
                # different
                if accumulate_data.shape[0] > 0:
                    voice_segments.append(accumulate_data)
                accumulate_data = data_slice

        if accumulate_data.shape[0] > 0:
            voice_segments.append(accumulate_data)

    return voice_segments


def main():
    orig_data, sr = librosa.load('C:/ASR/audio/dialog_no_silence.wav', mono=True)
    print('original', orig_data.shape[0])
    segments = extract_nonsilence(orig_data, 1.0, samplerate=sr, threshold=0.002)
    print(len(segments))

    for i, data in enumerate(segments):
        soundfile.write('C:/ASR/audio/tmp/dialog_no_silence_part{}.wav'.format(i+1), data, sr)

    # voice_segments = segment_by_voice(segments, samplerate=sr, threshold=150)
    # print(len(voice_segments))


if __name__ == '__main__':
    main()
    # orig_data1, sr1 = librosa.load('C:/ASR/audio/speaker1.wav', mono=True)
    # orig_data2, sr2 = librosa.load('C:/ASR/audio/speaker2.wav', mono=True)
    # segment_length = int(sr1 / 100)
    # speaker = Speaker(orig_data1, sr1)
    # speaker.belong_to(orig_data2)
    
