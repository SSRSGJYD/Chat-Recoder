import string

import numpy as np
import speech_recognition as sr
from matplotlib import pyplot as plt
from pyannote.audio.embedding.extraction import SequenceEmbedding
from pyannote.audio.labeling.extraction import SequenceLabeling
from pyannote.audio.signal import Binarize
from pyannote.audio.signal import Peak
from pyannote.core import SlidingWindowFeature
from pyannote.core import Timeline
from pyannote.core import notebook
from GUI import AudioGUI

from model.DeepSpeech import DeepSpeechWrapper

SAD_MODEL = 'data/speech_activity_detection/train/AMI.SpeakerDiarization.MixHeadset.train/weights/0280.pt'
SCD_MODEL = 'data/speaker_change_detection/train/AMI.SpeakerDiarization.MixHeadset.train/weights/0870.pt'
EMB_MODEL = 'data/speaker_embedding/train/VoxCeleb.SpeakerVerification.VoxCeleb1.train/weights/2000.pt'


def l2_dist(a, b):
    return np.mean((a - b) ** 2)


class SpeakerSegmentation:
    def __init__(self, sad_model_path, scd_model_path, emb_model_path):
        self.sad = SequenceLabeling(model=sad_model_path)
        self.scd = SequenceLabeling(model=scd_model_path)
        self.emb = SequenceEmbedding(model=emb_model_path, duration=1., step=0.2)
        self.fa = None

    def find_parent(self, x):
        if self.fa[x] == -1:
            return x
        self.fa[x] = self.find_parent(self.fa[x])
        return self.fa[x]

    def min_spanning_tree(self, input, target_component=1, target_threshold=1e9):
        self.fa = np.full(len(input), -1, np.int32)
        n_component = len(input)
        dist = []
        for i in range(len(input)):
            for j in range(i + 1, len(input)):
                dist.append([l2_dist(input[i][1], input[j][1]), i, j])
        dist = list(sorted(dist))
        for d, i, j in dist:
            print('pending dist:', d)
            if d > target_threshold:
                break
            if n_component <= target_component:
                break
            print('accepted dist:', d)  # fixme
            i = self.find_parent(i)
            j = self.find_parent(j)
            if i != j:
                self.fa[j] = i
                n_component -= 1

        res = []
        namespace = {}
        names = iter(string.ascii_uppercase)  # will fail if more than 26 people
        for i, (segment, _) in enumerate(input):
            p = self.find_parent(i)
            if p not in namespace:
                namespace[p] = next(names)
            name = namespace[p]
            if i > 0 and res[-1][1] == name:
                res[-1] = (res[-1][0].from_json({
                    'start': res[-1][0].start,
                    'end': segment.end
                }), res[-1][1])
            else:
                res.append((segment, name))
        return res

    def annotate_speakers(self, filename, num_people, visualization=True):
        test_file = {'uri': 'filename', 'audio': filename}

        sad_scores = self.sad(test_file)
        binarize = Binarize(offset=0.5, onset=0.70, log_scale=True)
        speech = binarize.apply(sad_scores, dimension=1)

        scd_scores = self.scd(test_file)
        peak = Peak(alpha=0.1, min_duration=1, log_scale=True)
        partition = peak.apply(scd_scores, dimension=1)

        speech_turns = partition.crop(speech)
        embeddings = self.emb(test_file)
        long_turns = Timeline(segments=[s for s in speech_turns if s.duration > 1.1])

        res = []
        for segment in long_turns:
            x = embeddings.crop(segment, mode='strict')
            if x.size == 0:
                continue
            x = np.mean(x, axis=0)
            if np.any(np.isnan(x)):
                continue
            res.append((segment, x))

        if visualization:

            dist = []
            for i in range(len(res)):
                for j in range(i + 1, len(res)):
                    dist.append(l2_dist(res[i][1], res[j][1]))
            fig, ax = plt.subplots()
            ax.scatter(np.arange(len(dist)), np.array(sorted(dist)))
            fig.show()

            # let's visualize SAD and SCD results using pyannote.core visualization API

            # helper function to make visualization prettier
            plot_ready = lambda scores: SlidingWindowFeature(np.exp(scores.data[:, 1:]), scores.sliding_window)

            # create a figure with 6 rows with matplotlib
            nrows = 6
            fig, ax = plt.subplots(nrows=nrows, ncols=1)
            fig.set_figwidth(20)
            fig.set_figheight(nrows * 2)

            # 1st row: reference annotation
            # notebook.plot_annotation(test_file['annotation'], ax=ax[0])
            # ax[0].text(notebook.crop.start + 0.5, 0.1, 'reference', fontsize=14)

            # 2nd row: SAD raw scores
            notebook.plot_feature(plot_ready(sad_scores), ax=ax[1])
            ax[1].text(notebook.crop.start + 0.5, 0.6, 'SAD\nscores', fontsize=14)
            ax[1].set_ylim(-0.1, 1.1)

            # 3rd row: SAD result
            notebook.plot_timeline(speech, ax=ax[2])
            ax[2].text(notebook.crop.start + 0.5, 0.1, 'SAD', fontsize=14)

            # 4th row: SCD raw scores
            notebook.plot_feature(plot_ready(scd_scores), ax=ax[3])
            ax[3].text(notebook.crop.start + 0.5, 0.3, 'SCD\nscores', fontsize=14)
            ax[3].set_ylim(-0.1, 0.6)

            # 5th row: SCD result
            notebook.plot_timeline(partition, ax=ax[4])
            ax[4].text(notebook.crop.start + 0.5, 0.1, 'SCD', fontsize=14)

            # 6th row: combination of SAD and SCD
            notebook.plot_timeline(speech_turns, ax=ax[5])
            ax[5].text(notebook.crop.start + 0.5, 0.1, 'speech turns', fontsize=14)

            fig.show()

        res = self.min_spanning_tree(res, target_component=num_people)
        return res


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


def recognize_by_DeepSpeech(file, dir, num_people):
    wrapper = DeepSpeechWrapper(dir)
    wrapper.set_input(file)
    slicer = SpeakerSegmentation(SAD_MODEL, SCD_MODEL, EMB_MODEL)
    slices = slicer.annotate_speakers(file, num_people, visualization=False)
    for segment, name in slices:
        text = wrapper.recognize_audio(segment.start, segment.end)
        print(name + ':', text, segment)
    print(wrapper.recognize_audio(0, wrapper.audio_length))


if __name__ == '__main__':
    # print(recognize_by_SpeechRecognizer("C:/ASR/audio/2830-3980-0043.wav"))
    # print(recognize_by_DeepSpeech("C:/ASR/audio/2830-3980-0043.wav"))

    # print(recognize_by_SpeechRecognizer("C:/ASR/audio/dialog.wav"))
    # print(recognize_by_DeepSpeech("C:/ASR/audio/dialog.wav"))
    recognize_by_DeepSpeech('data/dialog.wav', 'data/deepspeech-0.5.1-models', 2)
