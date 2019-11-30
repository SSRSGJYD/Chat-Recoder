from pyannote.audio.labeling.extraction import SequenceLabeling
from pyannote.audio.embedding.extraction import SequenceEmbedding
from pyannote.audio.signal import Binarize
from pyannote.audio.signal import Peak
from pyannote.core import Timeline
import numpy as np

SAD_MODEL = '/run/media/gerw/HDD/data/git-task/pyannote-audio/tutorials/models/speech_activity_detection/train/AMI.SpeakerDiarization.MixHeadset.train/weights/0280.pt'  # todo relative path
SCD_MODEL = '/run/media/gerw/HDD/data/git-task/pyannote-audio/tutorials/models/speaker_change_detection/train/AMI.SpeakerDiarization.MixHeadset.train/weights/0870.pt'
EMB_MODEL = '/run/media/gerw/HDD/data/git-task/pyannote-audio/tutorials/models/speaker_embedding/train/VoxCeleb.SpeakerVerification.VoxCeleb1.train/weights/2000.pt'

test_file = {'uri': 'filename', 'audio': '/run/media/gerw/HDD/data/git-task/ChatRecognizer/data/dialog.wav'}

sad = SequenceLabeling(model=SAD_MODEL)
scd = SequenceLabeling(model=SCD_MODEL)

sad_scores = sad(test_file)

binarize = Binarize(offset=0.94, onset=0.70, log_scale=True)

speech = binarize.apply(sad_scores, dimension=1)

for segment in speech:
    print(segment.start, segment.end)

scd_scores = scd(test_file)
peak = Peak(alpha=0.08, min_duration=0.40, log_scale=True)
partition = peak.apply(scd_scores, dimension=1)
print('------------------------------')
for segment in partition:
    print(segment.start, segment.end, segment.end - segment.start)

speech_turns = partition.crop(speech)

emb = SequenceEmbedding(model=EMB_MODEL, duration=1., step=0.2)
embeddings = emb(test_file)

long_turns = Timeline(segments=[s for s in speech_turns if s.duration > 1.1])

X = []
for segment in long_turns:
    x = embeddings.crop(segment, mode='strict')
    # print(x.shape)
    x = np.mean(x, axis=0)
    # print(x.shape)
    if np.any(np.isnan(x)):
        print('fxxk NaN!', segment.start, segment.end)
        print(segment.duration)
        continue
    X.append(x)

X = np.vstack(X)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, metric='cosine')
X_2d = tsne.fit_transform(X)

fig, ax = plt.subplots()
fig.set_figheight(5)
fig.set_figwidth(5)
plt.scatter(*X_2d.T)

fig.show()
