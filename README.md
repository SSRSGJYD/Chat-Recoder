# ChatRecognizer



## Submodules

### Silence Removal

example usage:

```python
import util
import librosa

orig_data, sr = librosa.load('C:/ASR/audio/dialog.wav', mono=True)
segment_list = util.verification.extract_nonsilence(orig_data, samplerate=sr)
```

where `segment_list` is a list of `numpy.ndarray` .



### Speech Recognition

Pretrained deepspeech model can be downloaded at https://github.com/mozilla/DeepSpeech/releases/download/v0.5.1/deepspeech-0.5.1-models.tar.gz.

[sox](https://sourceforge.net/projects/sox/files/sox/14.4.2/) should be installed and its path be added to environment variables.

example usage:

```python
import recognizer

text = recognizer.recognize_by_DeepSpeech('path/to/wav', dir='path/to/pretrained/model')
```

