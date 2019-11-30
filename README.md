# ChatRecognizer



## Submodules

### MP3 to WAV

example usage:

```python
import util

util.mp3_to_wav.trans_mp3_to_wav('path/to/input/mp3', 'path/to/output/wav')
```



### Silence Removal

example usage:

```python
import util
import librosa

orig_data, sr = librosa.load('path/to/wav', mono=True)
segment_list = util.verification.extract_nonsilence(orig_data, samplerate=sr)
```

where `segment_list` is a list of `numpy.ndarray` .



### Speech Recognition

Pretrained deepspeech model can be downloaded at https://github.com/mozilla/DeepSpeech/releases/download/v0.5.1/deepspeech-0.5.1-models.tar.gz.

[sox](https://sourceforge.net/projects/sox/files/sox/14.4.2/) should be installed and its path be added to environment variables.

example usage:

```python
import main

text = main.recognize_by_DeepSpeech('path/to/wav', dir='path/to/pretrained/model')
```

