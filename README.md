# ChatRecognizer



## Submodules

### MP3 to WAV

example usage:

```python
import util

util.mp3_to_wav.trans_mp3_to_wav('path/to/input/mp3', 'path/to/output/wav')
```



### Silence Removal

Example usage:

```python
import util
import librosa

orig_data, sr = librosa.load('path/to/wav', mono=True)
segment_list = util.verification.extract_nonsilence(orig_data, min_segment_duration=1.0, samplerate=sr)
```

where `segment_list` is a list of `numpy.ndarray` .

You can set `min_segment_duration` which is the minimum duration of a segment in second.



### Split sentences

To make sure that different sentences be split, set a larger `threshold` . By experience, 0.002 is enough.

```python
import util
import librosa

orig_data, sr = librosa.load('path/to/wav', mono=True)
segment_list = util.verification.extract_nonsilence(orig_data, min_segment_duration=1.0, samplerate=sr, threshold=0.002)
```



### Speech Recognition

Pretrained deepspeech model can be downloaded at https://github.com/mozilla/DeepSpeech/releases/download/v0.5.1/deepspeech-0.5.1-models.tar.gz.

[sox](https://sourceforge.net/projects/sox/files/sox/14.4.2/) should be installed and its path be added to environment variables.

example usage:

```python
import recognizer

text = recognizer.recognize_by_DeepSpeech('path/to/wav', dir='path/to/pretrained/model')
```

