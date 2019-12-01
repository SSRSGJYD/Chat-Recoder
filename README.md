# Chat Recorder

This is a note-taker that can distinguish between different talkers and take down what they said sequentially.



## Installation

1. Create a Python 3.x environment

2. Install required python packages in `requirements.txt`

   ```shell
   pip install -r requirements.txt
   ```

3. Install `pyannote` from develop branch of [github repo](https://github.com/pyannote/pyannote-audio) 

4. Download and install [sox](https://sourceforge.net/projects/sox/files/sox/14.4.2/) , add it to `$PATH`  

5. Download [pretrained deepspeech model](https://github.com/mozilla/DeepSpeech/releases/download/v0.5.1/deepspeech-0.5.1-models.tar.gz) to  `/data` and decompress it there



## Getting Started

1. Run ` main,py` .
2. Click `select` button and choose your input `wav` file.
3. Click `exec` button and the recognized text will be shown in the text editor below.
4. Click `clear` to clear all the text.

[Here](./demo/demo.avi) is a demo video.



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



### Audio Segmentation & Speaker Verification

Audio segmentation and speaker verification are implemented based on [`pyannote-audio`](http://www.github.com/pyannote/pyannote-audio) . 



### Speech Recognition

Speech recognition is implemented based on [DeepSpeech](https://github.com/mozilla/DeepSpeech). 