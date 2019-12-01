#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import os
import shlex
import subprocess
import sys
import wave

from deepspeech import Model, printVersions
from timeit import default_timer as timer

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_ALPHA = 0.75

# The beta hyperparameter of the CTC decoder. Word insertion bonus.
LM_BETA = 1.85

# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9


def convert_samplerate(audio_path):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate 16000 --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(
        quote(audio_path))
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use 16kHz files or install it: {}'.format(e.strerror))

    return 16000, np.frombuffer(output, np.int16)


def metadata_to_string(metadata):
    return ''.join(item.character for item in metadata.items)


class VersionAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(VersionAction, self).__init__(nargs=0, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        printVersions()
        exit(0)


class DeepSpeechWrapper:
    def __init__(self, dir):
        parser = argparse.ArgumentParser(description='Running DeepSpeech inference.')
        parser.add_argument('--model', default=os.path.join(dir, 'output_graph.pbmm'),
                            help='Path to the model (protocol buffer binary file)')
        parser.add_argument('--alphabet', default=os.path.join(dir, 'alphabet.txt'),
                            help='Path to the configuration file specifying the alphabet used by the network')
        parser.add_argument('--lm', nargs='?', default=os.path.join(dir, 'lm.binary'),
                            help='Path to the language model binary file')
        parser.add_argument('--trie', nargs='?', default=os.path.join(dir, 'trie'),
                            help='Path to the language model trie file created with native_client/generate_trie')
        parser.add_argument('--version', action=VersionAction,
                            help='Print version and exits')
        parser.add_argument('--extended', required=False, action='store_true',
                            help='Output string from extended metadata')
        self.args = parser.parse_args('')  # shadow the system args

        self.ds = Model(self.args.model, N_FEATURES, N_CONTEXT, self.args.alphabet, BEAM_WIDTH)
        
        self.audio = None
        self.audio_length = 0
        self.fs = 16000

        if self.args.lm and self.args.trie:
            # print('Loading language model from files {} {}'.format(self.args.lm, self.args.trie), file=sys.stderr)
            # lm_load_start = timer()
            self.ds.enableDecoderWithLM(self.args.alphabet, self.args.lm, self.args.trie, LM_ALPHA, LM_BETA)
            # lm_load_end = timer() - lm_load_start
            # print('Loaded language model in {:.3}s.'.format(lm_load_end), file=sys.stderr)

    def set_input(self, filename):
        fin = wave.open(filename, 'rb')
        self.fs = fin.getframerate()
        if self.fs != 16000:
            print(
                'Warning: original sample rate ({}) is different than 16kHz. Resampling might produce erratic speech recognition.'.format(
                    self.fs), file=sys.stderr)
            self.fs, audio = convert_samplerate(filename)
        else:
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

        self.audio = audio
        self.audio_length = fin.getnframes() / self.fs
        fin.close()

    def recognize_audio(self, start_time, end_time):
        start_frame = int(start_time * self.fs)
        end_frame = int(end_time * self.fs)
        seq = self.audio[start_frame:end_frame]
        if self.args.extended:
            return metadata_to_string(self.ds.sttWithMetadata(seq, self.fs))
        else:
            return self.ds.stt(seq, self.fs)