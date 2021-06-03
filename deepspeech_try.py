import deepspeech
import numpy as np
import os
import pyaudio
import time

if __name__ == '__main__':

    # DeepSpeech parameters
    ALPHA = 0.75
    BETA = 1.85
    # Make DeepSpeech ModeL
    model = deepspeech.Model("deepspeech-0.9.3-models.pbmm")
    model.enableExternalScorer('deepspeech-0.9.3-models.scorer')
    model.setScorerAlphaBeta(ALPHA, BETA)

    # Create a Streaming session
    context = model.createStream()

    # transcript text
    text_so_far = ''

    # callback function for pyaudio
    def process_audio(in_data, frame_count, time_info, status):
        global text_so_far
        data16 = np.frombuffer(in_data, dtype=np.int16)
        context.feedAudioContent(data16)
        text = context.intermediateDecode()
        if text != text_so_far:
            print('Interim text = {}'.format(text))
            text_so_far = text
        return in_data, pyaudio.paContinue


    # PyAudio parameters
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    CHUNK_SIZE = 1024 * 15

    # Feed audio to deepspeech in a callback to PyAudio
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        stream_callback=process_audio
    )

    print('Please start speaking, when done press Ctrl-C ...')
    stream.start_stream()

    try:
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        # PyAudio
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print('Finished recording.')
        # DeepSpeech
        text = model.finishStream(context)
        print('Final text = {}'.format(text))