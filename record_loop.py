import pyaudio
import wave
import time
import os
from datetime import datetime
from threading import *

form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 4096 # 2^12 samples for buffer
record_secs = 600 # seconds to record
dev_index = 10 # device index found by p.get_device_info_by_index(ii)
wav_output_filename = 'test1.wav' # name of .wav file
write_batch_duration = 5


class Save(Thread):
    def __init__(self, frames):
        super(Save, self).__init__()
        self.frames = frames
    
    def run(self):
        path = 'water-sound-classifier/samples/'
        curr_hr = datetime.now().strftime('%Y%m%d%H')
        folder_path = os.path.join(path,curr_hr)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
        curr_timestamp = int(datetime.now().timestamp())
        file_name = str(curr_timestamp)+'.wav'
        print('writing the recorded file for name: '+file_name)
        # print(file_name,chans,samp_rate,len(self.frames))
        wavefile = wave.open(folder_path+'/'+file_name,'wb')
        wavefile.setnchannels(chans)
        wavefile.setsampwidth(audio.get_sample_size(form_1))
        wavefile.setframerate(samp_rate)
        wavefile.writeframes(b''.join(self.frames))
        wavefile.close()

def init():
    # create pyaudio instantiation
    audio = pyaudio.PyAudio() 
    # create pyaudio stream
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                        input_device_index = dev_index,input = True, \
                        frames_per_buffer=chunk)
    print("recording")
    return stream,audio

# def save_audio(curr_frames):
#     # save the audio frames as .wav file
#     curr_timestamp = time.time()
#     file_name = 'segment_'+curr_timestamp
#     wavefile = wave.open('recorded/'+file_name,'wb')
#     wavefile.setnchannels(chans)
#     wavefile.setsampwidth(audio.get_sample_size(form_1))
#     wavefile.setframerate(samp_rate)
#     wavefile.writeframes(b''.join(curr_frames))
#     wavefile.close()

def listen(stream,audio):
    frames = []
    write_frame_size = int((samp_rate/chunk)*write_batch_duration)
    print(write_frame_size, int((samp_rate/chunk)*record_secs))

    # loop through stream and append audio chunks to frame array
    for ii in range(0,int((samp_rate/chunk)*record_secs)):
        # print(ii,len(frames))
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)
        if len(frames) == write_frame_size:
            # print("matche", len(frames))
            copied_frames = frames.copy()
            writer = Save(copied_frames)
            writer.start()
            frames.clear()


    # stop the stream, close it, and terminate the pyaudio instantiation
    stream.stop_stream()
    stream.close()
    audio.terminate()

    print("finished recording")


stream, audio = init()
listen(stream, audio)

