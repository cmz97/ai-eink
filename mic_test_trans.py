import pyaudio
import wave
from Drivers.SAM.sam import SAM
import threading
import time
from faster_whisper import WhisperModel
from einkDSP import einkDSP
from Ebook_GUI import EbookGUI
from utils import * 
# from rag_script import query_pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class AudioRecorder:

    FORMAT = pyaudio.paInt16 # 16-bit resolution
    CHANNELS = 1 # 1 channel
    RATE = 44100 # 44.1kHz sampling rate
    CHUNK = 4096 # 2^12 samples for buffer
    record_secs = 3 # seconds to record
    dev_index = 2 # device index found by p.get_device_info_by_index(ii)
    wav_output_filename = './temp.wav' # name of .wav file

    def __init__(self):
        self.p = pyaudio.PyAudio() # create pyaudio instantiation
        self.frames = []
        self.recording = False
        self.st = threading.Event()
        self.recording_thread = None
        self.file_ready_event = threading.Event()  # Event to signal file is ready

    def record_control(self):
        if not self.recording : 
            self.start_record()
            return True
        self.stop()
        return False
        
    def start_record(self):
        self.frames = []
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

        # Disable the Start Recording button and enable the Stop Recording button
        self.recording = True

        # Start the recording loop in a separate thread
        self.st.set()
        threading.Thread(target=self.record_loop).start()

    def stop(self):
        # Signal the recording loop to stop
        self.st.clear()
        self.recording = False

    def record_loop(self):
        self.file_ready_event.clear()  # Clear the event at the start
        
        while self.st.is_set():
            data = self.stream.read(self.CHUNK)
            self.frames.append(data)
            print("* recording")
        self.stream.close()

        wf = wave.open(self.wav_output_filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.file_ready_event.set()  # Set the event after file is written


class Whisper:
    audio = './temp.wav'
    def __init__(self):
        self.model = WhisperModel('tiny', device="cpu", compute_type="int8")

    def transcribe(self):
        st = time.time()
        segments, info = self.model.transcribe(self.audio, beam_size=5, language='en')
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            yield segment.text
        print(f"inference time {time.time() - st}")

    def translate(self):
        pass # potentially only ok when tranlate to en



class Application:
    def __init__(self):


        self.text_buffer = []
        self.image_buffer = []

        self.eink = einkDSP()
        self.eink.epd_init_fast()
        self.eink.PIC_display_Clear()
        self.in_4g = False
        self.gui = EbookGUI()

        # translation
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
        self.cn_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")


        self.wp = Whisper()
        self.ar = AudioRecorder()
        self.sam = SAM(self.press_callback)


        
        self.image = self.gui.canvas

        # self.rag_pipeline = query_pipeline

    def eink_display_4g(self, hex_pixels):
        logging.info('eink_display_4g')
        self.eink.epd_w21_init_4g()
        self.eink.pic_display_4g(hex_pixels)
        self.eink.epd_sleep()
        self.in_4g = True

    def eink_init(self):
        self.eink.epd_init_fast()
        self.eink.PIC_display_Clear()

    def eink_display_2g(self, hex_pixels):
        logging.info('eink_display_2g')
        if self.in_4g : 
            self.transit()
            self.in_4g = False

        self.eink.epd_init_part()
        self.eink.PIC_display(hex_pixels)

    def transit(self):
        self.eink.epd_init_fast()
        self.eink.PIC_display_Clear()
    
    def clear_screen(self):
        self.gui.clear_page()
        image = Image.new("L", (eink_width, eink_height), "white")
        hex_pixels = dump_1bit(np.array(image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.uint8))
        self.eink.epd_init_part()
        self.eink.PIC_display(hex_pixels)

    def part_screen(self, hex_pixels):
        self.locked = True
        self.eink.epd_init_part()
        self.eink.PIC_display(hex_pixels)
        self.locked = False
        
    def full_screen(self, hex_pixels):
        self.eink.epd_w21_init_4g()
        self.eink.pic_display_4g(hex_pixels)
        self.eink.epd_sleep()

    def _status_check(self):
        if self.in_4g : 
            self.transit()
            self.in_4g = False
        
    def _fast_text_display(self, text="loading ..."):
        image = fast_text_display(self.image, text)
        grayscale = image.transpose(Image.FLIP_TOP_BOTTOM).convert('L')
        hex_pixels = dump_1bit_with_dithering(np.array(grayscale, dtype=np.float32))
        # hex_pixels = dump_1bit(np.array(image.transpose(Image.FLIP_TOP_BOTTOM), dtype=np.uint8))
        self.part_screen(hex_pixels)


    def update_screen(self, image = None):
        if not image : image = self.image
        # image = self._prepare_menu(self.image)
        # update screen
        grayscale = image.transpose(Image.FLIP_TOP_BOTTOM).convert('L')
        logging.info('preprocess image done')
        hex_pixels = dump_1bit_with_dithering(np.array(grayscale, dtype=np.float32))
        logging.info('2bit pixels dump done')
        self.part_screen(hex_pixels)

    def translate(self, text):
        st = time.time()
        translated = self.cn_model.generate(**self.tokenizer([text], return_tensors="pt", padding=True))
        res = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        print(f"translation: {res}")
        temp_buffer = [['->']] # newline
        # for word in res[0]: 
        #     self.stream_text(word, xscale=0.7)
        w, h = self.gui.text_area[1][0] - self.gui.text_area[0][0] , self.gui.text_area[1][1] - self.gui.text_area[0][1]
        for word in res[0]:
            temp_buffer, _ = format_text(
                temp_buffer, 
                word, 
                boxWidth= w,
                boxHeight= h,
                fontWidth=self.gui.font_size * 0.7, 
                fontHeight=self.gui.font_size * 1.4 * 0.65)
        
        time_taken = time.time() - st
        print(f"Translation done. Time taken: {time_taken:.2f} seconds")
        self.text_buffer.extend(temp_buffer)
        self.image = self.gui.draw_text_on_canvas(self.gui.canvas, [" ".join(x) for x in self.text_buffer])
        self.update_screen()        

    def stream_text(self, text, xscale = 0.55, yscale = 0.55):
        self._status_check()
        # screen_buffer = 10
        w, h = self.gui.text_area[1][0] - self.gui.text_area[0][0] , self.gui.text_area[1][1] - self.gui.text_area[0][1]
        self.text_buffer, new_page = format_text(
            self.text_buffer, 
            text, 
            boxWidth= w,
            boxHeight= h,
            fontWidth=self.gui.font_size * xscale, 
            fontHeight=self.gui.font_size * 1.4 * yscale)

        # call for screen update
        if new_page : self.gui.clear_page()
        self.image = self.gui.draw_text_on_canvas(self.gui.canvas, [" ".join(x) for x in self.text_buffer])
        self.update_screen()        

    
    def press_callback(self, key):
        if not self.ar.recording : 
            # new page
            self.text_buffer.append(['+']) # newline
            # self.gui.clear_page()
            # self.image = self.gui.canvas
            
            self._fast_text_display("recoding ...") # call for recording 
            self.ar.record_control()  # Start or stop recording
            return
        
        # else finish the recording and display
        self.ar.record_control()  # Start or stop recording
        # Wait for the recording to finish and file to be ready
        self.ar.file_ready_event.wait()  
        # Now proceed with transcription
        # for sentence in self.wp.transcribe():
        for sentence in self.wp.transcribe():
            # process the sentences first in a thread
            thread = threading.Thread(target=self.translate, args=(sentence,))
            thread.start()
            for word in sentence.split():
                self.stream_text(word, xscale=0.5, yscale=0.65)
        

if __name__ == "__main__":
    
    app = Application()
    
    while True:
        print("ping")
        time.sleep(0.5)




# import time

# # or run on CPU with INT8

# # audio = "./chinese_sample.wav"
# audio = "/home/kevin/ai/whisper.cpp/samples/jfk.wav"
# st = time.time()
# # segments, info = model.transcribe(audio, beam_size=5, task = "translate", language='zh')
# segments, info = model.transcribe(audio, beam_size=5, language='en')
# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


# print(f"inference time {time.time() - st}")
