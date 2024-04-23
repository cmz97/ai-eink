import pyaudio
import wave
from Drivers.SAM.sam import SAM
import threading
import time
from faster_whisper import WhisperModel
from einkDSP import einkDSP
from Ebook_GUI import EbookGUI
from utils import * 

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


class Application:
    def __init__(self):
        self.eink = einkDSP()
        self.eink.epd_init_fast()
        self.eink.PIC_display_Clear()
        self.in_4g = False
        self.gui = EbookGUI()

        self.wp = Whisper()
        self.ar = AudioRecorder()
        self.sam = SAM(self.press_callback)


        self.image = self.gui.canvas

        self.text_buffer = []
        self.image_buffer = []

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


    def stream_text(self, text):
        self._status_check()
        # screen_buffer = 10
        w, h = self.gui.text_area[1][0] - self.gui.text_area[0][0] , self.gui.text_area[1][1] - self.gui.text_area[0][1]
        scale = 0.55
        self.text_buffer, new_page = format_text(
            self.text_buffer, 
            text, 
            boxWidth= w,
            boxHeight= h,
            fontWidth=self.gui.font_size * scale, 
            fontHeight=self.gui.font_size * 1.4 * scale)

        # call for screen update
        if new_page : self.gui.clear_page()
        self.image = self.gui.draw_text_on_canvas(self.gui.canvas, [" ".join(x) for x in self.text_buffer])
        self.update_screen()        

    def press_callback(self, key):
        if not self.ar.recording : 
            self._fast_text_display("recoding ...") # call for recording 
        
        if not self.ar.record_control(): # print on screen
            # TRIGGER TRANSCRIBE
            for sentence in self.wp.transcribe():
                for word in sentence.split():
                    self.stream_text(word)
        

if __name__ == "__main__":
    
    app = Application()

    while True:
        print("ping")
        time.sleep(0.2)




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
