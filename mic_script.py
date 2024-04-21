# import ctypes
# import pathlib
# from pydub import AudioSegment
# import re
# from typing import Tuple
# import numpy as np


# # class WhisperFullParams(ctypes.Structure):
# #     _fields_ = [
# #         ("strategy", ctypes.c_int),
# #         #
# #         ("n_threads", ctypes.c_int),
# #         ("n_max_text_ctx", ctypes.c_int),
# #         ("offset_ms", ctypes.c_int),
# #         ("duration_ms", ctypes.c_int),
# #         #
# #         ("translate", ctypes.c_bool),
# #         ("no_context", ctypes.c_bool),
# #         ("no_timestamps", ctypes.c_bool),
# #         ("single_segment", ctypes.c_bool),
# #         ("print_special", ctypes.c_bool),
# #         ("print_progress", ctypes.c_bool),
# #         ("print_realtime", ctypes.c_bool),
# #         ("print_timestamps", ctypes.c_bool),
# #         #
# #         ("token_timestamps", ctypes.c_bool),
# #         ("thold_pt", ctypes.c_float),
# #         ("thold_ptsum", ctypes.c_float),
# #         ("max_len", ctypes.c_int),
# #         ("split_on_word", ctypes.c_bool),
# #         ("max_tokens", ctypes.c_int),
# #         #
# #         ("speed_up", ctypes.c_bool),
# #         ("debug_mode", ctypes.c_bool),
# #         ("audio_ctx", ctypes.c_int),
# #         #
# #         ("tdrz_enable", ctypes.c_bool),
# #         #
# #         ("initial_prompt", ctypes.c_char_p),
# #         ("prompt_tokens", ctypes.c_void_p),
# #         ("prompt_n_tokens", ctypes.c_int),
# #         #
# #         ("language", ctypes.c_char_p),
# #         ("detect_language", ctypes.c_bool),
# #         #
# #         ("suppress_blank", ctypes.c_bool),
# #         ("suppress_non_speech_tokens", ctypes.c_bool),
# #         #
# #         ("temperature", ctypes.c_float),
# #         ("max_initial_ts", ctypes.c_float),
# #         ("length_penalty", ctypes.c_float),
# #         #
# #         ("temperature_inc", ctypes.c_float),
# #         ("entropy_thold", ctypes.c_float),
# #         ("logprob_thold", ctypes.c_float),
# #         ("no_speech_thold", ctypes.c_float),
# #         #
# #         ("greedy", ctypes.c_int * 1),
# #         ("beam_search", ctypes.c_int * 2),
# #         #
# #         ("new_segment_callback", ctypes.c_void_p),
# #         ("new_segment_callback_user_data", ctypes.c_void_p),
# #         #
# #         ("progress_callback", ctypes.c_void_p),
# #         ("progress_callback_user_data", ctypes.c_void_p),
# #         #
# #         ("encoder_begin_callback", ctypes.c_void_p),
# #         ("encoder_begin_callback_user_data", ctypes.c_void_p),
# #         #
# #         ("logits_filter_callback", ctypes.c_void_p),
# #         ("logits_filter_callback_user_data", ctypes.c_void_p),
# #         #
# #         ("grammar_rules", ctypes.POINTER(ctypes.c_void_p)),
# #         ("n_grammar_rules", ctypes.c_size_t),
# #         ("i_start_rule", ctypes.c_size_t),
# #         ("grammar_penalty", ctypes.c_float),
# #     ]

# # def transcribe(whisper, params, ctx, audio_segment: AudioSegment) -> Tuple[str, float]:
# #     if len(audio_segment) <= 100:
# #         return "", 0.0
# #     normalized = (
# #         np.frombuffer(
# #             audio_segment.set_frame_rate(16000).raw_data, dtype=np.int16
# #         ).astype("float32")
# #         / 32768.0
# #     )

# #     result = whisper.whisper_full(
# #         ctypes.c_void_p(ctx),
# #         params,
# #         normalized.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
# #         len(normalized),
# #     )
# #     if result != 0:
# #         print("Error: {}".format(result))
# #         exit(1)
# #     text: str = whisper.whisper_full_get_segment_text(ctypes.c_void_p(ctx), 0).decode(
# #         "utf-8"
# #     )
# #     # heuristic to filter out non-speech
# #     if not re.search(r"^\w.*", text.strip()):
# #         return "", 0.0
# #     return text, 1.0


# # class WhisperCPPTranscriber:
# #     def __init__(self, libname: str, fname_model: str):
# #         self.libname = libname
# #         self.fname_model = fname_model

# #         # whisper cpp
# #         # load library and model
# #         libname = pathlib.Path().absolute() / self.libname  # type: ignore
# #         self.whisper = ctypes.CDLL(libname)

# #         # tell Python what are the return types of the functions
# #         self.whisper.whisper_init_from_file.restype = ctypes.c_void_p
# #         self.whisper.whisper_full_default_params.restype = WhisperFullParams
# #         self.whisper.whisper_full_get_segment_text.restype = ctypes.c_char_p

# #         # initialize whisper.cpp context
# #         self.ctx = self.whisper.whisper_init_from_file(self.fname_model.encode("utf-8"))

# #         # get default whisper parameters and adjust as needed
# #         self.params = self.whisper.whisper_full_default_params()
# #         self.params.print_realtime = False
# #         self.params.print_progress = False
# #         self.params.single_segment = True

# #     def transcribe(self, audio_segment: AudioSegment) -> str:
# #         transcription, _ = transcribe(
# #             self.whisper,
# #             self.params,
# #             self.ctx,
# #             audio_segment,
# #         )
# #         return transcription
# from whisper_cpp_cdll.core import run_whisper
# from whisper_cpp_cdll.util import read_audio



# if __name__ == "__main__":
#     # your whisper.cpp files path
#     libname = '/home/kevin/ai/whisper.cpp/libwhisper.so'
#     fname_model = '/home/kevin/ai/whisper.cpp/models/ggml-small.bin'
#     d = read_audio('/home/kevin/ai/whisper.cpp/samples/jfk.wav')
#     result = run_whisper(data = d, libname = libname, fname_model = fname_model)
#     print(result)





from faster_whisper import WhisperModel
import time

# or run on CPU with INT8
model = WhisperModel('tiny', device="cpu", compute_type="int8")

# audio = "./chinese_sample.wav"
audio = "/home/kevin/ai/whisper.cpp/samples/jfk.wav"
st = time.time()
# segments, info = model.transcribe(audio, beam_size=5, task = "translate", language='zh')
segments, info = model.transcribe(audio, beam_size=5, language='en')
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


print(f"inference time {time.time() - st}")
