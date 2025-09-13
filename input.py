import pvporcupine
import pyaudio
import struct
import speech_recognition as sr
import threading
import queue
import select
import sys
from dotenv import load_dotenv
import os

load_dotenv()

_recognizer = None
_microphone = None
_speech_initialized = False

def _initialize_speech():
    """Initialize speech recognition for voice commands"""
    global _recognizer, _microphone, _speech_initialized
    
    if not _speech_initialized:
        _recognizer = sr.Recognizer()
        _microphone = sr.Microphone()
        
        with _microphone as source:
            _recognizer.adjust_for_ambient_noise(source, duration=1)
        _speech_initialized = True

def _get_voice_command():
    """Get voice command using speech recognition after wake word is detected"""
    _initialize_speech()
    
    try:
        with _microphone as source:
            print("Listening for command...")
            audio = _recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        try:
            command = _recognizer.recognize_google(audio)
            return command
        except (sr.UnknownValueError, sr.RequestError):
            return None
            
    except sr.WaitTimeoutError:
        return None

def _porcupine_wake_word_listener(input_queue, listening_flag):
    """Porcupine wake word detection in background thread"""
    access_key = os.getenv("PVPORCUPINE_ACCESS_TOKEN")
    
    if not access_key:
        input_queue.put(("error", "Missing Porcupine access token"))
        return
    
    try:
        porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=["computer"]
        )

        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )

        while listening_flag[0]:
            try:
                pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:
                    voice_command = _get_voice_command()
                    if voice_command:
                        input_queue.put(("voice", voice_command))

            except Exception:
                if listening_flag[0]:
                    break

    except Exception as e:
        input_queue.put(("error", f"Porcupine failed: {e}"))
        return

    finally:
        try:
            audio_stream.stop_stream()
            audio_stream.close()
            pa.terminate()
            porcupine.delete()
        except:
            pass

def get_input():
    """
    Get user input via text or voice with 'computer' wake word detection.
    
    Returns:
        tuple: (input_type, content) where:
            - input_type: "voice", "text", "error", or "quit"
            - content: The actual input content or error message
    """
    
    input_queue = queue.Queue()
    listening_flag = [True]
    
    wake_word_thread = threading.Thread(
        target=_porcupine_wake_word_listener,
        args=(input_queue, listening_flag),
        daemon=True
    )
    wake_word_thread.start()
    
    try:
        while True:
            
            if not input_queue.empty():
                listening_flag[0] = False
                return input_queue.get()
            
            if hasattr(select, 'select'):
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    text_input = input().strip()
                    listening_flag[0] = False
                    
                    if text_input.lower() == 'quit':
                        return "quit", None
                    elif text_input:
                        return "text", text_input
            else:
                
                try:
                    import msvcrt
                    if msvcrt.kbhit():
                        text_input = input().strip()
                        listening_flag[0] = False
                        
                        if text_input.lower() == 'quit':
                            return "quit", None
                        elif text_input:
                            return "text", text_input
                except ImportError:
                    text_input = input().strip()
                    listening_flag[0] = False
                    
                    if text_input.lower() == 'quit':
                        return "quit", None
                    elif text_input:
                        return "text", text_input
            
    except KeyboardInterrupt:
        listening_flag[0] = False
        return "quit", None

if __name__ == "__main__":
    while True:
        input_type, content = get_input()
        
        if input_type == "quit":
            print("Goodbye!")
            break
        elif input_type == "error":
            print(f"Error: {content}")
            break
        else:
            print(f"Received {input_type}: {content}")