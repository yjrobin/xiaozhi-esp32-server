import os
import json
import uuid
import requests
import struct
from config.logger import setup_logging
from datetime import datetime
from core.providers.tts.base import TTSProviderBase
from xml.etree import ElementTree
from requests.auth import HTTPBasicAuth

TAG = __name__
logger = setup_logging()

class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.url = config.get("url", "https://ai-cloud.4paradigm.com:9443/ttsgate/cognitiveservices")
        self.format = config.get("format", "wav")
        self.audio_file_type = config.get("format", "wav")
        self.output_file = config.get("output_dir", "tmp/")
        self.username = config.get("username")
        self.password = config.get("password")
        self.method = "POST"
        self.headers = {
            'User-Agent': '4PD-TTS-Client',
            'Content-Type': 'application/ssml+xml'
        }
        self.lang = config.get("lang", "zh")
        if config.get("private_voice"):
            self.voice = config.get("private_voice")
        else:
            self.voice = config.get("voice")
        self.first = True
        self.auth = HTTPBasicAuth(self.username, self.password)
        logger.bind(tag=TAG).info("iluvatar tts inited")

    def generate_filename(self):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}.{self.format}")

    def _construct_wav_header(self, pcm_data):
        """
        Constructs a WAV file header for 16kHz, 16-bit mono PCM data.
        """
        num_channels = 1
        sample_width = 2  # 16-bit
        sample_rate = 16000
        num_frames = len(pcm_data) // (num_channels * sample_width)

        wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
                                 b'RIFF',
                                 36 + num_frames * num_channels * sample_width,
                                 b'WAVE',
                                 b'fmt ',
                                 16,
                                 1,
                                 num_channels,
                                 sample_rate,
                                 sample_rate * num_channels * sample_width,
                                 num_channels * sample_width,
                                 sample_width * 8,
                                 b'data',
                                 num_frames * num_channels * sample_width)
        return wav_header

    async def text_to_speak(self, text, output_file):
        # Construct the SSML payload
        xml_body = ElementTree.Element('speak', version='1.0')
        xml_body.set('lang', self.lang)
        voice = ElementTree.SubElement(xml_body, 'voice')
        voice.set('lang', self.lang)
        voice.set('name', self.voice)
        voice.text = text
        if self.first:
            params = {'fast_infer': 1}
            self.first = False
        else:
            params = {}

        body = ElementTree.tostring(xml_body, encoding="utf-8")

        if self.method.upper() == "POST":
            resp = requests.post(self.url, params=params, data=body, headers=self.headers, auth=self.auth)
        else:
            resp = requests.get(self.url, params=params, data=body, headers=self.headers, auth=self.auth)
        if resp.status_code == 200:
            pcm_data = resp.content
            wav_header = self._construct_wav_header(pcm_data)
            if output_file:
                with open(output_file, "wb") as file:
                    file.write(wav_header)
                    file.write(pcm_data)
            else:
                return wav_header + resp.content
        else:
            error_msg = f"iluvatar_tts请求失败: {resp.status_code} - {resp.text}"
            logger.bind(tag=TAG).error(error_msg)
            raise Exception(error_msg)  # 抛出异常，让调用方捕获
