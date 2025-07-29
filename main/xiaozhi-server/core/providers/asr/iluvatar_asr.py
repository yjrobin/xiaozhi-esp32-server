import base64
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
import os
from typing import Optional, Tuple, List
from core.providers.asr.dto.dto import InterfaceType
import requests
from core.providers.asr.base import ASRProviderBase
from config.logger import setup_logging
from requests.auth import HTTPBasicAuth

TAG = __name__
logger = setup_logging()


class ASRProvider(ASRProviderBase):

    def __init__(self, config: dict, delete_audio_file: bool = True):
        super().__init__()
        self.interface_type = InterfaceType.NON_STREAM
        self.username = config.get("username")
        self.password = config.get("password")
        self.output_dir = config.get("output_dir", "tmp/")
        self.delete_audio_file = delete_audio_file
        self.auth = HTTPBasicAuth(self.username, self.password)
        self.url = config.get("url")
        self.lang = config.get("lang","zh")
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

    async def speech_to_text(
        self, opus_data: List[bytes], session_id: str, audio_format="opus"
    ) -> Tuple[Optional[str], Optional[str]]:
        """将语音数据转换为文本"""
        if not opus_data:
            logger.bind(tag=TAG).warning("音频数据为空！")
            return None, None

        file_path = None
        try:
            # 将Opus音频数据解码为PCM
            if audio_format == "pcm":
                pcm_data = opus_data
            else:
                pcm_data = self.decode_opus(opus_data)
            combined_pcm_data = b"".join(pcm_data)

            # 判断是否保存为WAV文件
            if self.delete_audio_file:
                pass
            else:
                self.save_audio_to_file(pcm_data, session_id)
            start_time = time.time()
            response = requests.post(
                self.url,
                params={
                    'language': self.lang
                },
                auth=self.auth,
                headers={
                            'Content-Type': 'audio/wav'
                        },
                data=combined_pcm_data
            )
            response.raise_for_status()
            json_result = response.json()
            assert json_result["RecognitionStatus"] == "Success"
            result = json_result["DisplayText"]
            logger.bind(tag=TAG).debug(
                f"天数识别耗时: {time.time() - start_time:.3f}s | 结果: {result}"
            )

            return result, file_path

        except Exception as e:
            logger.bind(tag=TAG).error(f"处理音频时发生错误！{e}", exc_info=True)
            return None, file_path
