from typing import List, Dict
from ..base import IntentProviderBase
from plugins_func.functions.play_music import initialize_music_handler
from config.logger import setup_logging
import re
import json
import hashlib
import time

TAG = __name__
logger = setup_logging()


class IntentProvider(IntentProviderBase):
    def __init__(self, config):
        super().__init__(config)
        self.llm = None
        self.promot = ""
        # 导入全局缓存管理器
        from core.utils.cache.manager import cache_manager, CacheType

        self.cache_manager = cache_manager
        self.CacheType = CacheType
        self.history_count = 4  # 默认使用最近4条对话记录
        self.pending_intents = {}

    def get_intent_system_prompt(self, functions_list: str) -> str:
        """
        根据配置的意图选项和可用函数动态生成系统提示词
        Args:
            functions: 可用的函数列表，JSON格式字符串
        Returns:
            格式化后的系统提示词
        """

        # 构建函数说明部分
        functions_desc = "可用的函数列表：\n"
        for func in functions_list:
            func_info = func.get("function", {})
            name = func_info.get("name", "")
            desc = func_info.get("description", "")
            params = func_info.get("parameters", {})

            functions_desc += f"\n函数名: {name}\n"
            functions_desc += f"描述: {desc}\n"

            if params:
                functions_desc += "参数:\n"
                for param_name, param_info in params.get("properties", {}).items():
                    param_desc = param_info.get("description", "")
                    param_type = param_info.get("type", "")
                    functions_desc += f"- {param_name} ({param_type}): {param_desc}\n"

            functions_desc += "---\n"

        prompt = (
            "你是一个意图识别助手。请分析用户的最后一句话，判断用户意图并调用相应的函数。\n\n"
            "- 如果用户使用疑问词（如'怎么'、'为什么'、'如何'）询问退出相关的问题（例如'怎么退出了？'），注意这不是让你退出，请返回 {'function_call': {'name': 'continue_chat'}\n"
            "- 仅当用户明确使用'退出系统'、'结束对话'、'我不想和你说话了'等指令时，才触发 handle_exit_intent\n\n"
            f"{functions_desc}\n"
            "处理步骤:\n"
            "1. 分析用户输入，确定用户意图\n"
            "2. 从可用函数列表中选择最匹配的函数\n"
            "3. 如果找到匹配的函数，生成对应的function_call 格式\n"
            '4. 如果没有找到匹配的函数，返回{"function_call": {"name": "continue_chat"}}\n\n' 
            "返回格式要求：\n"
            "1. 必须返回纯JSON格式\n"
            "2. 必须包含function_call字段\n"
            "3. function_call必须包含name字段\n"
            "4. 如果函数需要参数，必须包含arguments字段\n\n"
            "示例：\n"
            "```\n"
            "用户: 现在几点了？\n"
            '返回: {"function_call": {"name": "get_time"}}\n'
            "```\n"
            "```\n"
            "用户: 当前电池电量是多少？\n"
            '返回: {"function_call": {"name": "get_battery_level", "arguments": {"response_success": "当前电池电量为{value}%", "response_failure": "无法获取Battery的当前电量百分比"}}}
'
            "```\n"
            "```\n"
            "用户: 当前屏幕亮度是多少？\n"
            '返回: {"function_call": {"name": "self_screen_get_brightness"}}\n'
            "```\n"
            "```\n"
            "用户: 设置屏幕亮度为50%\n"
            '返回: {"function_call": {"name": "self_screen_set_brightness", "arguments": {"brightness": 50}}}
'
            "```\n"
            "```\n"
            "用户: 我想结束对话\n"
            '返回: {"function_call": {"name": "handle_exit_intent", "arguments": {"say_goodbye": "goodbye"}}}
'
            "```\n"
            "```\n"
            "用户: 你好啊\n"
            '返回: {"function_call": {"name": "continue_chat"}}\n'
            "```\n\n"
            "注意：\n"
            "1. 只返回JSON格式，不要包含任何其他文字\n"
            '2. 如果没有找到匹配的函数，返回{"function_call": {"name": "continue_chat"}}\n'
            "3. 确保返回的JSON格式正确，包含所有必要的字段\n"
            "特殊说明：\n"
            "- 当用户单次输入包含多个指令时（如'打开灯并且调高音量'）\n"
            "- 请返回多个function_call组成的JSON数组\n"
            "- 示例：{'function_calls': [{name:'light_on'}, {name:'volume_up'}]}"
        )
        return prompt

    def replyResult(self, text: str, original_text: str):
        llm_result = self.llm.response_no_stream(
            system_prompt=text,
            user_prompt="请根据以上内容，像人类一样说话的口吻回复用户，要求简洁，请直接返回结果。用户现在说：" 
            + original_text,
        )
        return llm_result

    async def detect_intent(self, conn, dialogue_history: List[Dict], text: str) -> str:
        session_id = conn.session_id
        if session_id in self.pending_intents:
            pending_intent = self.pending_intents.pop(session_id)
            missing_param_name = pending_intent['missing_params'][0]
            pending_intent['arguments'][missing_param_name] = text
            del pending_intent['missing_params']
            return json.dumps({'function_call': pending_intent})

        if not self.llm:
            raise ValueError("LLM provider not set")
        if conn.func_handler is None:
            return '{"function_call": {"name": "continue_chat"}}'

        total_start_time = time.time()
        model_info = getattr(self.llm, "model_name", str(self.llm.__class__.__name__))
        logger.bind(tag=TAG).debug(f"使用意图识别模型: {model_info}")

        cache_key = hashlib.md5((conn.device_id + text).encode()).hexdigest()
        cached_intent = self.cache_manager.get(self.CacheType.INTENT, cache_key)
        if cached_intent is not None:
            logger.bind(tag=TAG).debug(f"使用缓存的意图: {cached_intent}")
            return cached_intent

        if self.promot == "":
            functions = conn.func_handler.get_functions()
            if hasattr(conn, "mcp_client"):
                mcp_tools = conn.mcp_client.get_available_tools()
                if mcp_tools:
                    if functions is None:
                        functions = []
                    functions.extend(mcp_tools)
            self.promot = self.get_intent_system_prompt(functions)

        music_config = initialize_music_handler(conn)
        music_file_names = music_config["music_file_names"]
        prompt_music = f"{self.promot}\n<musicNames>{music_file_names}\n</musicNames>"

        home_assistant_cfg = conn.config["plugins"].get("home_assistant")
        if home_assistant_cfg:
            devices = home_assistant_cfg.get("devices", [])
        else:
            devices = []
        if devices:
            hass_prompt = "\n下面是我家智能设备列表（位置，设备名，entity_id），可以通过homeassistant控制\n"
            for device in devices:
                hass_prompt += device + "\n"
            prompt_music += hass_prompt

        msgStr = ""
        start_idx = max(0, len(dialogue_history) - self.history_count)
        for i in range(start_idx, len(dialogue_history)):
            msgStr += f"{dialogue_history[i].role}: {dialogue_history[i].content}\n"
        msgStr += f"User: {text}\n"
        user_prompt = f"current dialogue:\n{msgStr}"

        intent = self.llm.response_no_stream(system_prompt=prompt_music, user_prompt=user_prompt)

        intent = intent.strip()
        match = re.search(r'{{.*}}', intent, re.DOTALL)
        if match:
            intent = match.group(0)

        try:
            intent_data = json.loads(intent)
            if "function_call" in intent_data:
                function_data = intent_data["function_call"]
                function_name = function_data.get("name")
                if function_name == "continue_chat":
                    return intent

                provided_args = function_data.get("arguments", {})
                if provided_args is None: provided_args = {}

                all_functions = conn.func_handler.get_functions()
                func_def = next((f.get("function") for f in all_functions if f.get("function", {}).get("name") == function_name), None)

                if func_def:
                    required_params = func_def.get("parameters", {}).get("required", [])
                    missing_params = [p for p in required_params if p not in provided_args or not provided_args.get(p)]

                    if missing_params:
                        self.pending_intents[session_id] = {
                            'name': function_name,
                            'arguments': provided_args,
                            'missing_params': missing_params
                        }
                        params_info = func_def.get("parameters", {}).get("properties", {})
                        param_descs = [params_info.get(p, {}).get("description", p) for p in missing_params]
                        question_prompt = f"我正在尝试执行 '{func_def.get('description', function_name)}'，但我还需要知道以下信息：{', '.join(param_descs)}。请你帮我用友好、简洁、自然的问句，向用户询问这些信息。"
                        question_to_user = self.llm.response_no_stream(
                            system_prompt="你是一个智能助手，你的任务是帮助用户完成操作。",
                            user_prompt=question_prompt
                        )
                        return json.dumps({"function_call": {"name": "ask_user_for_info", "arguments": {"question": question_to_user}}})

                self.cache_manager.set(self.CacheType.INTENT, cache_key, intent)
                return intent
            else:
                self.cache_manager.set(self.CacheType.INTENT, cache_key, intent)
                return intent
        except json.JSONDecodeError:
            logger.bind(tag=TAG).error(f"无法解析意图JSON: {intent}")
            return '{"intent": "继续聊天"}'