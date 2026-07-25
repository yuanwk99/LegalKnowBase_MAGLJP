# encoding=utf-8
import requests
import json
import os
from http import HTTPStatus
import time


def ask_llm(prompt, model='qwen-max'):
    return ask_tyqw(prompt, model)


def ask_tyqw(prompt, model='qwen-max'):
    try:
        import dashscope
        from dashscope import Generation
    except ImportError as exc:
        raise ImportError(
            "dashscope is required only for online LLM calls. "
            "Install it with `pip install dashscope` before running graph generation."
        ) from exc

    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "Your-Key")
    if type(prompt) is str:
        s_time = time.time()
        response = Generation.call(model,
                                   prompt=prompt
                                   )
        e_time = time.time()

        
        # print(response)
        if response.status_code == HTTPStatus.OK:
            used_time = e_time-s_time
            input_tokens = response['usage'].input_tokens
            output_tokens = response['usage'].output_tokens
            token_infomration = {"used_time":used_time,"input_tokens":input_tokens,"output_tokens":output_tokens}
            return response["output"]["text"],token_infomration
                
        
        return None,{"used_time":None,"input_tokens":None,"output_tokens":None}

    elif type(prompt) is list:
        response = Generation.call(model,
                                   messages=prompt,
                                   result_format='message'  # 设置输出为'message'格式
                                   )
        if response.status_code == HTTPStatus.OK:
            return response["output"]["choices"][0]["message"]["content"]
        else:
            return None
