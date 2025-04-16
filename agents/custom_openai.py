import os
import time
import asyncio
import openai
from openai import OpenAI, AsyncOpenAI
from types import SimpleNamespace
from .base import AsyncBaseAgent


class CustomOpenAIAgent(AsyncBaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__()
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()
        
        # 获取API密钥
        api_key = os.getenv('OPENAI_API_KEY')
        if hasattr(self.args, 'api_key') and self.args.api_key:
            api_key = self.args.api_key
            
        # 初始化客户端，允许自定义API基础地址
        if hasattr(self.args, 'api_base') and self.args.api_base:
            self.client = OpenAI(api_key=api_key, base_url=self.args.api_base)
            self.async_client = AsyncOpenAI(api_key=api_key, base_url=self.args.api_base)
        else:
            self.client = OpenAI(api_key=api_key)
            self.async_client = AsyncOpenAI(api_key=api_key)

    def _set_default_args(self):
        super()._set_default_args()
        if not hasattr(self.args, 'model'):
            self.args.model = "gpt-3.5-turbo"
        if not hasattr(self.args, 'response_format'):
            self.args.response_format = None

    def preprocess_input(self, text):
        """预处理输入文本"""
        return text

    def postprocess_output(self, outputs):
        """处理模型输出"""
        try:
            responses = [c.message.content.strip() for c in outputs.choices]
            return responses[0] if responses else ""
        except AttributeError:
            # 处理完成和流式响应格式之间的差异
            if hasattr(outputs, 'choices') and outputs.choices:
                if hasattr(outputs.choices[0], 'text'):
                    return outputs.choices[0].text.strip()
            return ""

    def generate(self, prompt, temperature=None, max_tokens=None):
        """生成文本响应"""
        messages = [{"role": "user", "content": prompt}]
        
        # 设置响应格式（如JSON）
        response_format = None
        if hasattr(self.args, 'response_format') and self.args.response_format:
            if self.args.response_format == "json":
                response_format = {"type": "json_object"}
                messages = [
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ]
        
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.args.model,
                    messages=messages,
                    temperature=self.args.temperature if temperature is None else temperature,
                    max_tokens=self.args.max_tokens if max_tokens is None else max_tokens,
                    top_p=self.args.top_p,
                    frequency_penalty=self.args.frequency_penalty, 
                    presence_penalty=self.args.presence_penalty,
                    response_format=response_format
                )
                break
            except (openai.RateLimitError, openai.APIError, openai.APIConnectionError) as e:
                print(f"请求错误: {e}")
                time.sleep(2)
                continue
        
        return completion

    async def async_generate(self, prompt, temperature=None, max_tokens=None):
        """异步生成文本响应"""
        messages = [{"role": "user", "content": prompt}]
        
        # 设置响应格式（如JSON）
        response_format = None
        if hasattr(self.args, 'response_format') and self.args.response_format:
            if self.args.response_format == "json":
                response_format = {"type": "json_object"}
                messages = [
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ]
        
        while True:
            try:
                completion = await self.async_client.chat.completions.create(
                    model=self.args.model,
                    messages=messages,
                    temperature=self.args.temperature if temperature is None else temperature,
                    max_tokens=self.args.max_tokens if max_tokens is None else max_tokens,
                    top_p=self.args.top_p,
                    frequency_penalty=self.args.frequency_penalty,
                    presence_penalty=self.args.presence_penalty,
                    response_format=response_format
                )
                break
            except (openai.RateLimitError, openai.APIError, openai.APIConnectionError) as e:
                print(f"请求错误: {e}")
                time.sleep(2)
                continue
        
        return completion

    async def batch_generate(self, prompts, temperature=None, max_tokens=None):
        """批量异步生成文本响应"""
        tasks = [self.async_generate(prompt, temperature, max_tokens) for prompt in prompts]
        return await asyncio.gather(*tasks)

    def interact_with_history(self, prompt, history=None, temperature=None, max_tokens=None):
        """支持历史对话的交互"""
        messages = []
        if history is not None:
            for idx, msg in enumerate(history):
                if idx % 2 == 0:
                    messages.append({"role": "user", "content": f"{msg}"})
                else:
                    messages.append({"role": "assistant", "content": f"{msg}"})
        messages.append({"role": "user", "content": f"{prompt}"})
        
        # 设置响应格式（如JSON）
        response_format = None
        if hasattr(self.args, 'response_format') and self.args.response_format:
            if self.args.response_format == "json":
                response_format = {"type": "json_object"}
                # 添加系统提示
                messages.insert(0, {"role": "system", "content": "You are a helpful assistant designed to output JSON."})
        
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.args.model,
                    messages=messages,
                    temperature=self.args.temperature if temperature is None else temperature,
                    max_tokens=self.args.max_tokens if max_tokens is None else max_tokens,
                    top_p=self.args.top_p,
                    frequency_penalty=self.args.frequency_penalty,
                    presence_penalty=self.args.presence_penalty,
                    response_format=response_format
                )
                break
            except (openai.RateLimitError, openai.APIError, openai.APIConnectionError) as e:
                print(f"请求错误: {e}")
                time.sleep(2)
                continue
        
        return completion
    
    def interact_with_history(self, prompt, history=None, temperature=None, max_tokens=None):
        """支持历史对话的交互"""
        messages = []
        if history is not None:
            for idx, msg in enumerate(history):
                if idx % 2 == 0:
                    messages.append({"role": "user", "content": f"{msg}"})
                else:
                    messages.append({"role": "assistant", "content": f"{msg}"})
        messages.append({"role": "user", "content": f"{prompt}"})
        
        # 设置响应格式（如JSON）
        response_format = None
        if hasattr(self.args, 'response_format') and self.args.response_format:
            if self.args.response_format == "json":
                response_format = {"type": "json_object"}
                # 添加系统提示
                messages.insert(0, {"role": "system", "content": "You are a helpful assistant designed to output JSON."})
        
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.args.model,
                    messages=messages,
                    temperature=self.args.temperature if temperature is None else temperature,
                    max_tokens=self.args.max_tokens if max_tokens is None else max_tokens,
                    top_p=self.args.top_p,
                    frequency_penalty=self.args.frequency_penalty,
                    presence_penalty=self.args.presence_penalty,
                    response_format=response_format
                )
                break
            except (openai.RateLimitError, openai.APIError, openai.APIConnectionError) as e:
                print(f"请求错误: {e}")
                time.sleep(2)
                continue
        
        return completion
        
    def interact(self, prompt, temperature=None, max_tokens=None):
        """实现基类的interact方法"""
        output = self.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        response = self.postprocess_output(output)
        return response
        
    def batch_interact(self, prompts, temperature=None, max_tokens=None):
        """批量处理多个提示并返回响应"""
        while True:
            try:
                outputs = asyncio.run(self.batch_generate(prompts, temperature=temperature, max_tokens=max_tokens))
                responses = [self.postprocess_output(output) for output in outputs]
                return responses
            except Exception as e:
                print(f"批处理错误: {e}")
                time.sleep(2)
                continue