import os
import ollama
from types import SimpleNamespace
from .base import AsyncBaseAgent

class AsyncOllamaAgent(AsyncBaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__()
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()
        
        # 设置Ollama API的基础URL，默认为本地
        if os.getenv("OLLAMA_HOST"):
            ollama.set_host(os.getenv("OLLAMA_HOST"))
        
        # 额外的Ollama特定参数设置
        if not hasattr(self.args, 'model'):
            raise ValueError("必须提供模型名称")
        
        # 保存模型名称
        self.model = self.args.model
        
        # 保存额外选项
        self.options = {}
        if hasattr(self.args, 'options'):
            self.options = self.args.options

    def generate(self, prompt, temperature=None, max_tokens=None):
        """使用ollama库生成响应"""
        # 构建参数字典
        params = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": self.args.temperature if temperature is None else temperature,
                "gpu": True,
                **self.options
            }
        }
        
        # 处理最大令牌数
        if max_tokens is not None:
            params["options"]["num_predict"] = max_tokens
        elif hasattr(self.args, 'max_tokens'):
            params["options"]["num_predict"] = self.args.max_tokens
        
        try:
            # 使用ollama库的generate方法
            response = ollama.generate(**params)
            return response
        except Exception as e:
            print(f"Ollama调用失败: {e}")
            return f"Error: {str(e)}"

    def preprocess_input(self, text):
        """预处理输入"""
        return text

    def postprocess_output(self, output):
        """后处理输出"""
        # 如果output是ollama库返回的响应对象
        if isinstance(output, dict) and "response" in output:
            return output["response"]
        
        # 如果output已经是字符串，直接返回
        if isinstance(output, str):
            return output
        
        # 如果是其他类型的响应，尝试提取文本内容
        try:
            return str(output)
        except:
            return "无法处理模型输出。" 