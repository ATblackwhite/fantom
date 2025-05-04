from modelscope import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from .base import BaseAgent

class DeepSeekAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__()

        # 获取设备信息
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.max_memory = 15 * 1024**3  # 20GB安全阈值
        self.batch_size = 1  # 根据显存动态调整

        # 简化参数设置
        self.temperature = kwargs.get('temperature', 0.1)
        self.max_tokens = kwargs.get('max_tokens', 10240)
        self.top_p = kwargs.get('top_p', 0.95)

        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.pipe = pipeline(
            task='text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            **kwargs
        )

        # 设置对话模板
        self.chat_template = [
            {"role": "system", "content": "You are a helpful assistant focusing on Situational understanding, give me the answer directly without reasoning"},
            {"role": "user", "content": "\n{content}\n"}
        ]

    def preprocess_input(self, text):
        """将输入文本转换为适合模型的格式"""
        formatted_prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant focusing on Situational understanding"},
                {"role": "user", "content": text}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        return self.tokenizer([formatted_prompt], return_tensors="pt").to(self.model.device)

    def postprocess_output(self, output):
        pass

    def generate(self, prompt, temperature=None, max_tokens=None):
        raise NotImplementedError

    def _check_memory(self):
        """实时监控显存使用"""
        used = torch.cuda.memory_allocated()
        if used > self.max_memory:
            self.batch_size = max(1, self.batch_size // 2)
            torch.cuda.empty_cache()

    def interact(self, prompt):
        self._check_memory()
        pre = self.preprocess_input(prompt)

        generated_ids = self.model.generate(
            **pre,
            max_new_tokens=2048,
            temperature = 0.1,
            # 新增防循环参数
            repetition_penalty=1.2,  # 重复惩罚系数（1.0表示无惩罚）
            no_repeat_ngram_size=3,   # 禁止3-gram重复
            do_sample=True,          # 启用采样模式
            top_k=50,                # 限制采样候选词数量
            top_p=0.95               # 核采样概率阈值
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(pre.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self._check_memory()

        try:
            return response.split('</think>')[1].strip()
        except (AttributeError, IndexError):
            return response.strip()

    def batch_generate(self, prompts, temperature=None, max_tokens=None):
        """批量生成回复"""
        return [self.generate(prompt, temperature, max_tokens) for prompt in prompts]

    def batch_interact(self, texts):
        raise NotImplementedError


class Qwen(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__()

        # 获取设备信息
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.max_memory = 20 * 1024**3  # 20GB安全阈值
        self.batch_size = 1  # 根据显存动态调整

        # 简化参数设置
        self.temperature = kwargs.get('temperature', 0.1)
        self.max_tokens = kwargs.get('max_tokens', 1024)
        self.top_p = kwargs.get('top_p', 0.95)

        model_name = "Qwen/Qwen2.5-7B-Instruct"
        if kwargs['size'] == 'small':
            model_name = "Qwen/Qwen2.5-3B-Instruct"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.pipe = pipeline(
            task='text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            **kwargs
        )

        # 设置对话模板
        self.chat_template = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant focusing on Situational understanding"},
            {"role": "user", "content": "\n{content}\n"}
        ]

    def preprocess_input(self, text):
        """将输入文本转换为适合模型的格式"""
        formatted_prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant focusing on Situational understanding"},
                {"role": "user", "content": text}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        return self.tokenizer([formatted_prompt], return_tensors="pt").to(self.model.device)

    def postprocess_output(self, output):
        pass

    def generate(self, prompt, temperature=None, max_tokens=None):
        raise NotImplementedError

    def _check_memory(self):
        """实时监控显存使用"""
        used = torch.cuda.memory_allocated()
        if used > self.max_memory:
            self.batch_size = max(1, self.batch_size // 2)
            torch.cuda.empty_cache()

    def interact(self, prompt):
        self._check_memory()
        pre = self.preprocess_input(prompt)

        generated_ids = self.model.generate(
            **pre,
            max_new_tokens=512
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(pre.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self._check_memory()

        return response

    def batch_generate(self, prompts, temperature=None, max_tokens=None):
        """批量生成回复"""
        return [self.generate(prompt, temperature, max_tokens) for prompt in prompts]

    def batch_interact(self, texts):
        raise NotImplementedError
