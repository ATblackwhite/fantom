from .gemini import AsyncGeminiAgent
from .ollama import AsyncOllamaAgent
from .huggingface import BitNetAgent, NemotronAgent, Phi4MiniInstructAgent, FlanT5Agent, FlanT5BaseAgent, DeepSeekAgent, DistilGPT2Agent, DistilBertAgent
from dotenv import load_dotenv
from .deepseek import DeepSeekAgent
from .qwen import Qwen
import os

load_dotenv()

def load_model(model_name, **kwargs):
    if model_name.startswith('gemini-'):
        model = AsyncGeminiAgent({'model': model_name, 'temperature': 0, 'max_tokens': 256})
    elif model_name.startswith('ollama-'):
        # 使用ollama:前缀来标识Ollama模型，如ollama:llama3
        ollama_model_name = model_name.split('-', 1)[1:]
        ollama_model_name = '-'.join(ollama_model_name)

        # 如果在kwargs中提供了host，设置环境变量
        if 'host' in kwargs:
            os.environ['OLLAMA_HOST'] = kwargs.pop('host')

        model = AsyncOllamaAgent({
            'model': ollama_model_name,
            'temperature': kwargs.get('temperature', 0),
            'max_tokens': kwargs.get('max_tokens', 512),
            'options': kwargs.get('options', {})
        })
    elif model_name.startswith('bitnet-'):
        model = BitNetAgent(**kwargs)
    elif model_name.startswith('Nemotron-'):
        model = NemotronAgent(**kwargs)
    elif model_name.startswith('phi4-'):
        model = Phi4MiniInstructAgent(**kwargs)
    elif model_name == "google/flan-t5-small":
        model = FlanT5Agent({'model': model_name, **kwargs})
    elif model_name == "google/flan-t5-base":
        model = FlanT5BaseAgent({'model': model_name, **kwargs})
    elif model_name == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
        model = DeepSeekAgent({'model': model_name, **kwargs})
    elif model_name == "distilbert/distilgpt2":
        model = DistilGPT2Agent({'model': model_name, **kwargs})
    elif model_name == "distilbert-base-cased":
        model = DistilBertAgent({'model': model_name, **kwargs})
    elif model_name.startswith('text-qwen-small'):
        model = Qwen({'size': 'small', **kwargs})
    elif 'qwen' in model_name:
        model = Qwen({'size': 'normal', **kwargs})
    elif 'deepseek' in model_name:
        model = DeepSeekAgent(**kwargs)
    else:
        raise NotImplementedError

    return model
