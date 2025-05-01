# from .claude import AsyncClaudeAgent
from .gemini import AsyncGeminiAgent
# from .gpt import GPT3BaseAgent, AsyncConversationalGPTBaseAgent
# from .huggingface import ZephyrAgent
# from .together_ai import AsyncTogetherAIAgent, AsyncLlama3Agent
# from .custom_openai import CustomOpenAIAgent
from .ollama import AsyncOllamaAgent
from .huggingface import BitNetAgent, NemotronAgent, Phi4MiniInstructAgent
from dotenv import load_dotenv
import os

load_dotenv()

def load_model(model_name, **kwargs):
    # if model_name.startswith("text-"):
    #     model = GPT3BaseAgent({'engine': model_name, 'temperature': 0, 'top_p': 1.0, 'frequency_penalty': 0.0, 'presence_penalty': 0.0})
    # elif model_name.startswith("gpt-"):
    #     # 如果提供了自定义API基础地址，使用CustomOpenAIAgent
    #     if 'api_base' in kwargs:
    #         model = CustomOpenAIAgent({'model': model_name, 'temperature': 0, 'top_p': 1.0, 
    #                                   'frequency_penalty': 0.0, 'presence_penalty': 0.0, **kwargs})
    #     else:
    #         model = AsyncConversationalGPTBaseAgent({'model': model_name, 'temperature': 0, 'top_p': 1.0, 
    #                                                'frequency_penalty': 0.0, 'presence_penalty': 0.0, **kwargs})
    # elif model_name == "custom-openai":
    #     # 直接使用自定义OpenAI代理
    #     model = CustomOpenAIAgent({'api_base': kwargs.get('api_base'), 'api_key': os.getenv('API_KEY')})
    # elif model_name.startswith('gemini-'):
    #     model = AsyncGeminiAgent({'model': model_name, 'temperature': 0, 'max_tokens': 256})
    # elif model_name.startswith('claude-'):
    #     model = AsyncClaudeAgent({'model': model_name, **kwargs})
    # elif model_name in ["meta-llama/Llama-3-70b-chat-hf-tg", "meta-llama/Llama-3-8b-chat-hf-tg"]:
    #     model = AsyncLlama3Agent({'model': model_name, 'temperature': 0, 'max_tokens': 256, **kwargs})
    # elif model_name.endswith('-tg'):
    #     model = AsyncTogetherAIAgent({'model': model_name.removesuffix("-tg"), 'temperature': 0, 'max_tokens': 128, **kwargs})
    # elif model_name.startswith('zephyr'):
    #     model = ZephyrAgent(**kwargs)
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
    else:
        raise NotImplementedError

    return model