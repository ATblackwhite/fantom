import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .base import BaseAgent
import os

class HuggingFaceAgent(BaseAgent):
    def __init__(self, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 8 #kwargs['batch_size']

    def init_pipeline(self):
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
        )
        self.pipe.tokenizer.padding_side = "left"

    def preprocess_input(self, text):
        return text

    def postprocess_output(self, response):
        return response

    def postprocess_pipeline_output(self, output):
        return output[0]['generated_text'].strip()

    def encode(self, texts):
        prompts = [self.preprocess_input(text) for text in texts]
        encoded_texts = self.tokenizer(prompts, add_special_tokens=False, max_length=512, padding='max_length', truncation=True, return_tensors="pt").to(self.model.device)
        return encoded_texts

    def decode(self, outputs):
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        responses = [self.postprocess_output(decoded_output) for decoded_output in decoded_outputs]
        return responses

    def raw_batch_interact(self, texts, do_sample=True):
        encoded_texts = self.encode(texts)
        with torch.no_grad():
            outputs = self.model.generate(**encoded_texts, max_new_tokens=365, do_sample=do_sample)
        responses = self.decode(outputs)
        return responses

    def batch_interact(self, texts, do_sample=True):
        prompts = [self.preprocess_input(text) for text in texts] # XXX: apply those chat-specific templates beforehand and make them into pipeline batch and directly feed the pipeline
        outputs = self.pipe(prompts, return_full_text=False, max_new_tokens=365, do_sample=True)
        responses = [self.postprocess_pipeline_output(output) for output in outputs]

        return responses

    def interact(self, text, do_sample=True):
        return self.batch_interact([text], do_sample)[0]

    def batch_compute_likelihood(self, input_texts, target_data):
        """ Compute the log-likelihood of the target data given the input text. """
        # We should pad after concatenating with target_outputs
        prompts = [self.preprocess_input(text) for text in input_texts] # apply those chat-specific templates
        data_appended_prompt = [p + d for p, d in zip(prompts, target_data)] # append the target responses to the prompts
        encoded_texts = self.tokenizer(data_appended_prompt, add_special_tokens=False, max_length=512, padding='max_length', truncation=True, return_tensors="pt").to(self.model.device)
        encoded_data = self.tokenizer(target_data, add_special_tokens=False, max_length=512, padding='max_length', truncation=True, return_tensors="pt").to(self.model.device) # this is actually for getting the attention mask to know which part of the input is the response

        with torch.no_grad():
            outputs = self.model(**encoded_texts, return_dict=True)

        vocab_distribution = torch.log_softmax(outputs.logits, dim=-1)
        data_token_logprobs = torch.gather(vocab_distribution[:,:-1,:], 2, encoded_data.input_ids.unsqueeze(-1)[:,1:,:])
        true_data_token_logprobs = (data_token_logprobs * encoded_data.attention_mask.unsqueeze(-1)[:, 1:, :]).squeeze(-1) # get only the logprobs of the response tokens
        data_log_likelihood = true_data_token_logprobs.sum(dim=1) / encoded_data.attention_mask.sum(dim=1)

        return data_log_likelihood
    
    def compute_data_likelihood(self, input_text, target_datum):
        return self.batch_compute_likelihood([input_text], [target_datum])[0]


class HuggingFaceChatAgent(HuggingFaceAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_output_token = "[/INST]"

    def preprocess_input(self, text):
        messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": text},
        ]
        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return chat_prompt

    def postprocess_output(self, response):
        return response.split(self.model_output_token)[-1].strip()

class Llama2Agent(HuggingFaceAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert kwargs['model_size'].lower() in ['7b', '13b', '70b']
        self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-2-{kwargs['model_size'].lower()}-hf", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-2-{kwargs['model_size'].lower()}-hf", device_map="auto",
                                                          torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.tokenizer.pad_token = self.tokenizer.eos_token # LLaMa tokenizer has no pad token
        self.init_pipeline()
        self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id

class Llama2ChatAgent(HuggingFaceChatAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert kwargs['model_size'].lower() in ['7b', '13b', '70b']
        self.tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-2-{kwargs['model_size'].lower()}-chat-hf", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-2-{kwargs['model_size'].lower()}-chat-hf", device_map="auto",
                                                          torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.tokenizer.pad_token = self.tokenizer.eos_token # LLaMa tokenizer has no pad token
        self.model_output_token = "[/INST]"
        self.init_pipeline()
        self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id

class NemotronAgent(HuggingFaceAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 设置模型缓存路径
        cache_dir = os.path.join(os.getcwd(), "data", "models", "nemotron")
        
        # 创建目录（如果不存在）
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            # 初始化tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "nvidia/Nemotron-H-8B-Base-8K", 
                cache_dir=cache_dir,
                trust_remote_code=True,
                padding_side='left'
            )
            
            # 针对8GB显存优化的加载选项
            # 使用bfloat16数据类型
            # 使用4位量化（需安装bitsandbytes库: pip install bitsandbytes）
            # 如无bitsandbytes，则移除load_in_4bit相关选项
            
            print("正在加载Nemotron-H-8B模型，针对8GB显存进行优化...")
            
            try:
                # 首先尝试4位量化加载
                import bitsandbytes as bnb
                self.model = AutoModelForCausalLM.from_pretrained(
                    "nvidia/Nemotron-H-8B-Base-8K",
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    load_in_4bit=True,
                    low_cpu_mem_usage=True,
                    use_cache=True
                )
                print("成功以4位量化方式加载模型")
            except (ImportError, ModuleNotFoundError):
                # 如果无法使用4位量化，尝试8位量化
                self.model = AutoModelForCausalLM.from_pretrained(
                    "nvidia/Nemotron-H-8B-Base-8K",
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    load_in_8bit=True,
                    low_cpu_mem_usage=True,
                    use_cache=True
                )
                print("成功以8位量化方式加载模型")
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 修改batch_size以适应较小显存
            self.batch_size = 1
            
            # 初始化pipeline
            try:
                self.init_pipeline()
                if hasattr(self, 'pipe') and self.pipe is not None:
                    self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id
                    print("Pipeline初始化成功")
                else:
                    print("Pipeline未成功初始化，将使用生成方法直接调用模型")
            except Exception as e:
                print(f"初始化pipeline失败: {e}")
                print("将使用generate方法直接调用模型")
                
        except Exception as e:
            print(f"加载Nemotron模型失败: {e}")
            print("可能需要更多显存或尝试不同的优化方法")
    
    def generate(self, prompt, temperature=0.7, max_tokens=None):
        """实现直接生成方法，以防pipeline不可用"""
        try:
            # 如果pipeline可用，使用pipeline
            if hasattr(self, 'pipe') and self.pipe is not None:
                return self.interact(prompt, do_sample=(temperature > 0))
            
            # 否则直接使用模型生成
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            
            generation_config = {
                "max_new_tokens": 365 if max_tokens is None else max_tokens,
                "do_sample": temperature > 0,
            }
            
            if temperature > 0:
                generation_config["temperature"] = temperature
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            return f"生成失败: {str(e)}"
    
    def interact(self, text, do_sample=True):
        """覆盖基类的interact方法，增加错误处理"""
        try:
            return super().interact(text, do_sample)
        except Exception as e:
            print(f"使用pipeline交互失败: {e}")
            print("回退到直接生成方法")
            return self.generate(text, temperature=0.7 if do_sample else 0)

class MistralAgent(HuggingFaceAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto",
                                                          torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.tokenizer.pad_token = self.tokenizer.eos_token # LLaMa tokenizer has no pad token
        # self.init_pipeline()
        # self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id

class MistralInstructAgent(HuggingFaceChatAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map="auto",
                                                          torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.tokenizer.pad_token = self.tokenizer.eos_token # LLaMa tokenizer has no pad token
        self.model_output_token = "[/INST]"
        self.init_pipeline()
        self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id

class MixtralInstructAgent(HuggingFaceChatAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", padding_size='left')
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", device_map="auto",
                                                          torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.tokenizer.pad_token = self.tokenizer.eos_token # LLaMa tokenizer has no pad token
        self.model_output_token = "[/INST]"

class ZephyrAgent(HuggingFaceChatAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", device_map="auto",
                                                          torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.model_output_token = "<|assistant|>"
        self.init_pipeline()

class GemmaAgent(HuggingFaceAgent):
    def __init__(self, **kwargs):
        assert kwargs['model_size'].lower() in ['2b', '7b']
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-{kwargs['model_size'].lower()}", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(f"google/gemma-{kwargs['model_size'].lower()}", device_map="auto",
                                                          torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.init_pipeline()

class GemmaInstructAgent(HuggingFaceChatAgent):
    def __init__(self, **kwargs):
        assert kwargs['model_size'].lower() in ['2b', '7b']
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-{kwargs['model_size'].lower()}-it", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(f"google/gemma-{kwargs['model_size'].lower()}-it", device_map="auto",
                                                          torch_dtype=torch.float16, attn_implementation="flash_attention_2")
        self.model_output_token = "\nmodel\n"
        self.init_pipeline()

class BitNetAgent(HuggingFaceChatAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 使用pip安装所需的fork版的transformers
        # pip install git+https://github.com/shumingma/transformers.git
        
        # # 设置模型缓存路径为data/models
        # cache_dir = os.path.join(os.getcwd(), "data", "models", "bitnet")
        
        # # 创建目录（如果不存在）
        # os.makedirs(cache_dir, exist_ok=True)
        
        # # 初始化tokenizer (使用LLaMA 3 tokenizer)
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     "microsoft/bitnet-b1.58-2B-4T", 
        #     cache_dir=cache_dir
        # )
        
        # # 设置环境变量以避免某些优化
        # os.environ["PYTORCH_TRITON"] = "0"  # 禁用PyTorch中的Triton使用
        # os.environ["TORCH_COMPILE_DISABLE_CUDA_GRAPH"] = "1"
        
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     "microsoft/bitnet-b1.58-2B-4T",
        #     device_map=None,  # 初始加载到CPU
        #     torch_dtype=torch.bfloat16,
        #     low_cpu_mem_usage=True,
        #     cache_dir=cache_dir,
        #     use_cache=True,
        #     local_files_only=True
        # )


        
        # 设置模型缓存路径为data/models
        cache_dir = os.path.join(os.getcwd(), "data", "models", "bitnetrun")
        
        # 创建目录（如果不存在）
        os.makedirs(cache_dir, exist_ok=True)
  
        # 初始化tokenizer (使用LLaMA 3 tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "rakmik/bitnetrun", 
            cache_dir=cache_dir
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "rakmik/bitnetrun",
            device_map=None,  # 初始加载到CPU
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            use_cache=True,
        )
        
        # 检查是否有可用的GPU
        if torch.cuda.is_available():
            # 手动将模型移到GPU
            try:
                device = torch.device("cuda")
                self.model = self.model.to(device)
                print("成功将模型加载到GPU")
            except Exception as e_gpu:
                print(f"移动模型到GPU失败: {e_gpu}")
                print("继续使用CPU")
        else:
            print("没有可用的GPU，使用CPU运行")
            
        
        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 设置model_output_token，用于回复后处理
        # 根据BitNet使用的是LLaMA 3风格的对话模板
        self.model_output_token = "[/INST]"
        
        # 初始化pipeline，仅当模型成功加载时
        if self.model is not None:
            try:
                self.init_pipeline()
                if hasattr(self, 'pipe') and self.pipe is not None:
                    self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id
                    print("Pipeline初始化成功")
                else:
                    print("Pipeline未成功初始化")
            except Exception as e:
                print(f"初始化pipeline失败: {e}")
                print("将使用generate方法直接调用模型")
        else:
            print("模型未加载，跳过pipeline初始化")
    
    def generate(self, prompt, temperature=None, max_tokens=None):
        """实现BaseAgent抽象类要求的generate方法
        
        Args:
            prompt: 输入提示文本
            temperature: 采样温度，控制输出随机性
            max_tokens: 生成的最大token数
            
        Returns:
            生成的文本回复
        """
        # 应用预处理到输入文本
        processed_prompt = self.preprocess_input(prompt)
        
        # 设置生成参数
        generation_kwargs = {
            "return_full_text": False,
            "max_new_tokens": 365 if max_tokens is None else max_tokens,
            "do_sample": True if temperature and temperature > 0 else False,
        }
        
        # 添加温度参数（如果提供）
        if temperature is not None and temperature > 0:
            generation_kwargs["temperature"] = temperature
        
        # 使用pipeline生成文本
        output = self.pipe(processed_prompt, **generation_kwargs)
        
        # 返回生成的第一个结果
        return self.postprocess_pipeline_output(output)
        
    def preprocess_input(self, text):
        """预处理用户输入为BitNet模型适用的格式"""
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": text},
        ]
        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return chat_prompt

class Phi4MiniInstructAgent(HuggingFaceChatAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 设置模型缓存路径
        cache_dir = os.path.join(os.getcwd(), "data", "models", "phi4-mini")
        
        # 创建目录（如果不存在）
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-4-mini-instruct", 
            cache_dir=cache_dir,
            padding_side='left'
        )
        
        # 根据GPU情况选择加载方式
        # 默认使用flash_attention_2加速
        attn_implementation = "flash_attention_2"
        
        # 对于V100或更早的GPU，使用eager模式
        # 如果遇到错误，可以修改为eager模式
        # attn_implementation = "eager" 
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-4-mini-instruct",
            device_map="auto",
            torch_dtype=torch.bfloat16,  # 使用BF16格式
            attn_implementation=attn_implementation,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True
        )
        
        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 设置model_output_token，用于回复后处理
        self.model_output_token = "[/INST]"
        
        # 初始化pipeline
        self.init_pipeline()
        self.pipe.tokenizer.pad_token_id = self.model.config.eos_token_id
    
    def preprocess_input(self, text):
        """预处理用户输入为Phi-4模型适用的格式"""
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": text},
        ]
        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return chat_prompt

    def generate(self, prompt, temperature=0.7, max_tokens=None):
        """实现生成方法，支持温度和最大令牌数参数"""
        # 应用预处理到输入文本
        processed_prompt = self.preprocess_input(prompt)
        
        # 设置生成参数
        generation_kwargs = {
            "return_full_text": False,
            "max_new_tokens": 365 if max_tokens is None else max_tokens,
            "do_sample": temperature > 0,
        }
        
        # 添加温度参数（如果需要）
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        
        # 使用pipeline生成文本
        try:
            output = self.pipe(processed_prompt, **generation_kwargs)
            return self.postprocess_pipeline_output(output)
        except Exception as e:
            print(f"使用pipeline生成失败: {e}")
            # 回退到直接使用模型生成
            inputs = self.tokenizer(processed_prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=generation_kwargs["max_new_tokens"],
                    do_sample=generation_kwargs["do_sample"],
                    temperature=temperature if temperature > 0 else 1.0
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self.postprocess_output(generated_text)

class FlanT5Agent:
    def __init__(self, kwargs: dict):
        self.model_name = kwargs.get("model", "google/flan-t5-small")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 128)

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=self.max_tokens,
            temperature=self.temperature,
            num_return_sequences=1,
            do_sample=True,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def interact(self, text):
        """
        Single interaction method.
        """
        return self.generate(text)

    def batch_interact(self, texts):
        """
        Batch interaction method.
        """
        responses = [self.generate(text) for text in texts]
        return responses

class FlanT5BaseAgent:
    def __init__(self, kwargs: dict):
        self.model_name = kwargs.get("model", "google/flan-t5-base")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 128)

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=self.max_tokens,
            temperature=self.temperature,
            num_return_sequences=1,
            do_sample=True,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def interact(self, text):
        """
        Single interaction method.
        """
        return self.generate(text)

    def batch_interact(self, texts):
        """
        Batch interaction method.
        """
        responses = [self.generate(text) for text in texts]
        return responses

from transformers import AutoTokenizer, AutoModelForCausalLM

class DeepSeekAgent:
    def __init__(self, kwargs: dict):
        self.model_name = kwargs.get("model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch.float32)  # 改为 float32
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 128)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 设置填充标记为结束标记

    def generate(self, prompt):
        """
        Generate a single response for the given prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=self.max_tokens,
            temperature=self.temperature,
            num_return_sequences=1,
            do_sample=True,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def interact(self, text):
        """
        Single interaction method.
        """
        return self.generate(text)

    def batch_interact(self, texts):
        """
        Batch interaction method.
        """
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=self.max_tokens,
            temperature=self.temperature,
            num_return_sequences=1,
            do_sample=True,
        )
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return responses

from transformers import AutoTokenizer, AutoModelForCausalLM

class DistilGPT2Agent:
    def __init__(self, kwargs: dict):
        self.model_name = kwargs.get("model", "distilbert/distilgpt2")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", torch_dtype=torch.float32)
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 128)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 设置填充标记为结束标记

    def generate(self, prompt):
        """
        Generate a single response for the given prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=self.max_tokens,
            temperature=self.temperature,
            num_return_sequences=1,
            do_sample=True,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def interact(self, text):
        """
        Single interaction method.
        """
        return self.generate(text)

    def batch_interact(self, texts):
        """
        Batch interaction method.
        """
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=self.max_tokens,
            temperature=self.temperature,
            num_return_sequences=1,
            do_sample=True,
        )
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return responses

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class DistilBertAgent:
    def __init__(self, kwargs: dict):
        self.model_name = kwargs.get("model", "distilbert-base-cased")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        
        # 设置 pad_token 为 eos_token 或自定义 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, torch_dtype=torch.float32)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # 手动将模型移动到设备
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 128)

    def generate(self, prompt):
        """
        Generate a single response for the given prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(dim=-1).item()
        return f"Predicted class: {predicted_class}"

    def interact(self, text):
        """
        Single interaction method.
        """
        return self.generate(text)

    def batch_interact(self, texts):
        """
        Batch interaction method.
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_classes = logits.argmax(dim=-1).tolist()
        return [f"Predicted class: {cls}" for cls in predicted_classes]