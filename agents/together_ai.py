import os
import together
from types import SimpleNamespace
from .base import BaseAgent, AsyncBaseAgent

class TogetherAIAgent(BaseAgent):
    def __init__(self, kwargs: dict):
        self.api_key = together.api_key = os.getenv('TOGETHERAI_API_KEY')
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()
        self.args.model = self.args.model.removesuffix("-tg")

    def _set_default_args(self):
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 1.0
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 256
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 1.0
        if not hasattr(self.args, 'frequency_penalty'):
            self.args.frequency_penalty = 0
        if not hasattr(self.args, 'presence_penalty'):
            self.args.presence_penalty = 0
    
    def generate(self, prompt, temperature=None, max_tokens=None):
        output = together.Complete.create(
            prompt=prompt,
            model=self.args.model,
            max_tokens = self.args.max_tokens if max_tokens is None else max_tokens,
            temperature = self.args.temperature if temperature is None else temperature,
        )

        return output

    def postprocess_output(self, output):
        responses = [c['text'].strip() for c in output['output']['choices']]
        return responses[0]

    def preprocess_input(self, text):
        return text

    def interact(self, prompt, temperature=None, max_tokens=None):
        output = self.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        response = self.postprocess_output(output)

        return response

class AsyncTogetherAIAgent(AsyncBaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__()
        self.api_key = together.api_key = os.getenv('TOGETHERAI_API_KEY')
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()
        self.args.model = self.args.model.removesuffix("-tg")

    def generate(self, prompt, temperature=None, max_tokens=None):
        output = together.Complete.create(
            prompt=prompt,
            model=self.args.model,
            max_tokens = self.args.max_tokens if max_tokens is None else max_tokens,
            temperature = self.args.temperature if temperature is None else temperature,
        )

        return output

    def postprocess_output(self, output):
        responses = [c['text'].strip() for c in output['output']['choices']]
        return responses[0]

    def preprocess_input(self, text):
        return text
