class modelconfig:
    def __init__(self, model, temp, max_tokens):
        if model in ['gpt', 'claude', 'gemini']:
            self.model = model
        else:
            raise ValueError("Invalid model. Choose from 'gpt', 'claude', or 'gemini'.")
        if 0 <= temp <= 2:
            self.temp = temp
        else:
            raise ValueError("Invalid temperature. Choose a value between 0 and 2.")
        if(max_tokens <= 0):
            raise ValueError("Invalid max_tokens. Choose a positive value.")
        self.max_tokens = max_tokens

    def __repr__(self):
        return f"modelconfig(model={self.model}, temp={self.temp}, max_tokens={self.max_tokens})"

model1 = modelconfig(input("Enter model: "), float(input("Enter temperature: ")), int(input("Enter max tokens: ")))
print(repr(model1))
print(model1)