import re

class PromptTemplate:

    def __init__(self, template, defaults=None):
        self.template = template
        self.defaults = defaults or {}

    def format(self, **variables):

        # Find {{variable}} placeholders
        placeholders = re.findall(r"\{\{(\w+)\}\}", self.template)

        result = self.template

        for name in placeholders:

            # Value provided by user
            if name in variables:
                value = variables[name]

            # Otherwise use default value
            elif name in self.defaults:
                value = self.defaults[name]

            # No value and no default
            else:
                raise ValueError(f"Missing required variable: {name}")

            result = result.replace("{{" + name + "}}", str(value))

        return result
# Testing
template = PromptTemplate(
    "Explain {{topic}} in a {{style}} way.",
    defaults={"style": "simple"}
)

print(template.format(topic="Machine Learning"))

print(template.format(
    topic="Neural Networks",
    style="technical"
))