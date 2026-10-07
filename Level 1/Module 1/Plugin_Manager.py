"""              ┌──────────────┐
                 │     USER     │
                 └──────┬───────┘
                        │
                        │ request
                        ▼
              ┌───────────────────┐
              │   PluginManager   │
              └─────────┬─────────┘
                        │
                  Find plugin
                        │
                        ▼
              ┌───────────────────┐
              │ SentimentAnalyzer │
              └─────────┬─────────┘
                        │
                  execute(data)
                        │
                        ▼
              ┌───────────────────┐
              │ Process the text  │
              └─────────┬─────────┘
                        │
                     result
                        │
                        ▼
              ┌───────────────────┐
              │   PluginManager   │
              └─────────┬─────────┘
                        │
                     result
                        │
                        ▼
                 ┌──────────────┐
                 │     USER     │
                 └──────────────┘
"""
class Plugin:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def execute():
        pass

class SentimentAnalyzer(Plugin):

    def __init__(self):
        super().__init__("SentimentAnalyzer", "Analyzes the sentiment of text.")

    def execute(self, text):
        # Placeholder for sentiment analysis logic
        return "Positive" if text == "Happy" else "Negative"

class TextSummarizer(Plugin):
    def __init__(self):
        super().__init__("TextSummarizer", "Summarizes text content.")

    def execute(self, text):
        # Placeholder for text summarization logic
        return "This is a summary."  # Example output

class CodeReviewer(Plugin):
    def __init__(self):
        super().__init__("CodeReviewer", "Reviews code for potential issues.")

    def execute(self, code):
        # Placeholder for code review logic
        return "No issues found."  # Example output

class PluginManager:
    def __init__(self):
        self.plugins = {}

    def register(self, plugin):
        if isinstance(plugin, Plugin):
            self.plugins[plugin.name] = plugin
        else:
            raise ValueError("Only instances of Plugin can be registered.")

    def list(self):
        return list(self.plugins.keys())

    def execute(self, plugin_name, *args, **kwargs):
        if plugin_name in self.plugins:
            return self.plugins[plugin_name].execute(*args, **kwargs)
        else:
            raise ValueError(f"Plugin '{plugin_name}' not found.")

manager = PluginManager()

manager.register(SentimentAnalyzer())
manager.register(TextSummarizer())
manager.register(CodeReviewer())

print(manager.list())

print(manager.execute("SentimentAnalyzer", "Happy"))
print(manager.execute("TextSummarizer", "Python is a programming language."))
print(manager.execute("CodeReviewer", "print('Hello')"))