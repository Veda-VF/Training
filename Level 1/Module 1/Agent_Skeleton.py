"""
Mini Agent Framework — OOP
Multiple agents with simple keyword-based tool calling
"""
# Agent Flow: 
"""
User input
    ↓
MainRouter
    ↓
find "calculate"
    ↓
CalculatorBot
    ↓
calculate tool
    ↓
"25 * 4"
    ↓
   100
"""
# ---------------------------------------------------------
# Message
# ---------------------------------------------------------

class Message:
    """A chat message with role and content."""

    def __init__(self, role, content):
        self.role = role
        self.content = content

    def __repr__(self):
        return f"Message({self.role}: {self.content[:40]}...)"

    def to_dict(self):
        return {
            "role": self.role,
            "content": self.content
        }


# ---------------------------------------------------------
# Conversation Memory
# ---------------------------------------------------------

class ConversationMemory:
    """Stores conversation history."""

    def __init__(self, max_messages=20):
        self._messages = []
        self.max_messages = max_messages

    def add(self, role, content):

        # Create a Message object
        msg = Message(role, content)

        # Add it to memory
        self._messages.append(msg)

        # Keep only the latest messages
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def get_messages(self):
        return [m.to_dict() for m in self._messages]

    def __len__(self):
        return len(self._messages)

    def clear(self):
        self._messages.clear()


# ---------------------------------------------------------
# Base Agent
# ---------------------------------------------------------

class Agent:
    """Basic AI agent with memory and tools."""

    def __init__(self, name, system_prompt="", memory=None):

        self.name = name
        self.system_prompt = system_prompt

        # If memory is not provided, create new memory
        self.memory = memory or ConversationMemory()

        # Dictionary to store tools
        self.tools = {}

    def add_tool(self, name, func, description=""):

        self.tools[name] = {
            "func": func,
            "description": description
        }

    def respond(self, user_input):

        # Store user's message
        self.memory.add("user", user_input)

        # Simple response for the base Agent
        response = f"[{self.name}] Acknowledged: {user_input}"

        # Store assistant response
        self.memory.add("assistant", response)

        return response

    def __repr__(self):

        tool_names = list(self.tools.keys())

        return (
            f"Agent(name='{self.name}', "
            f"tools={tool_names}, "
            f"memory={len(self.memory)} msgs)"
        )


# ---------------------------------------------------------
# Tool Agent
# ---------------------------------------------------------

class ToolAgent(Agent):
    """
    Agent that can find and execute one of its tools.
    """

    def respond(self, user_input):

        # Store user's message
        self.memory.add("user", user_input)

        # Check every tool available to this agent
        for tool_name in self.tools:

            # Check if tool name appears in user input
            if tool_name in user_input.lower():

                # Special handling for calculator
                if tool_name == "calculate":

                    # Remove the word 'calculate'
                    # so that only the expression remains
                    expression = (
                        user_input.lower()
                        .replace("calculate", "")
                        .strip()
                    )

                    result = self.tools[tool_name]["func"](expression)

                else:

                    # For other tools, pass the full input
                    result = self.tools[tool_name]["func"](user_input)

                # Create final response
                response = (
                    f"[{self.name}] "
                    f"Tool '{tool_name}' executed: {result}"
                )

                # Store assistant response
                self.memory.add("assistant", response)

                return response

        # If no tool matched
        response = (
            f"[{self.name}] "
            f"No tool matched: {user_input}"
        )

        self.memory.add("assistant", response)

        return response


# ---------------------------------------------------------
# Tool Calling / Agent Router
# ---------------------------------------------------------

class ToolCalling(Agent):
    """
    Routes the user's request to the correct agent
    using simple keyword matching.
    """

    def __init__(self, name, system_prompt=""):

        super().__init__(name, system_prompt)

        # Stores agents using keywords
        self.agents = {}

    def add_agent(self, keyword, agent):

        # Example:
        # "calculate" -> CalculatorBot
        self.agents[keyword] = agent

    def respond(self, user_input):

        user_input = user_input.lower()

        # Check which keyword is present
        for keyword, agent in self.agents.items():

            if keyword in user_input:

                # Send the request to the selected agent
                return agent.respond(user_input)

        # No matching agent found
        return (
            f"[{self.name}] "
            f"No agent matched your request."
        )


# ---------------------------------------------------------
# Main Program
# ---------------------------------------------------------

if __name__ == "__main__":

    # -----------------------------------------------------
    # Create Calculator Agent
    # -----------------------------------------------------

    calculator_agent = ToolAgent(
        name="CalculatorBot",
        system_prompt="You are a calculator agent."
    )

    calculator_agent.add_tool(
        "calculate",
        lambda expression: eval(expression),
        "Evaluate mathematical expressions"
    )


    # -----------------------------------------------------
    # Create Research Agent
    # -----------------------------------------------------

    research_agent = ToolAgent(
        name="ResearchBot",
        system_prompt="You are a research assistant."
    )

    research_agent.add_tool(
        "search",
        lambda query: f"Results for: {query}",
        "Search for information"
    )


    # -----------------------------------------------------
    # Create Router
    # -----------------------------------------------------

    router = ToolCalling(
        name="MainRouter",
        system_prompt="Choose the correct agent."
    )

    # Tell router which keyword belongs to which agent
    router.add_agent("calculate", calculator_agent)
    router.add_agent("search", research_agent)


    # -----------------------------------------------------
    # Take input from user
    # -----------------------------------------------------

    print("Mini Agent Framework")
    print("Type 'exit' to stop.")

    while True:

        user_input = input("\nYou: ")

        # Stop the program
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Router chooses the appropriate agent
        response = router.respond(user_input)

        print(response)