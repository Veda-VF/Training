"""
(TOPIC - Tool Calling)
A command-line agent that remembers previous messages, uses tools across multiple turns, combines earlier results 
with new queries, and asks clarifying questions when it cannot determine the required parameters.
This version uses simulated weather and city-information tools, so it runs without an API key or extra packages. 
The agent uses rule-based intent detection rather than an actual LLM, but demonstrates the core tool-using agent workflow."""

import json
import re
from datetime import datetime

# ---------------------------------------------
# 1. SIMULATED DATA
# ---------------------------------------------
# These values are examples, not live weather data.

CITY_DATA = {
    "ahmedabad": {
        "name": "Ahmedabad",
        "weather": "Sunny, 34°C",
        "population": "Approximately 8 million",
        "country": "India"
    },
    "mumbai": {
        "name": "Mumbai",
        "weather": "Cloudy, 30°C",
        "population": "Approximately 21 million",
        "country": "India"
    },
    "delhi": {
        "name": "Delhi",
        "weather": "Partly cloudy, 32°C",
        "population": "Approximately 33 million",
        "country": "India"
    },
    "london": {
        "name": "London",
        "weather": "Light rain, 16°C",
        "population": "Approximately 9 million",
        "country": "United Kingdom"
    }
}

LOG_FILE = "conversation_log.jsonl"


# ---------------------------------------------
# 2. CONVERSATION MEMORY
# ---------------------------------------------

history = []
memory = {
    "last_city": None,
    "last_tool": None,
    "last_result": None,
    "last_weather_city": None
}

# If the agent needs clarification, save the question
# and the request that is waiting for an answer.
pending_question = None


# ---------------------------------------------
# 3. LOGGING
# ---------------------------------------------

def log_event(event):
    """Append one event to a JSON Lines conversation log."""
    event["timestamp"] = datetime.now().isoformat(timespec="seconds")

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")


def remember(role, content):
    """Save a message in memory and in the log."""
    entry = {"role": role, "content": content}
    history.append(entry)
    log_event({"type": "message", **entry})


# ---------------------------------------------
# 4. TOOLS
# ---------------------------------------------

def get_weather(city):
    """Return simulated weather for a city."""
    city_key = city.lower()

    if city_key not in CITY_DATA:
        return {"success": False, "error": "City not found."}

    result = {
        "city": CITY_DATA[city_key]["name"],
        "weather": CITY_DATA[city_key]["weather"],
        "source": "simulated data"
    }

    memory["last_city"] = CITY_DATA[city_key]["name"]
    memory["last_weather_city"] = CITY_DATA[city_key]["name"]
    return {"success": True, **result}


def get_city_information(city):
    """Return simulated population and country information."""
    city_key = city.lower()

    if city_key not in CITY_DATA:
        return {"success": False, "error": "City not found."}

    data = CITY_DATA[city_key]
    memory["last_city"] = data["name"]

    return {
        "success": True,
        "city": data["name"],
        "population": data["population"],
        "country": data["country"],
        "source": "simulated data"
    }


def compare_weather(city_a, city_b):
    """Compare the simulated weather of two cities."""
    result_a = get_weather(city_a)
    result_b = get_weather(city_b)

    if not result_a["success"]:
        return result_a
    if not result_b["success"]:
        return result_b

    return {
        "success": True,
        "comparison": [
            result_a,
            result_b
        ]
    }


# Keep tool names and functions together.
TOOLS = {
    "get_weather": get_weather,
    "get_city_information": get_city_information,
    "compare_weather": compare_weather
}


def call_tool(tool_name, arguments):
    """Log a tool call and its result."""
    log_event({
        "type": "tool_call",
        "tool": tool_name,
        "arguments": arguments
    })

    result = TOOLS[tool_name](**arguments)

    log_event({
        "type": "tool_result",
        "tool": tool_name,
        "result": result
    })

    memory["last_tool"] = tool_name
    memory["last_result"] = result

    return result


# ---------------------------------------------
# 5. CITY EXTRACTION
# ---------------------------------------------

def find_cities(message):
    """Find supported city names mentioned in a message."""
    text = message.lower()
    found = []

    for city in CITY_DATA:
        if re.search(r"\b" + re.escape(city) + r"\b", text):
            found.append(city)

    return found


# ---------------------------------------------
# 6. INTENT DETECTION AND TOOL SELECTION
# ---------------------------------------------

def handle_message(message):
    global pending_question

    text = message.lower().strip()
    cities = find_cities(message)

    # ----- Resolve a clarification question first -----
    if pending_question:
        if len(cities) == 1:
            requested_action = pending_question["action"]
            city = cities[0]
            pending_question = None
            return execute_request(requested_action, city)

        if text in {"cancel", "never mind", "nevermind"}:
            pending_question = None
            return "Okay, I cancelled that request."

        return (
            "Please name one supported city: Ahmedabad, Mumbai, "
            "Delhi, or London. You can also type 'cancel'."
        )

    # ----- Compare two cities -----
    if any(word in text for word in ["compare", "difference between"]):
        if len(cities) >= 2:
            result = call_tool(
                "compare_weather",
                {"city_a": cities[0], "city_b": cities[1]}
            )
            return format_result(result)

        if len(cities) == 1 and memory["last_weather_city"]:
            previous = memory["last_weather_city"].lower()

            if cities[0] != previous:
                result = call_tool(
                    "compare_weather",
                    {"city_a": previous, "city_b": cities[0]}
                )
                return format_result(result)

        pending_question = {"action": "compare_weather"}
        return (
            "Which two cities should I compare? "
            "For example, 'Compare Ahmedabad and Mumbai'."
        )

    # ----- Weather request -----
    weather_words = [
        "weather", "temperature", "rain", "sunny",
        "cloudy", "forecast"
    ]

    if any(word in text for word in weather_words):
        if len(cities) > 1:
            pending_question = {"action": "choose_weather_city"}
            return (
                "Which city's weather do you want? "
                "Please name one city."
            )

        if len(cities) == 1:
            return execute_request("get_weather", cities[0])

        # Follow-up: "check that city's weather again"
        if memory["last_weather_city"]:
            city = memory["last_weather_city"].lower()
            return execute_request("get_weather", city)

        pending_question = {"action": "get_weather"}
        return "Which city's weather should I check?"

    # ----- City information request -----
    info_words = [
        "population", "country", "city information",
        "tell me about", "information about"
    ]

    if any(word in text for word in info_words):
        if len(cities) > 1:
            pending_question = {"action": "choose_information_city"}
            return "Which city do you want information about?"

        if len(cities) == 1:
            return execute_request("get_city_information", cities[0])

        # Follow-up: "What is its population?"
        if memory["last_city"]:
            return execute_request(
                "get_city_information",
                memory["last_city"].lower()
            )

        pending_question = {"action": "get_city_information"}
        return "Which city's information do you need?"

    # ----- Reuse the previous tool result -----
    if any(word in text for word in [
        "again", "repeat", "that result", "same city"
    ]):
        if memory["last_weather_city"]:
            return execute_request(
                "get_weather",
                memory["last_weather_city"].lower()
            )

        if memory["last_city"]:
            return execute_request(
                "get_city_information",
                memory["last_city"].lower()
            )

    return (
        "I can check weather, retrieve city information, or compare "
        "two cities. Try 'Weather in Ahmedabad'."
    )


# ---------------------------------------------
# 7. EXECUTE REQUESTS AND FORMAT RESULTS
# ---------------------------------------------

def execute_request(action, city):
    if action == "choose_weather_city":
        action = "get_weather"
    elif action == "choose_information_city":
        action = "get_city_information"

    if action == "get_weather":
        result = call_tool("get_weather", {"city": city})
    elif action == "get_city_information":
        result = call_tool("get_city_information", {"city": city})
    else:
        return "I don't know how to perform that action."

    return format_result(result)


def format_result(result):
    if not result.get("success"):
        return "Sorry, " + result.get("error", "something went wrong")

    if "comparison" in result:
        first, second = result["comparison"]
        return (
            f"Weather comparison:\n"
            f"- {first['city']}: {first['weather']}\n"
            f"- {second['city']}: {second['weather']}\n"
            f"(These are simulated values.)"
        )

    if "weather" in result:
        return (
            f"Weather in {result['city']}: {result['weather']} "
            f"(simulated data)"
        )

    return (
        f"City: {result['city']}\n"
        f"Population: {result['population']}\n"
        f"Country: {result['country']} "
        f"(simulated data)"
    )


# ---------------------------------------------
# 8. REPL: READ, EVALUATE, PRINT, LOOP
# ---------------------------------------------

def main():
    print("Conversational Tool-Using Agent")
    print("Tools: weather, city information, compare weather")
    print("Supported cities: Ahmedabad, Mumbai, Delhi, London")
    print("Type 'exit' to quit.\n")

    log_event({"type": "session_started"})

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding conversation.")
            break

        if message.lower() in {"exit", "quit"}:
            remember("user", message)
            remember("assistant", "Goodbye!")
            print("Agent: Goodbye!")
            break

        if not message:
            continue

        remember("user", message)

        response = handle_message(message)

        print("Agent:", response)
        remember("assistant", response)

    log_event({
        "type": "session_ended",
        "total_messages": len(history),
        "last_city": memory["last_city"],
        "last_tool": memory["last_tool"]
    })

    print(f"\nConversation and tool logs saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
