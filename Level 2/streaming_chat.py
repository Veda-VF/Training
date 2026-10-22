# (TOPIC: Streaming Responses)

import time
import threading
import itertools
import random

# -----------------------------------------
# 1. SETTINGS
# -----------------------------------------

FIRST_TOKEN_DELAY = 1.2  # Simulates waiting for the model
TOKEN_DELAY = 0.08       # Delay between simulated tokens

# Stores all messages so later responses can use earlier context.
conversation_history = []


# -----------------------------------------
# 2. SPINNING INDICATOR
# -----------------------------------------

def show_spinner(stop_event):
    """Display a spinner until the first token arrives."""
    symbols = itertools.cycle(["|", "/", "-", "\\"])

    while not stop_event.is_set():
        print(f"\rWaiting for response {next(symbols)}", end="", flush=True)
        time.sleep(0.1)

    # Clear the spinner line when finished.
    print("\r" + " " * 35 + "\r", end="", flush=True)


# -----------------------------------------
# 3. SIMULATED STREAMING RESPONSE
# -----------------------------------------

def stream_response(history):
    """
    Simulate a language model streaming tokens.
    The first delay imitates time spent waiting for the first token.
    """

    # Simulate model processing before the first token.
    time.sleep(FIRST_TOKEN_DELAY)

    # Read the latest user message.
    latest_message = history[-1]["content"].lower()

    # Use previous conversation messages for basic context.
    previous_user_messages = [
        item["content"]
        for item in history[:-1]
        if item["role"] == "user"
    ]

    if any(word in latest_message for word in ["hello", "hi", "hey"]):
        response = "Hello! How can I help you today?"

    elif "my name is" in latest_message:
        name = latest_message.split("my name is", 1)[1].strip()
        response = f"Nice to meet you, {name}! I will use that context in this conversation."

    elif any(word in latest_message for word in ["what did i ask", "previous question"]):
        if previous_user_messages:
            response = (
                "Your previous question was: "
                + previous_user_messages[-1]
            )
        else:
            response = "This is the first question in our conversation."

    elif "explain it" in latest_message or "explain that" in latest_message:
        if previous_user_messages:
            response = (
                "You were previously asking about: "
                + previous_user_messages[-1]
                + ". Please tell me which part you want explained."
            )
        else:
            response = "What topic would you like me to explain?"

    else:
        response = (
            "You said: "
            + history[-1]["content"]
            + ". This is a simulated streaming response. "
            "Connect a language model API for intelligent answers."
        )

    # Split into word-sized chunks to imitate streaming tokens.
    # The trailing spaces preserve natural word spacing.
    words = response.split()

    for index, word in enumerate(words):
        token = word if index == len(words) - 1 else word + " "
        yield token
        time.sleep(TOKEN_DELAY)


# -----------------------------------------
# 4. STREAM AND MEASURE THE RESPONSE
# -----------------------------------------

def generate_response():
    start_time = time.perf_counter()

    # Create the stream iterator before starting the spinner.
    token_stream = iter(stream_response(conversation_history))

    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(
        target=show_spinner,
        args=(stop_spinner,),
        daemon=True
    )

    spinner_thread.start()

    try:
        # Wait for the first token while the spinner runs.
        first_token = next(token_stream)
    except StopIteration:
        stop_spinner.set()
        spinner_thread.join()
        print("No response generated.")
        return
    except Exception as error:
        stop_spinner.set()
        spinner_thread.join()
        print(f"\nStreaming error: {error}")
        conversation_history.append({
            "role": "assistant",
            "content": f"Streaming failed: {error}"
        })
        return

    # Time from starting generation until first token is available.
    ttft = time.perf_counter() - start_time

    stop_spinner.set()
    spinner_thread.join()

    print("Assistant: ", end="", flush=True)

    response_parts = [first_token]
    token_count = 1

    # Print the first token immediately.
    print(first_token, end="", flush=True)

    # Print each remaining token as it arrives.
    try:
        for token in token_stream:
            print(token, end="", flush=True)
            response_parts.append(token)
            token_count += 1
    except Exception as error:
        print(f"\n[Streaming interrupted: {error}]", end="")
    finally:
        print()

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # This demo counts word-sized chunks as approximate tokens.
    # Actual language models use their own tokenizers.
    tokens_per_second = (
        token_count / total_time if total_time > 0 else 0
    )

    complete_response = "".join(response_parts).strip()

    # Save the full assistant response for future turns.
    conversation_history.append({
        "role": "assistant",
        "content": complete_response
    })

    print("\n--- Streaming Statistics ---")
    print(f"Time to first token (TTFT): {ttft:.3f} seconds")
    print(f"Approximate tokens received: {token_count}")
    print(f"Approximate tokens/second: {tokens_per_second:.2f}")
    print(f"Total response time: {total_time:.3f} seconds")
    print("-----------------------------\n")


# -----------------------------------------
# 5. COMMAND-LINE CHAT LOOP
# -----------------------------------------

def main():
    print("Streaming Chat Interface")
    print("Type 'exit' to end the conversation.")
    print("Responses are simulated, not generated by a real AI model.\n")

    while True:
        try:
            # Prompt character shown while waiting for user input.
            user_message = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChat ended.")
            break

        if user_message.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_message:
            continue

        # Store the user message before generating the response.
        conversation_history.append({
            "role": "user",
            "content": user_message
        })

        generate_response()


if __name__ == "__main__":
    main()