# CUSTOMER SERVICE FUNCTION SUITE (TOPIC: Function Calling)
"""
Concept	How the code implements it
Function calling	    choose_function() selects a function, and AVAILABLE_FUNCTIONS executes it.
Function schemas	    FUNCTION_SCHEMAS defines descriptions, parameters, types, and required fields.
Simulated data	        Python dictionaries store customers and orders.
Error handling	        Functions return structured error messages for invalid IDs or disallowed operations.
Conversation history	conversation_history stores user messages and assistant results.
Follow-up context	    context remembers the latest customer and order IDs."""


import re
import json

# -----------------------------------------
# 1. SIMULATED CUSTOMER AND ORDER DATA
# -----------------------------------------

customers = {
    "C1001": {
        "name": "Aarav Patel",
        "email": "aarav@example.com",
        "phone": "9876543210"
    },
    "C1002": {
        "name": "Diya Shah",
        "email": "diya@example.com",
        "phone": "9876501234"
    }
}

orders = {
    "O5001": {
        "customer_id": "C1001",
        "item": "Wireless Headphones",
        "status": "Shipped",
        "tracking_id": "TRK12345",
        "refund_eligible": True,
        "shipping_address": "Ahmedabad, Gujarat"
    },
    "O5002": {
        "customer_id": "C1002",
        "item": "Laptop Stand",
        "status": "Processing",
        "tracking_id": None,
        "refund_eligible": False,
        "shipping_address": "Surat, Gujarat"
    }
}

conversation_history = []
context = {
    "customer_id": None,
    "order_id": None
}


# -----------------------------------------
# 2. FUNCTION SCHEMAS
# These describe each function's inputs.
# They follow a JSON Schema-style structure.
# -----------------------------------------

FUNCTION_SCHEMAS = {
    "lookup_customer": {
        "description": "Look up a customer using their customer ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Customer ID, e.g. C1001"
                }
            },
            "required": ["customer_id"],
            "additionalProperties": False
        }
    },

    "check_order_status": {
        "description": "Check an order's status and tracking details.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Order ID, e.g. O5001"
                }
            },
            "required": ["order_id"],
            "additionalProperties": False
        }
    },

    "process_refund": {
        "description": "Simulate a refund request for an order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Order ID to refund"
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for requesting a refund"
                }
            },
            "required": ["order_id", "reason"],
            "additionalProperties": False
        }
    },

    "update_shipping_address": {
        "description": "Update the shipping address of an order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Order ID to update"
                },
                "new_address": {
                    "type": "string",
                    "description": "New complete shipping address"
                }
            },
            "required": ["order_id", "new_address"],
            "additionalProperties": False
        }
    },

    "escalate_to_human": {
        "description": "Create a simulated request for human support.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": ["string", "null"],
                    "description": "Customer ID, if known"
                },
                "issue": {
                    "type": "string",
                    "description": "Reason for escalation"
                },
                "order_id": {
                    "type": ["string", "null"],
                    "description": "Related order ID, if known"
                }
            },
            "required": ["customer_id", "issue", "order_id"],
            "additionalProperties": False
        }
    }
}


# -----------------------------------------
# 3. CUSTOMER SERVICE FUNCTIONS
# -----------------------------------------

def lookup_customer(customer_id):
    customer = customers.get(customer_id)

    if not customer:
        return {"success": False, "message": "Customer not found."}

    context["customer_id"] = customer_id

    return {
        "success": True,
        "customer_id": customer_id,
        "customer": customer
    }


def check_order_status(order_id):
    order = orders.get(order_id)

    if not order:
        return {"success": False, "message": "Order not found."}

    context["order_id"] = order_id
    context["customer_id"] = order["customer_id"]

    return {
        "success": True,
        "order_id": order_id,
        "item": order["item"],
        "status": order["status"],
        "tracking_id": order["tracking_id"]
    }


def process_refund(order_id, reason):
    order = orders.get(order_id)

    if not order:
        return {"success": False, "message": "Order not found."}

    if not reason.strip():
        return {"success": False, "message": "Please provide a refund reason."}

    if not order["refund_eligible"]:
        return {
            "success": False,
            "message": "This order is not eligible for a refund in the demo."
        }

    # Simulation only: no real payment or refund is processed.
    order["status"] = "Refund Requested"
    context["order_id"] = order_id
    context["customer_id"] = order["customer_id"]

    return {
        "success": True,
        "message": "Refund request recorded in the demo.",
        "order_id": order_id,
        "reason": reason,
        "status": order["status"]
    }


def update_shipping_address(order_id, new_address):
    order = orders.get(order_id)

    if not order:
        return {"success": False, "message": "Order not found."}

    if not new_address.strip():
        return {"success": False, "message": "Address cannot be empty."}

    # Demo rule: only allow changes while the order is processing.
    if order["status"] != "Processing":
        return {
            "success": False,
            "message": "Address cannot be changed after shipping."
        }

    order["shipping_address"] = new_address.strip()
    context["order_id"] = order_id
    context["customer_id"] = order["customer_id"]

    return {
        "success": True,
        "message": "Shipping address updated in the demo.",
        "order_id": order_id,
        "new_address": order["shipping_address"]
    }


def escalate_to_human(customer_id, issue, order_id=None):
    # Simulate creating a support ticket.
    ticket_id = f"T{1000 + len(conversation_history)}"

    return {
        "success": True,
        "message": "A simulated support ticket has been created.",
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "order_id": order_id,
        "issue": issue,
        "status": "Waiting for human support"
    }


# Map function names to executable Python functions.
AVAILABLE_FUNCTIONS = {
    "lookup_customer": lookup_customer,
    "check_order_status": check_order_status,
    "process_refund": process_refund,
    "update_shipping_address": update_shipping_address,
    "escalate_to_human": escalate_to_human
}


# -----------------------------------------
# 4. EXTRACT IDS FROM USER MESSAGES
# -----------------------------------------

def extract_id(message, prefix):
    pattern = rf"\b{prefix}\d+\b"
    match = re.search(pattern, message.upper())
    return match.group(0) if match else None


# -----------------------------------------
# 5. CHOOSE A FUNCTION FROM THE MESSAGE
# This is a simple rule-based router, not an LLM.
# -----------------------------------------

def choose_function(message):
    text = message.lower()

    customer_id = extract_id(message, "C")
    order_id = extract_id(message, "O")

    if customer_id:
        context["customer_id"] = customer_id

    if order_id:
        context["order_id"] = order_id

    # Human support takes priority.
    if any(word in text for word in [
        "human", "agent", "representative", "speak to someone",
        "manager", "escalate"
    ]):
        return "escalate_to_human", {
            "customer_id": context["customer_id"],
            "issue": message,
            "order_id": context["order_id"]
        }

    if any(word in text for word in [
        "refund", "money back", "return my money"
    ]):
        reason = message
        if order_id:
            reason = re.sub(
                rf"\b{re.escape(order_id)}\b", "", message,
                flags=re.IGNORECASE
            ).strip()
        if not reason or reason.lower() == "refund":
            reason = "Customer requested a refund"

        if not context["order_id"]:
            return None, {"error": "Please provide an order ID, e.g. O5001."}

        return "process_refund", {
            "order_id": context["order_id"],
            "reason": reason
        }

    if any(word in text for word in [
        "change address", "update address", "shipping address",
        "delivery address"
    ]):
        # Expected format:
        # Update address O5002 to 10 Main Road, Surat
        match = re.search(
            r"\bto\s+(.+)$", message, flags=re.IGNORECASE
        )

        if not context["order_id"]:
            return None, {"error": "Please provide an order ID, e.g. O5002."}

        if not match:
            return None, {
                "error": "Please use: Update address O5002 to your new address"
            }

        return "update_shipping_address", {
            "order_id": context["order_id"],
            "new_address": match.group(1).strip()
        }

    if any(word in text for word in [
        "order status", "track order", "where is my order",
        "tracking", "delivery status", "has it shipped"
    ]):
        if not context["order_id"]:
            return None, {"error": "Please provide an order ID, e.g. O5001."}

        return "check_order_status", {
            "order_id": context["order_id"]
        }

    if any(word in text for word in [
        "customer details", "look up customer", "lookup customer",
        "my account", "customer information"
    ]):
        if not context["customer_id"]:
            return None, {
                "error": "Please provide a customer ID, e.g. C1001."
            }

        return "lookup_customer", {
            "customer_id": context["customer_id"]
        }

    # A message containing only a customer ID can look up that customer.
    if customer_id:
        return "lookup_customer", {"customer_id": customer_id}

    # A message containing only an order ID checks its status.
    if order_id:
        return "check_order_status", {"order_id": order_id}

    return None, {
        "error": (
            "I couldn't identify the request. Try asking about a customer, "
            "order status, refund, address update, or human support."
        )
    }


# -----------------------------------------
# 6. INTERACTIVE CHAT LOOP
# -----------------------------------------

def main():
    print("Customer Service Chatbot")
    print("Type 'exit' to stop.")
    print("Demo IDs: customers C1001/C1002; orders O5001/O5002.")
    print("Example: Check order status O5001\n")

    while True:
        message = input("You: ").strip()

        if message.lower() in {"exit", "quit", "bye"}:
            print("Bot: Goodbye!")
            break

        if not message:
            print("Bot: Please enter a message.")
            continue

        # Store the user's message for conversation history.
        conversation_history.append({
            "role": "user",
            "content": message
        })

        function_name, arguments = choose_function(message)

        if function_name is None:
            response = arguments
        else:
            # Execute the selected function with its arguments.
            try:
                result = AVAILABLE_FUNCTIONS[function_name](**arguments)
                response = {
                    "function_called": function_name,
                    "arguments": arguments,
                    "result": result
                }
            except (TypeError, ValueError) as error:
                response = {
                    "error": f"Could not run the function: {error}"
                }

        # Print structured output so it is easy to inspect.
        print("Bot:", json.dumps(response, indent=2))

        # Store the function call/result or error for follow-up context.
        conversation_history.append({
            "role": "assistant",
            "content": response
        })


if __name__ == "__main__":
    main()

"""
Example conversation:
You
Check order status O5001

Bot — check_order_status
- Item: Wireless Headphones
- Status: Shipped
- Tracking ID: TRK12345

You
I want a refund for it

Bot — process_refund
The refund request is recorded in the demo. The order status is now Refund Requested."""