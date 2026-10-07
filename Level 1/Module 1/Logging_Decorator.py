"""
In Python, a logger is an object provided by the built-in logging module that acts as the primary interface for recording events, errors, and informational messages during a program's execution.

function fails
     ↓
decorator catches error
     ↓
write error to log
     ↓
raise again
     ↓
program receives the original error"""

import time
import traceback
from functools import wraps


def logger(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            with open("app.log", "a") as file:
                file.write(f"Function: {func.__name__}\n")
                file.write(f"Arguments: {args}, {kwargs}\n")
                file.write(f"Return value: {result}\n")
                file.write(f"Execution time: {execution_time} seconds\n")
                file.write("-" * 40 + "\n")

            return result

        except Exception as error:

            with open("app.log", "a") as file:
                file.write(f"Function: {func.__name__}\n")
                file.write(f"Arguments: {args}, {kwargs}\n")
                file.write(f"Exception: {error}\n")
                file.write("Traceback:\n")
                file.write(traceback.format_exc())
                file.write("-" * 40 + "\n")

            raise

    return wrapper
@logger
def add(a, b):
    return a + b

@logger
def greet(name):
    return f"Hello {name}"

@logger
def divide(a, b):
    return a / b

print(add(10, 20))
print(greet("Veda"))
print(divide(10, 0))