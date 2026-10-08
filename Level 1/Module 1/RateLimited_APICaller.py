import asyncio
import time
import random

async def apicaller(id):  # Simulates an API call
    latency = random.uniform(0.5, 2)
    await asyncio.sleep(latency)
    print(f"API call {id} completed in {latency:.2f} seconds")
    return f"Response for API call {id}"

class RateLimiter:
    def __init__(self, requests_per_second):  # Set request limit
        self.delay = 1 / requests_per_second
        self.lock = asyncio.Lock()
        self.last_request = 0

    async def wait(self):  # Controls when the next request can start
        async with self.lock:
            now = time.time()
            wait_time = self.delay - (now - self.last_request)

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            self.last_request = time.time()

limiter = RateLimiter(5)  # Allow 5 requests per second

async def limited_apicaller(id):  # Applies rate limit before API call
    await limiter.wait()
    return await apicaller(id)

async def main():  # Creates 20 API calls
    tasks = [limited_apicaller(i) for i in range(1, 21)]  # Create 20 tasks
    results = await asyncio.gather(*tasks)  # Run tasks concurrently

    for result in results:  # Print all responses
        print(result)

asyncio.run(main())  # Start the program

"""asyncio.run(main())
        ↓
     main()
        ↓
Create 20 coroutine objects
        ↓
asyncio.gather()
        ↓
Start 20 async tasks
        ↓
limited_apicaller()
        ↓
limiter.wait()
        ↓
Check last request time
        ↓
Need to wait?
   ↙            ↘
 YES             NO
  ↓               ↓
sleep()        continue
  ↓               ↓
  └───────┬───────┘
          ↓
    apicaller()
          ↓
random latency
          ↓
await asyncio.sleep()
          ↓
Event loop handles other tasks
          ↓
API finishes
          ↓
return response
          ↓
gather() collects response
          ↓
All 20 finished?
   ↓
 YES
   ↓
results
   ↓
print responses

| Code | What Python is doing |

| `async def` | Creates a coroutine function |
| `await` | Pause this task and let other async tasks run |
| `asyncio.gather()` | Run multiple coroutines concurrently and collect results |
| `limiter.wait()` | Controls when each request is allowed to start |"""