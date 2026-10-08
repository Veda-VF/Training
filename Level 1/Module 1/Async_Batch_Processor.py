"""batch_processor()
      │
      ├── divide prompts into batches
      │
      ├── for each batch
      │       │
      │       ├── create async tasks
      │       │
      │       ├── asyncio.gather()
      │       │
      │       ├── collect success/failure
      │       │
      │       └── calculate latency
      │
      └── return results"""

import asyncio
import time
import random

async def process_prompt(prompt):  # Simulates an API call

    latency = random.uniform(0.5, 2)
    await asyncio.sleep(latency)
    if random.random() < 0.2:
        raise Exception("API failed")
    return f"Response for: {prompt}"

async def safe_process(prompt):  # Handles success/failure and measures latency

    start = time.time()
    try:
        result = await process_prompt(prompt)
        return {"prompt": prompt, "success": True, "latency": time.time() - start, "result": result}
    except Exception as error:
        return {"prompt": prompt, "success": False, "latency": time.time() - start, "error": str(error)}

async def batch_processor(prompts, batch_size):  # Processes prompts in batches

    all_results = []
    for i in range(0, len(prompts), batch_size):  # Divide prompts into batches
        batch = prompts[i:i + batch_size]
        print(f"\nProcessing batch: {batch}")

        tasks = [safe_process(prompt) for prompt in batch]  # Create async tasks

        results = await asyncio.gather(*tasks)  # Run tasks concurrently

        all_results.extend(results)

    return all_results

async def main():  # Runs the batch processor

    prompts = [
        "Explain AI",
        "Explain ML",
        "Explain Python",
        "Explain SQL",
        "Explain APIs",
        "Explain CNN",
        "Explain RNN",
        "Explain NLP"
    ]
    results = await batch_processor(prompts, 3)
    print("\nResults:")
    
    for result in results:  # Display each result
        print(result)

asyncio.run(main())  # Start the async program