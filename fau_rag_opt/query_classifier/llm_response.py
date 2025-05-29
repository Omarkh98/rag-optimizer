# ------------------------------------------------------------------------------
# query_classifier/llm_labeler.py - Employing an LLM to generate an answer/verdict.
# ------------------------------------------------------------------------------
"""
Takes a prompt containing the user's query and passes it on to an LLM for
response generation.
"""

import aiohttp
import asyncio
import logging
import time

from ..helpers.rate_limiter import RateLimiter

rate_limiter = RateLimiter(max_calls=1000, period=60)

CONCURRENT_REQUESTS = 10
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

async def llm_response(api_key: str, model: str, base_url: str, prompt: str) -> str:
    start_time = time.time()
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    print(f"\n📨 Sending prompt to LLM...")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(base_url, headers=headers, json=data) as response:
                print(f"🔄 Received response status: {response.status}")
                if response.status != 200:
                    error_message = await response.text()
                    print(f"❗ Error message: {error_message}")
                    raise ValueError(f"❗ LLM endpoint returned {response.status}")

                response_data = await response.json()

                if 'choices' not in response_data or len(response_data['choices']) == 0:
                    print(f"❗ No 'choices' found in the LLM response!")
                    raise ValueError("❗ LLM response missing 'choices'")

                answer = response_data['choices'][0]['message']['content']
                elapsed_time = time.time() - start_time
                print(f"✅ LLM answered in {elapsed_time:.2f} seconds. Genereted Answer: {answer.strip()}")
                return answer.strip()
        except Exception as e:
            print(f"❗ Exception during LLM call: {str(e)}")
            raise

async def query_university_endpoint(session, api_key: str, model: str, base_url: str, prompt: str):
    start_time = time.time()
    headers = {
        "Content-Type": "application/json",
        'Authorization': f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    } 
    max_retries = 10 
    backoff = 13
    for attempt in range(1, max_retries + 1):
        await rate_limiter.acquire()
        async with semaphore:
            try:
                async with session.post(base_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        if 'choices' in response_data and len(response_data['choices']) > 0:
                            answer = response_data['choices'][0]['message']['content']
                            elapsed_time = time.time() - start_time
                            print(f"✅ LLM answered in {elapsed_time:.2f} seconds. Genereted Answer: {answer.strip()}")
                            return answer.strip()
                        else:
                            return "No answer found in the response"
                    elif response.status == 429:
                        logging.warning(f"Rate limit exceeded on attempt {attempt}/{max_retries}. Waiting for {backoff} seconds before retrying.")
                    else:
                        error_text = await response.text()
                        logging.error(f"Error: {response.status} - {error_text} (attempt {attempt}/{max_retries}).")
            except Exception as e:
                logging.error(f"Exception on attempt {attempt}/{max_retries}: {str(e)}")
        await asyncio.sleep(backoff)
        backoff *= 2
    return f"Error: Failed after {max_retries} attempts due to rate limiting or other errors."