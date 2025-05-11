# ------------------------------------------------------------------------------
# query_classifier/llm_labeler.py - Employing an LLM to generate an answer/verdict.
# ------------------------------------------------------------------------------
"""
Takes a prompt containing the user's query and passes it on to an LLM for
response generation.
"""

import aiohttp
import time

async def llm_labeler(api_key: str, model: str, base_url: str, prompt: str) -> str:
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