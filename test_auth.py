"""
Authentication Test for DataForSEO AI Optimization API
✅ Uses correct "target" parameter (not "keyword")
"""

import os
import base64
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

LOGIN = os.getenv("DATAFORSEO_LOGIN")
PASSWORD = os.getenv("DATAFORSEO_PASSWORD")

if not LOGIN or not PASSWORD:
    print("❌ Set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD in .env")
    exit(1)

# Create auth header
creds = f"{LOGIN}:{PASSWORD}"
encoded = base64.b64encode(creds.encode()).decode()
HEADERS = {
    "Authorization": f"Basic {encoded}",
    "Content-Type": "application/json"
}


async def test_api():
    print("=" * 80)
    print("TESTING DATAFORSEO API - REAL CALLS")
    print("=" * 80)
    print(f"Account: {LOGIN}\n")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        
        # Test 1: ChatGPT Live
        print("1. Testing ChatGPT Live...")
        print("-" * 80)
        try:
            response = await client.post(
                "https://api.dataforseo.com/v3/ai_optimization/chat_gpt/llm_responses/live",
                headers=HEADERS,
                json=[{
                    "user_prompt": "What is SEO?",
                    "model_name": "gpt-4o-mini",
                    "max_output_tokens": 100
                }]
            )
            
            data = response.json()
            print(f"API Status: {data.get('status_code')} - {data.get('status_message')}")
            
            if data.get("tasks"):
                task = data["tasks"][0]
                print(f"Task Status: {task.get('status_code')} - {task.get('status_message')}")
                print(f"Cost: ${task.get('cost', 0)}")
                
                if task.get("result") and len(task["result"]) > 0:
                    result = task["result"][0]
                    if result.get("items"):
                        item = result["items"][0]
                        if item.get("sections"):
                            text = item["sections"][0].get("text", "")[:100]
                            print(f"Answer Preview: {text}...")
                            print("✅ WORKS!\n")
                        else:
                            print("⚠️  Got response but no sections\n")
                    else:
                        print("⚠️  Got result but no items\n")
                else:
                    print("⚠️  No result data\n")
            else:
                print("❌ No tasks in response\n")
                
        except Exception as e:
            print(f"❌ ERROR: {e}\n")
        
        
        # Test 2: Claude Live
        print("2. Testing Claude Live...")
        print("-" * 80)
        try:
            response = await client.post(
                "https://api.dataforseo.com/v3/ai_optimization/claude/llm_responses/live",
                headers=HEADERS,
                json=[{
                    "user_prompt": "What is SEO?",
                    "model_name": "claude-3-5-haiku-20241022",
                    "max_output_tokens": 100
                }]
            )
            
            data = response.json()
            print(f"API Status: {data.get('status_code')} - {data.get('status_message')}")
            
            if data.get("tasks"):
                task = data["tasks"][0]
                print(f"Task Status: {task.get('status_code')} - {task.get('status_message')}")
                print(f"Cost: ${task.get('cost', 0)}")
                
                if task.get("status_code") == 20000:
                    print("✅ WORKS!\n")
                else:
                    print(f"⚠️  Task status: {task.get('status_code')}\n")
            else:
                print("❌ No tasks\n")
                
        except Exception as e:
            print(f"❌ ERROR: {e}\n")
        
        
        # Test 3: Gemini Live
        print("3. Testing Gemini Live...")
        print("-" * 80)
        try:
            response = await client.post(
                "https://api.dataforseo.com/v3/ai_optimization/gemini/llm_responses/live",
                headers=HEADERS,
                json=[{
                    "user_prompt": "What is SEO?",
                    "model_name": "gemini-1.5-flash-latest",
                    "max_output_tokens": 100
                }]
            )
            
            data = response.json()
            print(f"API Status: {data.get('status_code')} - {data.get('status_message')}")
            
            if data.get("tasks"):
                task = data["tasks"][0]
                print(f"Task Status: {task.get('status_code')} - {task.get('status_message')}")
                print(f"Cost: ${task.get('cost', 0)}")
                
                if task.get("status_code") == 20000:
                    print("✅ WORKS!\n")
                else:
                    print(f"⚠️  Task status: {task.get('status_code')}\n")
            else:
                print("❌ No tasks\n")
                
        except Exception as e:
            print(f"❌ ERROR: {e}\n")
        
        
        # Test 4: Perplexity Live
        print("4. Testing Perplexity Live...")
        print("-" * 80)
        try:
            response = await client.post(
                "https://api.dataforseo.com/v3/ai_optimization/perplexity/llm_responses/live",
                headers=HEADERS,
                json=[{
                    "user_prompt": "What is SEO?",
                    "model_name": "sonar",
                    "max_output_tokens": 100
                }]
            )
            
            data = response.json()
            print(f"API Status: {data.get('status_code')} - {data.get('status_message')}")
            
            if data.get("tasks"):
                task = data["tasks"][0]
                print(f"Task Status: {task.get('status_code')} - {task.get('status_message')}")
                print(f"Cost: ${task.get('cost', 0)}")
                
                if task.get("status_code") == 20000:
                    print("✅ WORKS!\n")
                else:
                    print(f"⚠️  Task status: {task.get('status_code')}\n")
            else:
                print("❌ No tasks\n")
                
        except Exception as e:
            print(f"❌ ERROR: {e}\n")
        
        
        # Test 5: Search Mentions (will likely need activation)
        print("5. Testing Search Mentions (LLM Mentions)...")
        print("-" * 80)
        try:
            response = await client.post(
                "https://api.dataforseo.com/v3/ai_optimization/llm_mentions/search/live",
                headers=HEADERS,
                json=[{
                    "target": "Google",  # ✅ CORRECTED: was "keyword"
                    "language_name": "English",
                    "location_name": "United States"
                }]
            )
            
            data = response.json()
            print(f"API Status: {data.get('status_code')} - {data.get('status_message')}")
            
            if data.get("tasks"):
                task = data["tasks"][0]
                print(f"Task Status: {task.get('status_code')} - {task.get('status_message')}")
                
                if task.get("status_code") == 40204:
                    print("🔒 NEEDS ACTIVATION - Email support@dataforseo.com\n")
                elif task.get("status_code") == 20000:
                    print("✅ WORKS!\n")
                else:
                    print(f"⚠️  Status: {task.get('status_code')}\n")
            else:
                print("❌ No tasks\n")
                
        except Exception as e:
            print(f"❌ ERROR: {e}\n")
        
        
        # Test 6: Top Domains (will likely need activation)
        print("6. Testing Top Domains (LLM Mentions)...")
        print("-" * 80)
        try:
            response = await client.post(
                "https://api.dataforseo.com/v3/ai_optimization/llm_mentions/top_domains/live",
                headers=HEADERS,
                json=[{
                    "target": "SEO tools",  # ✅ CORRECTED: was "keyword"
                    "language_name": "English",
                    "location_name": "United States"
                }]
            )
            
            data = response.json()
            print(f"API Status: {data.get('status_code')} - {data.get('status_message')}")
            
            if data.get("tasks"):
                task = data["tasks"][0]
                print(f"Task Status: {task.get('status_code')} - {task.get('status_message')}")
                
                if task.get("status_code") == 40204:
                    print("🔒 NEEDS ACTIVATION - Email support@dataforseo.com\n")
                elif task.get("status_code") == 20000:
                    print("✅ WORKS!\n")
                else:
                    print(f"⚠️  Status: {task.get('status_code')}\n")
            else:
                print("❌ No tasks\n")
                
        except Exception as e:
            print(f"❌ ERROR: {e}\n")
        
        
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print()
        print("✅ If all 4 LLM endpoints work → You can use the server!")
        print("🔒 If LLM Mentions need activation → Email support@dataforseo.com")
        print()
        print("Next step: python server.py")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_api())
