"""
Quick authentication test for DataForSEO API
Run this to verify your credentials work before starting the MCP server
"""

import os
import base64
import asyncio
import httpx
from dotenv import load_dotenv

load_dotenv()

DATAFORSEO_LOGIN = os.getenv("DATAFORSEO_LOGIN")
DATAFORSEO_PASSWORD = os.getenv("DATAFORSEO_PASSWORD")

if not DATAFORSEO_LOGIN or not DATAFORSEO_PASSWORD:
    print("ERROR: DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD must be set in .env file")
    exit(1)

credentials = f"{DATAFORSEO_LOGIN}:{DATAFORSEO_PASSWORD}"
encoded_credentials = base64.b64encode(credentials.encode()).decode()
AUTH_HEADER = {"Authorization": f"Basic {encoded_credentials}"}


async def test_connection():
    print("=" * 80)
    print("TESTING DATAFORSEO API CONNECTION")
    print("=" * 80)
    print(f"Login: {DATAFORSEO_LOGIN}")
    print(f"Testing endpoint: /v3/ai_optimization/search_mentions/live")
    print("=" * 80)
    
    url = "https://api.dataforseo.com/v3/ai_optimization/search_mentions/live"
    payload = [{
        "keyword": "Semrush",
        "language_name": "English",
        "location_name": "United States"
    }]
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers={**AUTH_HEADER, "Content-Type": "application/json"},
                json=payload
            )
            
            print(f"\n✅ HTTP Status Code: {response.status_code}")
            
            result = response.json()
            
            print(f"✅ API Status Code: {result.get('status_code')}")
            print(f"✅ API Message: {result.get('status_message')}")
            
            if result.get("tasks"):
                task_result = result["tasks"][0].get("result", [{}])[0]
                total_count = task_result.get("total_count", 0)
                print(f"✅ Total mentions found: {total_count}")
                
                if total_count > 0:
                    items = task_result.get("items", [])
                    print(f"✅ First mention: {items[0].get('domain', 'N/A')}")
            
            print("\n" + "=" * 80)
            print("✅ SUCCESS! Your DataForSEO credentials are working!")
            print("=" * 80)
            
    except httpx.HTTPError as e:
        print(f"\n❌ HTTP Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your .env file has correct credentials")
        print("2. Verify no extra spaces in DATAFORSEO_LOGIN or DATAFORSEO_PASSWORD")
        print("3. Confirm your DataForSEO account has credits")
        print("4. Check https://app.dataforseo.com/api-dashboard for API status")
        
    except Exception as e:
        print(f"\n❌ Unexpected Error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_connection())