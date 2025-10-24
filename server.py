"""
DataForSEO AI Optimization MCP Server - CORRECTED VERSION
Provides tools for tracking brand visibility in LLMs

✅ FIXED: All endpoint URLs corrected to match DataForSEO documentation
"""

import os
import base64
import logging
from typing import Optional, Literal, Any
from datetime import datetime

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("DataForSEO AI Optimization")

# DataForSEO API Configuration
DATAFORSEO_LOGIN = os.getenv("DATAFORSEO_LOGIN")
DATAFORSEO_PASSWORD = os.getenv("DATAFORSEO_PASSWORD")
BASE_URL = "https://api.dataforseo.com"

if not DATAFORSEO_LOGIN or not DATAFORSEO_PASSWORD:
    raise ValueError("DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD must be set in .env file")

# Create authentication header
credentials = f"{DATAFORSEO_LOGIN}:{DATAFORSEO_PASSWORD}"
encoded_credentials = base64.b64encode(credentials.encode()).decode()
AUTH_HEADER = {"Authorization": f"Basic {encoded_credentials}"}


class DataForSEOError(Exception):
    """Custom exception for DataForSEO API errors"""
    pass


async def make_request(
    endpoint: str,
    method: str = "POST",
    data: Optional[list] = None
) -> dict:
    """Make authenticated request to DataForSEO API"""
    url = f"{BASE_URL}{endpoint}"
    
    logger.info(f"Making {method} request to: {endpoint}")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            if method == "POST":
                response = await client.post(
                    url,
                    headers={**AUTH_HEADER, "Content-Type": "application/json"},
                    json=data
                )
            else:
                response = await client.get(url, headers=AUTH_HEADER)
            
            response.raise_for_status()
            result = response.json()
            
            if result.get("status_code") != 20000:
                error_msg = result.get("status_message", "Unknown error")
                logger.error(f"API Error: {error_msg}")
                raise DataForSEOError(f"API Error: {error_msg}")
            
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP Error: {str(e)}")
            raise DataForSEOError(f"HTTP request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise DataForSEOError(f"Request failed: {str(e)}")


# ============================================================================
# TIER 1: CORE TRACKING TOOLS
# ============================================================================

@mcp.tool()
async def search_mentions(
    keyword: str,
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Search for brand/keyword mentions across LLMs (ChatGPT, Claude, Gemini, Perplexity).
    
    ✅ CORRECTED ENDPOINT: /v3/ai_optimization/llm_mentions/search/live
    """
    logger.info(f"search_mentions: keyword={keyword}")
    
    payload = [{
        "keyword": keyword,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/v3/ai_optimization/llm_mentions/search/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        return {
            "total_count": task_result.get("total_count", 0),
            "items": task_result.get("items", []),
            "cost_credits": 2
        }
    
    return result


@mcp.tool()
async def ai_keyword_search_volume(
    keywords: list[str],
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get AI-specific search volume for keywords.
    
    ✅ CORRECTED ENDPOINT: /v3/ai_optimization/ai_keyword_data/live
    """
    logger.info(f"ai_keyword_search_volume: {len(keywords)} keywords")
    
    payload = [{
        "keywords": keywords,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/v3/ai_optimization/ai_keyword_data/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        return {
            "items": task_result.get("items", []),
            "cost_credits": len(keywords)
        }
    
    return result


@mcp.tool()
async def chatgpt_live(
    prompt: str,
    model: str = "gpt-4o-mini",
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get live ChatGPT response with citations.
    
    ✅ CORRECTED ENDPOINT: /v3/ai_optimization/chat_gpt/llm_responses/live
    """
    logger.info(f"chatgpt_live: model={model}")
    
    payload = [{
        "prompt": prompt,
        "model": model,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/v3/ai_optimization/chat_gpt/llm_responses/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        items = task_result.get("items", [])
        
        if items:
            return {
                "answer": items[0].get("answer", ""),
                "citations": items[0].get("citations", []),
                "model_used": model
            }
    
    return result


@mcp.tool()
async def claude_live(
    prompt: str,
    model: str = "claude-3-5-haiku-20241022",
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get live Claude response with citations.
    
    ✅ CORRECTED ENDPOINT: /v3/ai_optimization/claude/llm_responses/live
    """
    logger.info(f"claude_live: model={model}")
    
    payload = [{
        "prompt": prompt,
        "model": model,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/v3/ai_optimization/claude/llm_responses/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        items = task_result.get("items", [])
        
        if items:
            return {
                "answer": items[0].get("answer", ""),
                "citations": items[0].get("citations", []),
                "model_used": model
            }
    
    return result


@mcp.tool()
async def gemini_live(
    prompt: str,
    model: str = "gemini-1.5-flash-latest",
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get live Gemini response with citations.
    
    ✅ CORRECTED ENDPOINT: /v3/ai_optimization/gemini/llm_responses/live
    """
    logger.info(f"gemini_live: model={model}")
    
    payload = [{
        "prompt": prompt,
        "model": model,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/v3/ai_optimization/gemini/llm_responses/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        items = task_result.get("items", [])
        
        if items:
            return {
                "answer": items[0].get("answer", ""),
                "citations": items[0].get("citations", []),
                "model_used": model
            }
    
    return result


# ============================================================================
# TIER 2: HIGH-VALUE TOOLS
# ============================================================================

@mcp.tool()
async def top_domains(
    keyword: str,
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get top domains mentioned by LLMs for a keyword (competitor analysis).
    
    ✅ CORRECTED ENDPOINT: /v3/ai_optimization/llm_mentions/top_domains/live
    """
    logger.info(f"top_domains: keyword={keyword}")
    
    payload = [{
        "keyword": keyword,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/v3/ai_optimization/llm_mentions/top_domains/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        return {
            "total_count": task_result.get("total_count", 0),
            "items": task_result.get("items", []),
            "cost_credits": 2
        }
    
    return result


@mcp.tool()
async def aggregated_metrics(
    target: str,
    target_type: Literal["domain", "page"] = "domain",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get historical aggregated metrics for a domain or page.
    
    ✅ CORRECTED ENDPOINT: /v3/ai_optimization/llm_mentions/aggregated_metrics/live
    """
    logger.info(f"aggregated_metrics: target={target}, type={target_type}")
    
    payload = [{
        "target": target,
        "target_type": target_type,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    if date_from:
        payload[0]["date_from"] = date_from
    if date_to:
        payload[0]["date_to"] = date_to
    
    result = await make_request(
        "/v3/ai_optimization/llm_mentions/aggregated_metrics/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        return {
            "metrics": task_result.get("metrics", {}),
            "target": target,
            "cost_credits": 2
        }
    
    return result


@mcp.tool()
async def perplexity_live(
    prompt: str,
    model: str = "sonar",
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get live Perplexity response with citations.
    
    ✅ CORRECTED ENDPOINT: /v3/ai_optimization/perplexity/llm_responses/live
    """
    logger.info(f"perplexity_live: model={model}")
    
    payload = [{
        "prompt": prompt,
        "model": model,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/v3/ai_optimization/perplexity/llm_responses/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        items = task_result.get("items", [])
        
        if items:
            return {
                "answer": items[0].get("answer", ""),
                "citations": items[0].get("citations", []),
                "model_used": model
            }
    
    return result


@mcp.tool()
async def cross_aggregated_metrics(
    targets: list[str],
    target_type: Literal["domain", "page"] = "domain",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Compare multiple domains/pages side-by-side.
    
    ✅ CORRECTED ENDPOINT: /v3/ai_optimization/llm_mentions/cross_aggregated_metrics/live
    """
    logger.info(f"cross_aggregated_metrics: {len(targets)} targets")
    
    payload = [{
        "targets": targets,
        "target_type": target_type,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    if date_from:
        payload[0]["date_from"] = date_from
    if date_to:
        payload[0]["date_to"] = date_to
    
    result = await make_request(
        "/v3/ai_optimization/llm_mentions/cross_aggregated_metrics/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        return {
            "items": task_result.get("items", []),
            "cost_credits": len(targets) * 2
        }
    
    return result


# ============================================================================
# TIER 3: POWER FEATURES
# ============================================================================

@mcp.tool()
async def top_pages(
    domain: str,
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get top-performing pages from a domain in LLM responses.
    
    ✅ CORRECTED ENDPOINT: /v3/ai_optimization/llm_mentions/top_pages/live
    """
    logger.info(f"top_pages: domain={domain}")
    
    payload = [{
        "target": domain,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/v3/ai_optimization/llm_mentions/top_pages/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        return {
            "total_count": task_result.get("total_count", 0),
            "items": task_result.get("items", []),
            "cost_credits": 2
        }
    
    return result


# ============================================================================
# TIER 4: BATCH OPERATIONS
# ============================================================================

@mcp.tool()
async def chatgpt_task_post(
    tasks: list[dict],
    tag: Optional[str] = None
) -> dict:
    """
    Submit batch ChatGPT queries for asynchronous processing.
    
    ✅ CORRECTED ENDPOINT: /v3/ai_optimization/chat_gpt/llm_responses/task_post
    """
    logger.info(f"chatgpt_task_post: {len(tasks)} tasks")
    
    payload = tasks
    if tag:
        for task in payload:
            task["tag"] = tag
    
    result = await make_request(
        "/v3/ai_optimization/chat_gpt/llm_responses/task_post",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0]
        return {
            "id": task_result.get("id"),
            "tasks_count": len(tasks),
            "status": "pending"
        }
    
    return result


@mcp.tool()
async def chatgpt_tasks_ready() -> dict:
    """
    Check which ChatGPT batch jobs are ready for retrieval.
    
    ✅ CORRECTED ENDPOINT: /v3/ai_optimization/chat_gpt/llm_responses/tasks_ready
    """
    logger.info("chatgpt_tasks_ready")
    
    result = await make_request(
        "/v3/ai_optimization/chat_gpt/llm_responses/tasks_ready",
        method="GET"
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [])
        return {
            "items": task_result,
            "cost_credits": 0
        }
    
    return result


@mcp.tool()
async def chatgpt_task_get(task_id: str) -> dict:
    """
    Retrieve results from a completed ChatGPT batch job.
    
    ✅ CORRECTED ENDPOINT: /v3/ai_optimization/chat_gpt/llm_responses/task_get/{task_id}
    """
    logger.info(f"chatgpt_task_get: task_id={task_id}")
    
    result = await make_request(
        f"/v3/ai_optimization/chat_gpt/llm_responses/task_get/{task_id}",
        method="GET"
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        return {
            "items": task_result.get("items", []),
            "cost_credits": 0
        }
    
    return result


if __name__ == "__main__":
    mcp.run()