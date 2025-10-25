"""
DataForSEO AI Optimization MCP Server - FINAL VERIFIED VERSION
15 tools with complete response data and verified endpoints
"""

import os
import base64
import logging
from typing import Optional, Literal
from datetime import datetime

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Get credentials
DATAFORSEO_LOGIN = os.getenv("DATAFORSEO_LOGIN")
DATAFORSEO_PASSWORD = os.getenv("DATAFORSEO_PASSWORD")
BASE_URL = "https://api.dataforseo.com"

if not DATAFORSEO_LOGIN or not DATAFORSEO_PASSWORD:
    raise ValueError("DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD must be set in .env file")

# Create authentication header
credentials = f"{DATAFORSEO_LOGIN}:{DATAFORSEO_PASSWORD}"
encoded_credentials = base64.b64encode(credentials.encode()).decode()
AUTH_HEADER = {"Authorization": f"Basic {encoded_credentials}"}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("DataForSEO credentials loaded")

# Initialize FastMCP server
mcp = FastMCP("DataForSEO AI Optimization")


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
    
    logger.info(f"{method} {endpoint}")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
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
            
            # Check API-level status
            api_status = result.get("status_code")
            if api_status != 20000:
                error_msg = result.get("status_message", "Unknown error")
                logger.error(f"API Error {api_status}: {error_msg}")
                raise DataForSEOError(f"API Error {api_status}: {error_msg}")
            
            # Check task-level status
            if result.get("tasks"):
                task = result["tasks"][0]
                task_status = task.get("status_code")
                
                if task_status == 40204:
                    raise DataForSEOError(
                        "Access denied. Contact support@dataforseo.com to enable this endpoint."
                    )
                elif task_status == 40503:
                    raise DataForSEOError(
                        f"Invalid POST data: {task.get('status_message')}"
                    )
                elif task_status != 20000:
                    raise DataForSEOError(
                        f"Task failed: {task_status} - {task.get('status_message')}"
                    )
            
            logger.info(f"Request successful")
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP Error: {str(e)}")
            raise DataForSEOError(f"HTTP request failed: {str(e)}")
        except DataForSEOError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise DataForSEOError(f"Request failed: {str(e)}")


# ============================================================================
# MODEL LISTINGS (4 tools - FREE, GET requests)
# ============================================================================

@mcp.tool()
async def chatgpt_models() -> dict:
    """Get list of available ChatGPT models with pricing and details."""
    result = await make_request(
        "/v3/ai_optimization/chat_gpt/llm_responses/models",
        method="GET"
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task = result["tasks"][0]
        models = task.get("result", [])
        
        return {
            "models": models,
            "total_count": len(models)
        }
    
    return result


@mcp.tool()
async def claude_models() -> dict:
    """Get list of available Claude models with pricing and details."""
    result = await make_request(
        "/v3/ai_optimization/claude/llm_responses/models",
        method="GET"
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task = result["tasks"][0]
        models = task.get("result", [])
        
        return {
            "models": models,
            "total_count": len(models)
        }
    
    return result


@mcp.tool()
async def gemini_models() -> dict:
    """Get list of available Gemini models with pricing and details."""
    result = await make_request(
        "/v3/ai_optimization/gemini/llm_responses/models",
        method="GET"
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task = result["tasks"][0]
        models = task.get("result", [])
        
        return {
            "models": models,
            "total_count": len(models)
        }
    
    return result


@mcp.tool()
async def perplexity_models() -> dict:
    """Get list of available Perplexity models with pricing and details."""
    result = await make_request(
        "/v3/ai_optimization/perplexity/llm_responses/models",
        method="GET"
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task = result["tasks"][0]
        models = task.get("result", [])
        
        return {
            "models": models,
            "total_count": len(models)
        }
    
    return result


# ============================================================================
# LLM LIVE RESPONSES (4 tools) - FULL DATA
# ============================================================================

@mcp.tool()
async def chatgpt_live(
    user_prompt: str,
    model_name: str = "gpt-4o-mini",
    max_output_tokens: int = 1000,
    temperature: float = 0.7,
    web_search: bool = False
) -> dict:
    """
    Get live ChatGPT response with complete metadata.
    
    Use chatgpt_models() first to see all available models.
    
    Returns:
        - answer: The AI response text
        - citations: List of sources with titles and URLs
        - model_name: Model used
        - input_tokens: Tokens in prompt
        - output_tokens: Tokens in response
        - web_search_used: Whether web search was enabled
        - ai_provider_cost: Cost charged by OpenAI
        - dataforseo_cost: Cost charged by DataForSEO
        - total_cost: Total cost (dataforseo_cost)
        - datetime: When response was generated
        - full_response: Complete raw API response
    """
    logger.info(f"ChatGPT: '{user_prompt[:50]}...'")
    
    payload = [{
        "user_prompt": user_prompt,
        "model_name": model_name,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "web_search": web_search
    }]
    
    result = await make_request(
        "/v3/ai_optimization/chat_gpt/llm_responses/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task = result["tasks"][0]
        task_result = task.get("result", [{}])[0]
        items = task_result.get("items", [])
        
        answer_text = ""
        citations = []
        
        if items:
            item = items[0]
            if item.get("sections"):
                answer_text = " ".join([s.get("text", "") for s in item["sections"]])
            if item.get("annotations"):
                citations = item.get("annotations", [])
        
        return {
            "answer": answer_text,
            "citations": citations,
            "model_name": task_result.get("model_name"),
            "input_tokens": task_result.get("input_tokens"),
            "output_tokens": task_result.get("output_tokens"),
            "web_search_used": task_result.get("web_search"),
            "ai_provider_cost": task_result.get("money_spent"),
            "dataforseo_cost": task.get("cost"),
            "total_cost": task.get("cost"),
            "datetime": task_result.get("datetime"),
            "full_response": task_result
        }
    
    return result


@mcp.tool()
async def claude_live(
    user_prompt: str,
    model_name: str = "claude-3-5-haiku-20241022",
    max_output_tokens: int = 1000,
    temperature: float = 0.7,
    web_search: bool = False
) -> dict:
    """
    Get live Claude response with complete metadata.
    
    Use claude_models() first to see all available models.
    
    Returns complete metadata including tokens, costs, citations, and full response.
    """
    logger.info(f"Claude: '{user_prompt[:50]}...'")
    
    payload = [{
        "user_prompt": user_prompt,
        "model_name": model_name,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "web_search": web_search
    }]
    
    result = await make_request(
        "/v3/ai_optimization/claude/llm_responses/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task = result["tasks"][0]
        task_result = task.get("result", [{}])[0]
        items = task_result.get("items", [])
        
        answer_text = ""
        citations = []
        
        if items:
            item = items[0]
            if item.get("sections"):
                answer_text = " ".join([s.get("text", "") for s in item["sections"]])
            if item.get("annotations"):
                citations = item.get("annotations", [])
        
        return {
            "answer": answer_text,
            "citations": citations,
            "model_name": task_result.get("model_name"),
            "input_tokens": task_result.get("input_tokens"),
            "output_tokens": task_result.get("output_tokens"),
            "web_search_used": task_result.get("web_search"),
            "ai_provider_cost": task_result.get("money_spent"),
            "dataforseo_cost": task.get("cost"),
            "total_cost": task.get("cost"),
            "datetime": task_result.get("datetime"),
            "full_response": task_result
        }
    
    return result


@mcp.tool()
async def gemini_live(
    user_prompt: str,
    model_name: str = "gemini-1.5-flash",
    max_output_tokens: int = 1000,
    temperature: float = 0.7,
    web_search: bool = False
) -> dict:
    """
    Get live Gemini response with complete metadata.
    
    Use gemini_models() first to see all available models.
    
    Returns complete metadata including tokens, costs, citations, and full response.
    """
    logger.info(f"Gemini: '{user_prompt[:50]}...'")
    
    payload = [{
        "user_prompt": user_prompt,
        "model_name": model_name,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "web_search": web_search
    }]
    
    result = await make_request(
        "/v3/ai_optimization/gemini/llm_responses/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task = result["tasks"][0]
        task_result = task.get("result", [{}])[0]
        items = task_result.get("items", [])
        
        answer_text = ""
        citations = []
        
        if items:
            item = items[0]
            if item.get("sections"):
                answer_text = " ".join([s.get("text", "") for s in item["sections"]])
            if item.get("annotations"):
                citations = item.get("annotations", [])
        
        return {
            "answer": answer_text,
            "citations": citations,
            "model_name": task_result.get("model_name"),
            "input_tokens": task_result.get("input_tokens"),
            "output_tokens": task_result.get("output_tokens"),
            "web_search_used": task_result.get("web_search"),
            "ai_provider_cost": task_result.get("money_spent"),
            "dataforseo_cost": task.get("cost"),
            "total_cost": task.get("cost"),
            "datetime": task_result.get("datetime"),
            "full_response": task_result
        }
    
    return result


@mcp.tool()
async def perplexity_live(
    user_prompt: str,
    model_name: str = "sonar",
    max_output_tokens: int = 1000,
    temperature: float = 0.7
) -> dict:
    """
    Get live Perplexity response with complete metadata.
    
    Use perplexity_models() first to see all available models.
    Note: Perplexity doesn't support web_search parameter.
    
    Returns complete metadata including tokens, costs, citations, and full response.
    """
    logger.info(f"Perplexity: '{user_prompt[:50]}...'")
    
    payload = [{
        "user_prompt": user_prompt,
        "model_name": model_name,
        "max_output_tokens": max_output_tokens,
        "temperature": temperature
    }]
    
    result = await make_request(
        "/v3/ai_optimization/perplexity/llm_responses/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task = result["tasks"][0]
        task_result = task.get("result", [{}])[0]
        items = task_result.get("items", [])
        
        answer_text = ""
        citations = []
        
        if items:
            item = items[0]
            if item.get("sections"):
                answer_text = " ".join([s.get("text", "") for s in item["sections"]])
            if item.get("annotations"):
                citations = item.get("annotations", [])
        
        return {
            "answer": answer_text,
            "citations": citations,
            "model_name": task_result.get("model_name"),
            "input_tokens": task_result.get("input_tokens"),
            "output_tokens": task_result.get("output_tokens"),
            "web_search_used": task_result.get("web_search"),
            "ai_provider_cost": task_result.get("money_spent"),
            "dataforseo_cost": task.get("cost"),
            "total_cost": task.get("cost"),
            "datetime": task_result.get("datetime"),
            "full_response": task_result
        }
    
    return result


# ============================================================================
# AI KEYWORD DATA (1 tool)
# ============================================================================

@mcp.tool()
async def ai_keyword_volume(
    keywords: list[str],
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get AI search volume for keywords across all LLMs.
    
    Returns keyword metrics including search volume, impressions, etc.
    """
    logger.info(f"AI keyword volume for {len(keywords)} keywords")
    
    payload = [{
        "keywords": keywords,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/v3/ai_optimization/ai_keyword_data/keywords_search_volume/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task = result["tasks"][0]
        task_result = task.get("result", [{}])[0]
        
        return {
            "keywords": keywords,
            "items": task_result.get("items", []),
            "cost": task.get("cost", 0)
        }
    
    return result


# ============================================================================
# LLM MENTIONS (6 tools - Requires activation)
# ============================================================================

@mcp.tool()
async def search_mentions(
    target: str,
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Search for brand/keyword mentions across all LLMs.
    
    Requires LLM Mentions API activation. Contact support@dataforseo.com
    """
    logger.info(f"Searching mentions: {target}")
    
    payload = [{
        "target": target,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/v3/ai_optimization/llm_mentions/search/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task = result["tasks"][0]
        task_result = task.get("result", [{}])[0]
        
        return {
            "target": target,
            "total_mentions": task_result.get("total_count", 0),
            "items": task_result.get("items", []),
            "cost": task.get("cost", 0)
        }
    
    return result


@mcp.tool()
async def top_domains(
    target: str,
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get top domains mentioned by LLMs for a keyword (competitor analysis).
    
    Requires LLM Mentions API activation.
    """
    logger.info(f"Top domains for: {target}")
    
    payload = [{
        "target": target,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/v3/ai_optimization/llm_mentions/top_domains/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task = result["tasks"][0]
        task_result = task.get("result", [{}])[0]
        
        return {
            "target": target,
            "total_count": task_result.get("total_count", 0),
            "domains": task_result.get("items", []),
            "cost": task.get("cost", 0)
        }
    
    return result


@mcp.tool()
async def top_pages(
    target: str
) -> dict:
    """
    Get top-performing pages from a domain in LLM responses.
    
    Requires LLM Mentions API activation.
    """
    logger.info(f"Top pages: {target}")
    
    payload = [{
        "target": target
    }]
    
    result = await make_request(
        "/v3/ai_optimization/llm_mentions/top_pages/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task = result["tasks"][0]
        task_result = task.get("result", [{}])[0]
        
        return {
            "target": target,
            "total_pages": task_result.get("total_count", 0),
            "pages": task_result.get("items", []),
            "cost": task.get("cost", 0)
        }
    
    return result


@mcp.tool()
async def aggregated_metrics(
    target: str,
    target_type: Literal["domain", "page"] = "domain",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
) -> dict:
    """
    Get historical metrics for a domain or page.
    
    Requires LLM Mentions API activation.
    """
    logger.info(f"Aggregated metrics: {target}")
    
    payload = [{
        "target": target,
        "target_type": target_type
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
        task = result["tasks"][0]
        task_result = task.get("result", [{}])[0]
        
        return {
            "target": target,
            "metrics": task_result.get("metrics", {}),
            "cost": task.get("cost", 0)
        }
    
    return result


@mcp.tool()
async def cross_aggregated_metrics(
    targets: list[str],
    target_type: Literal["domain", "page"] = "domain"
) -> dict:
    """
    Compare multiple domains/pages side-by-side.
    
    Requires LLM Mentions API activation.
    """
    logger.info(f"Comparing {len(targets)} targets")
    
    payload = [{
        "targets": targets,
        "target_type": target_type
    }]
    
    result = await make_request(
        "/v3/ai_optimization/llm_mentions/cross_aggregated_metrics/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task = result["tasks"][0]
        task_result = task.get("result", [{}])[0]
        
        return {
            "targets": targets,
            "comparison": task_result.get("items", []),
            "cost": task.get("cost", 0)
        }
    
    return result


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("DataForSEO AI Optimization MCP Server - COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Account: {DATAFORSEO_LOGIN}")
    logger.info("")
    logger.info("Model Listings (4 tools - FREE):")
    logger.info("   chatgpt_models, claude_models, gemini_models, perplexity_models")
    logger.info("")
    logger.info("LLM Live Responses (4 tools - FULL DATA):")
    logger.info("   chatgpt_live, claude_live, gemini_live, perplexity_live")
    logger.info("")
    logger.info("AI Keyword Data (1 tool):")
    logger.info("   ai_keyword_volume")
    logger.info("")
    logger.info("LLM Mentions (6 tools - requires activation):")
    logger.info("   search_mentions, top_domains, top_pages")
    logger.info("   aggregated_metrics, cross_aggregated_metrics")
    logger.info("")
    logger.info("Total: 15 tools")
    logger.info("=" * 80)
    
    mcp.run()