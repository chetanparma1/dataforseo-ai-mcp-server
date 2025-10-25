"""
DataForSEO AI Optimization MCP Server - PRODUCTION
Tool names: chatgpt_llm_response, claude_llm_response, etc.
"""

import os
import base64
import logging
from typing import Optional, Literal

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


def format_llm_response(task_result: dict, task: dict) -> dict:
    """Minimal format: Answer → Citations → Metadata"""
    items = task_result.get("items", [])
    
    answer_text = ""
    citations = []
    
    if items:
        item = items[0]
        if item.get("sections"):
            answer_text = " ".join([s.get("text", "") for s in item["sections"]])
        if item.get("annotations"):
            citations = item.get("annotations", [])
    
    # Minimal formatting
    formatted_response = answer_text
    
    # Simple citations
    if citations:
        formatted_response += "\n\n------\nCitations:\n"
        for i, citation in enumerate(citations, 1):
            formatted_response += f"{i}. {citation.get('title', 'Source')}\n   {citation.get('url', '')}\n"
    
    # Simple metadata
    formatted_response += "\n----\nRESPONSE METADATA:\n"
    formatted_response += f"Model: {task_result.get('model_name')}\n"
    formatted_response += f"Tokens: {task_result.get('input_tokens', 0):,} in / {task_result.get('output_tokens', 0):,} out\n"
    formatted_response += f"Cost: ${task.get('cost', 0):.6f}\n"
    
    return {
        "response": formatted_response,
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


# ============================================================================
# MODEL LISTINGS (4 tools)
# ============================================================================

@mcp.tool()
async def chatgpt_models() -> dict:
    """Get list of available ChatGPT models."""
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
    """Get list of available Claude models."""
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
    """Get list of available Gemini models."""
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
    """Get list of available Perplexity models."""
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
# LLM LIVE RESPONSES (4 tools) - RENAMED
# ============================================================================

@mcp.tool()
async def chatgpt_llm_response(
    user_prompt: str,
    model_name: str = "gpt-5-2025-08-07",
    web_search: bool = True
) -> dict:
    """
    Get ChatGPT LLM response with citations and metadata.
    
    Args:
        user_prompt: Your question
        model_name: Model (default: gpt-5-2025-08-07)
        web_search: Enable web search for citations (default: True)
    """
    logger.info(f"ChatGPT: '{user_prompt[:50]}...'")
    
    payload = [{
        "user_prompt": user_prompt,
        "model_name": model_name,
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
        return format_llm_response(task_result, task)
    
    return result


@mcp.tool()
async def claude_llm_response(
    user_prompt: str,
    model_name: str = "claude-sonnet-4-20250514",
    web_search: bool = True
) -> dict:
    """
    Get Claude LLM response with citations and metadata.
    
    Args:
        user_prompt: Your question
        model_name: Model (default: claude-sonnet-4-20250514)
        web_search: Enable web search for citations (default: True)
    """
    logger.info(f"Claude: '{user_prompt[:50]}...'")
    
    payload = [{
        "user_prompt": user_prompt,
        "model_name": model_name,
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
        return format_llm_response(task_result, task)
    
    return result


@mcp.tool()
async def gemini_llm_response(
    user_prompt: str,
    model_name: str = "gemini-2.5-pro",
    web_search: bool = True
) -> dict:
    """
    Get Gemini LLM response with citations and metadata.
    
    Args:
        user_prompt: Your question
        model_name: Model (default: gemini-2.5-pro)
        web_search: Enable web search for citations (default: True)
    """
    logger.info(f"Gemini: '{user_prompt[:50]}...'")
    
    payload = [{
        "user_prompt": user_prompt,
        "model_name": model_name,
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
        return format_llm_response(task_result, task)
    
    return result


@mcp.tool()
async def perplexity_llm_response(
    user_prompt: str,
    model_name: str = "sonar-reasoning-pro"
) -> dict:
    """
    Get Perplexity LLM response with citations and metadata.
    
    Args:
        user_prompt: Your question
        model_name: Model (default: sonar-reasoning-pro)
    
    Note: Perplexity doesn't support web_search parameter.
    """
    logger.info(f"Perplexity: '{user_prompt[:50]}...'")
    
    payload = [{
        "user_prompt": user_prompt,
        "model_name": model_name
    }]
    
    result = await make_request(
        "/v3/ai_optimization/perplexity/llm_responses/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task = result["tasks"][0]
        task_result = task.get("result", [{}])[0]
        return format_llm_response(task_result, task)
    
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
    """Get AI search volume for keywords."""
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
            "language": language_name,
            "location": location_name,
            "items": task_result.get("items", []),
            "cost": task.get("cost", 0)
        }
    
    return result


# ============================================================================
# LLM MENTIONS (6 tools)
# ============================================================================

@mcp.tool()
async def search_mentions(
    target: str,
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """Search for brand/keyword mentions."""
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
            "language": language_name,
            "location": location_name,
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
    """Get top domains mentioned by LLMs."""
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
            "language": language_name,
            "location": location_name,
            "total_count": task_result.get("total_count", 0),
            "domains": task_result.get("items", []),
            "cost": task.get("cost", 0)
        }
    
    return result


@mcp.tool()
async def top_pages(target: str) -> dict:
    """Get top-performing pages from a domain."""
    logger.info(f"Top pages: {target}")
    
    payload = [{"target": target}]
    
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
    target_type: Literal["domain", "page"] = "domain"
) -> dict:
    """Get historical metrics for a domain or page."""
    logger.info(f"Aggregated metrics: {target}")
    
    payload = [{
        "target": target,
        "target_type": target_type
    }]
    
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
    """Compare multiple domains/pages."""
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
    logger.info("DataForSEO AI Optimization MCP Server - PRODUCTION")
    logger.info("=" * 80)
    logger.info(f"Account: {DATAFORSEO_LOGIN}")
    logger.info("")
    logger.info("LLM Response Tools (4):")
    logger.info("   chatgpt_llm_response, claude_llm_response")
    logger.info("   gemini_llm_response, perplexity_llm_response")
    logger.info("")
    logger.info("Total: 15 tools | 67 models | Minimal format")
    logger.info("=" * 80)
    
    mcp.run()