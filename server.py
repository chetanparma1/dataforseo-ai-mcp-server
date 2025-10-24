"""
DataForSEO AI Optimization MCP Server
Provides tools for tracking brand visibility in LLMs (ChatGPT, Claude, Gemini, Perplexity)

Tiers Included:
- Tier 1: Core mention tracking & keyword volume (5 tools)
- Tier 2: Competitor analysis & historical data (5 tools)
- Tier 3: Advanced features & model listings (5 tools)
- Tier 4: Batch operations (6 tools)

Total: 21 high-impact tools
"""

import os
import base64
import logging
from typing import Optional, Literal, Any
from datetime import datetime, date

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
BASE_URL = "https://api.dataforseo.com/v3"

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
    method: str = "GET",
    data: Optional[dict] = None
) -> dict:
    """
    Make authenticated request to DataForSEO API
    
    Args:
        endpoint: API endpoint path
        method: HTTP method (GET or POST)
        data: Request payload for POST requests
        
    Returns:
        API response as dictionary
        
    Raises:
        DataForSEOError: If API returns error or request fails
    """
    url = f"{BASE_URL}{endpoint}"
    
    logger.info(f"Making {method} request to: {endpoint}")
    if data:
        logger.info(f"Request payload: {data}")
    
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
            
            logger.info(f"Response status: {result.get('status_code')}")
            logger.info(f"Response message: {result.get('status_message')}")
            
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
# MODEL TYPE DEFINITIONS
# ============================================================================

ChatGPTModel = Literal[
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
    "o1-preview",
    "o1-mini"
]

ClaudeModel = Literal[
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307"
]

GeminiModel = Literal[
    "gemini-2.0-flash-exp",
    "gemini-exp-1206",
    "gemini-exp-1121",
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest"
]

PerplexityModel = Literal[
    "sonar",
    "sonar-pro"
]


# ============================================================================
# TIER 1: MUST-HAVE TOOLS (Solve 80% of demand)
# ============================================================================

@mcp.tool()
async def search_mentions(
    keyword: str,
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Search for brand/keyword mentions across LLMs (ChatGPT, Claude, Gemini, Perplexity).
    
    **SOLVES: "Does ChatGPT mention my brand at all?"**
    
    This is the #1 requested feature - 43.4% of queries get ZERO brand mentions.
    Use this to discover if your brand appears in LLM responses.
    
    Args:
        keyword: Search term or brand name to track (e.g., "Semrush", "Nike", "Python programming")
        language_name: Language for search (default: "English")
        location_name: Geographic location for context (default: "United States")
    
    Returns:
        Dictionary containing:
        - total_count: Number of LLM mentions found
        - items: List of mentions with details (domain, title, type, description, etc.)
    
    Cost: $0.002 per request (2 credits)
    """
    logger.info("=" * 80)
    logger.info("TOOL: search_mentions")
    logger.info(f"INPUT: keyword={keyword}, language={language_name}, location={location_name}")
    logger.info("=" * 80)
    
    payload = [{
        "keyword": keyword,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/ai_optimization/search_mentions/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        total_count = task_result.get("total_count", 0)
        items = task_result.get("items", [])
        
        logger.info(f"OUTPUT: total_count={total_count}, items_returned={len(items)}")
        
        return {
            "total_count": total_count,
            "items": items,
            "cost_credits": 2,
            "timestamp": datetime.now().isoformat()
        }
    
    return result


@mcp.tool()
async def ai_keyword_search_volume(
    keywords: list[str],
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get AI-specific search volume for keywords (not traditional Google search volume).
    
    **SOLVES: "What's the AI search volume for my keywords?"**
    
    Traditional keyword volume doesn't translate to AI prompt volume.
    This returns actual AI search volume + 12-month trends.
    
    Args:
        keywords: List of keywords to check (max 1000 per request)
        language_name: Language for search (default: "English")
        location_name: Geographic location (default: "United States")
    
    Returns:
        Dictionary containing:
        - keyword: The searched keyword
        - search_volume: Monthly AI search volume
        - monthly_searches: 12-month trend data
        - competition: Competition level (low/medium/high)
        
    Cost: $0.001 per keyword (1 credit each)
    """
    logger.info("=" * 80)
    logger.info("TOOL: ai_keyword_search_volume")
    logger.info(f"INPUT: keywords={keywords}, language={language_name}, location={location_name}")
    logger.info("=" * 80)
    
    payload = [{
        "keywords": keywords,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/ai_optimization/keyword_data/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        items = task_result.get("items", [])
        
        logger.info(f"OUTPUT: keywords_processed={len(items)}")
        for item in items[:3]:
            logger.info(f"  - {item.get('keyword')}: {item.get('search_volume')} monthly searches")
        
        return {
            "items": items,
            "cost_credits": len(keywords),
            "timestamp": datetime.now().isoformat()
        }
    
    return result


@mcp.tool()
async def chatgpt_live(
    prompt: str,
    model: ChatGPTModel = "gpt-4o-mini",
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get live ChatGPT response with citations.
    
    **SOLVES: "What does ChatGPT say about my brand?"**
    
    ChatGPT has 80%+ LLM market share. This is the most critical LLM to monitor.
    
    Args:
        prompt: Question/prompt to send to ChatGPT
        model: ChatGPT model to use (default: "gpt-4o-mini" for cost efficiency)
        language_name: Language for response (default: "English")
        location_name: Geographic context (default: "United States")
    
    Returns:
        Dictionary containing:
        - answer: Full ChatGPT response text
        - citations: List of cited domains with URLs
        - model_used: Which model generated the response
        
    Cost: Varies by model (gpt-4o-mini: ~5 credits, gpt-4o: ~20 credits)
    """
    logger.info("=" * 80)
    logger.info("TOOL: chatgpt_live")
    logger.info(f"INPUT: prompt={prompt}, model={model}, language={language_name}, location={location_name}")
    logger.info("=" * 80)
    
    payload = [{
        "prompt": prompt,
        "model": model,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/ai_optimization/llm_responses/chatgpt/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        items = task_result.get("items", [])
        
        if items:
            answer = items[0].get("answer", "")
            citations = items[0].get("citations", [])
            
            logger.info(f"OUTPUT: answer_length={len(answer)} chars, citations_count={len(citations)}")
            
            return {
                "answer": answer,
                "citations": citations,
                "model_used": model,
                "timestamp": datetime.now().isoformat()
            }
    
    return result


@mcp.tool()
async def claude_live(
    prompt: str,
    model: ClaudeModel = "claude-3-5-haiku-20241022",
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get live Claude response with citations.
    
    **SOLVES: Multi-LLM comparison baseline**
    
    Claude is the #2 LLM. Compare Claude vs ChatGPT responses to find discrepancies.
    
    Args:
        prompt: Question/prompt to send to Claude
        model: Claude model to use (default: "claude-3-5-haiku-20241022" for speed)
        language_name: Language for response (default: "English")
        location_name: Geographic context (default: "United States")
    
    Returns:
        Dictionary containing:
        - answer: Full Claude response text
        - citations: List of cited domains with URLs
        - model_used: Which model generated the response
        
    Cost: Varies by model (~5-20 credits)
    """
    logger.info("=" * 80)
    logger.info("TOOL: claude_live")
    logger.info(f"INPUT: prompt={prompt}, model={model}, language={language_name}, location={location_name}")
    logger.info("=" * 80)
    
    payload = [{
        "prompt": prompt,
        "model": model,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/ai_optimization/llm_responses/claude/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        items = task_result.get("items", [])
        
        if items:
            answer = items[0].get("answer", "")
            citations = items[0].get("citations", [])
            
            logger.info(f"OUTPUT: answer_length={len(answer)} chars, citations_count={len(citations)}")
            
            return {
                "answer": answer,
                "citations": citations,
                "model_used": model,
                "timestamp": datetime.now().isoformat()
            }
    
    return result


@mcp.tool()
async def gemini_live(
    prompt: str,
    model: GeminiModel = "gemini-1.5-flash-latest",
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get live Gemini response with citations.
    
    **SOLVES: Multi-LLM comparison**
    
    Google Gemini is the #3 LLM. Essential for complete multi-LLM coverage.
    
    Args:
        prompt: Question/prompt to send to Gemini
        model: Gemini model to use (default: "gemini-1.5-flash-latest" for speed)
        language_name: Language for response (default: "English")
        location_name: Geographic context (default: "United States")
    
    Returns:
        Dictionary containing:
        - answer: Full Gemini response text
        - citations: List of cited domains with URLs
        - model_used: Which model generated the response
        
    Cost: Varies by model (~5-15 credits)
    """
    logger.info("=" * 80)
    logger.info("TOOL: gemini_live")
    logger.info(f"INPUT: prompt={prompt}, model={model}, language={language_name}, location={location_name}")
    logger.info("=" * 80)
    
    payload = [{
        "prompt": prompt,
        "model": model,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/ai_optimization/llm_responses/gemini/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        items = task_result.get("items", [])
        
        if items:
            answer = items[0].get("answer", "")
            citations = items[0].get("citations", [])
            
            logger.info(f"OUTPUT: answer_length={len(answer)} chars, citations_count={len(citations)}")
            
            return {
                "answer": answer,
                "citations": citations,
                "model_used": model,
                "timestamp": datetime.now().isoformat()
            }
    
    return result


# ============================================================================
# TIER 2: HIGH-VALUE TOOLS (Competitor analysis & historical data)
# ============================================================================

@mcp.tool()
async def top_domains(
    keyword: str,
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get top domains mentioned by LLMs for a keyword (competitor analysis).
    
    **SOLVES: "Which competitors are winning in AI?"**
    
    Discover which domains dominate LLM responses for your target keywords.
    Essential for competitive intelligence.
    
    Args:
        keyword: Search term to analyze
        language_name: Language for search (default: "English")
        location_name: Geographic location (default: "United States")
    
    Returns:
        Dictionary containing:
        - total_count: Number of domains found
        - items: List of domains with mention counts, impression shares, URLs
        
    Cost: $0.002 per request (2 credits)
    """
    logger.info("=" * 80)
    logger.info("TOOL: top_domains")
    logger.info(f"INPUT: keyword={keyword}, language={language_name}, location={location_name}")
    logger.info("=" * 80)
    
    payload = [{
        "keyword": keyword,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/ai_optimization/top_domains/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        total_count = task_result.get("total_count", 0)
        items = task_result.get("items", [])
        
        logger.info(f"OUTPUT: total_count={total_count}, domains_returned={len(items)}")
        
        return {
            "total_count": total_count,
            "items": items,
            "cost_credits": 2,
            "timestamp": datetime.now().isoformat()
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
    Get historical aggregated metrics for a domain or page across LLMs.
    
    **SOLVES: Historical tracking & trend analysis**
    
    Track how your brand's LLM visibility changes over time.
    Essential for measuring AI optimization ROI.
    
    Args:
        target: Domain (e.g., "semrush.com") or page URL to track
        target_type: Either "domain" or "page"
        date_from: Start date in YYYY-MM-DD format (default: 30 days ago)
        date_to: End date in YYYY-MM-DD format (default: today)
        language_name: Language for search (default: "English")
        location_name: Geographic location (default: "United States")
    
    Returns:
        Dictionary containing:
        - metrics: Aggregated impressions, mentions, citations
        - trend_data: Time series data for visualization
        
    Cost: $0.002 per request (2 credits)
    """
    logger.info("=" * 80)
    logger.info("TOOL: aggregated_metrics")
    logger.info(f"INPUT: target={target}, type={target_type}, date_from={date_from}, date_to={date_to}")
    logger.info("=" * 80)
    
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
        "/ai_optimization/aggregated_metrics/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        metrics = task_result.get("metrics", {})
        
        logger.info(f"OUTPUT: metrics={metrics}")
        
        return {
            "metrics": metrics,
            "target": target,
            "target_type": target_type,
            "cost_credits": 2,
            "timestamp": datetime.now().isoformat()
        }
    
    return result


@mcp.tool()
async def perplexity_live(
    prompt: str,
    model: PerplexityModel = "sonar",
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get live Perplexity response with citations.
    
    **SOLVES: Complete multi-LLM coverage**
    
    Perplexity is the #4 LLM and growing fast. Essential for comprehensive tracking.
    
    Args:
        prompt: Question/prompt to send to Perplexity
        model: Perplexity model to use (default: "sonar")
        language_name: Language for response (default: "English")
        location_name: Geographic context (default: "United States")
    
    Returns:
        Dictionary containing:
        - answer: Full Perplexity response text
        - citations: List of cited domains with URLs
        - model_used: Which model generated the response
        
    Cost: ~5-10 credits per request
    """
    logger.info("=" * 80)
    logger.info("TOOL: perplexity_live")
    logger.info(f"INPUT: prompt={prompt}, model={model}, language={language_name}, location={location_name}")
    logger.info("=" * 80)
    
    payload = [{
        "prompt": prompt,
        "model": model,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/ai_optimization/llm_responses/perplexity/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        items = task_result.get("items", [])
        
        if items:
            answer = items[0].get("answer", "")
            citations = items[0].get("citations", [])
            
            logger.info(f"OUTPUT: answer_length={len(answer)} chars, citations_count={len(citations)}")
            
            return {
                "answer": answer,
                "citations": citations,
                "model_used": model,
                "timestamp": datetime.now().isoformat()
            }
    
    return result


@mcp.tool()
async def chatgpt_scraper_live(
    prompt: str,
    model: ChatGPTModel = "gpt-4o-mini",
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get ChatGPT response with full HTML content extraction (deep analysis).
    
    **SOLVES: Full HTML + citations extraction**
    
    Unlike chatgpt_live, this returns the complete HTML structure of the response,
    useful for detailed content analysis and citation tracking.
    
    Args:
        prompt: Question/prompt to send to ChatGPT
        model: ChatGPT model to use (default: "gpt-4o-mini")
        language_name: Language for response (default: "English")
        location_name: Geographic context (default: "United States")
    
    Returns:
        Dictionary containing:
        - answer: Full ChatGPT response text
        - html: Raw HTML content
        - citations: Detailed citation data with positions
        - model_used: Which model generated the response
        
    Cost: ~10-25 credits per request (more expensive than chatgpt_live)
    """
    logger.info("=" * 80)
    logger.info("TOOL: chatgpt_scraper_live")
    logger.info(f"INPUT: prompt={prompt}, model={model}, language={language_name}, location={location_name}")
    logger.info("=" * 80)
    
    payload = [{
        "prompt": prompt,
        "model": model,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/ai_optimization/llm_scraper/chatgpt/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        items = task_result.get("items", [])
        
        if items:
            answer = items[0].get("answer", "")
            html = items[0].get("html", "")
            citations = items[0].get("citations", [])
            
            logger.info(f"OUTPUT: answer_length={len(answer)} chars, html_length={len(html)} chars, citations={len(citations)}")
            
            return {
                "answer": answer,
                "html": html,
                "citations": citations,
                "model_used": model,
                "timestamp": datetime.now().isoformat()
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
    Compare multiple domains/pages side-by-side (competitive benchmarking).
    
    **SOLVES: Multi-domain competitor analysis**
    
    Track your domain vs competitors in a single request.
    Perfect for Share of Voice calculations.
    
    Args:
        targets: List of domains or page URLs to compare (max 10)
        target_type: Either "domain" or "page"
        date_from: Start date in YYYY-MM-DD format (default: 30 days ago)
        date_to: End date in YYYY-MM-DD format (default: today)
        language_name: Language for search (default: "English")
        location_name: Geographic location (default: "United States")
    
    Returns:
        Dictionary containing:
        - items: List of metrics for each target
        - comparison: Side-by-side comparison data
        
    Cost: $0.002 per target (2 credits each)
    """
    logger.info("=" * 80)
    logger.info("TOOL: cross_aggregated_metrics")
    logger.info(f"INPUT: targets={targets}, type={target_type}, date_from={date_from}, date_to={date_to}")
    logger.info("=" * 80)
    
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
        "/ai_optimization/cross_aggregated_metrics/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        items = task_result.get("items", [])
        
        logger.info(f"OUTPUT: targets_compared={len(items)}")
        
        return {
            "items": items,
            "targets": targets,
            "target_type": target_type,
            "cost_credits": len(targets) * 2,
            "timestamp": datetime.now().isoformat()
        }
    
    return result


# ============================================================================
# TIER 3: POWER FEATURES (Model listings & advanced features)
# ============================================================================

@mcp.tool()
async def top_pages(
    domain: str,
    language_name: str = "English",
    location_name: str = "United States"
) -> dict:
    """
    Get top-performing pages from a domain in LLM responses.
    
    **SOLVES: Content optimization insights**
    
    Discover which pages from your domain (or competitors) get cited most by LLMs.
    Use this to understand what content types perform best.
    
    Args:
        domain: Domain to analyze (e.g., "semrush.com")
        language_name: Language for search (default: "English")
        location_name: Geographic location (default: "United States")
    
    Returns:
        Dictionary containing:
        - total_count: Number of pages found
        - items: List of pages with mention counts, URLs, titles
        
    Cost: $0.002 per request (2 credits)
    """
    logger.info("=" * 80)
    logger.info("TOOL: top_pages")
    logger.info(f"INPUT: domain={domain}, language={language_name}, location={location_name}")
    logger.info("=" * 80)
    
    payload = [{
        "target": domain,
        "language_name": language_name,
        "location_name": location_name
    }]
    
    result = await make_request(
        "/ai_optimization/top_pages/live",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        total_count = task_result.get("total_count", 0)
        items = task_result.get("items", [])
        
        logger.info(f"OUTPUT: total_count={total_count}, pages_returned={len(items)}")
        
        return {
            "total_count": total_count,
            "items": items,
            "cost_credits": 2,
            "timestamp": datetime.now().isoformat()
        }
    
    return result


@mcp.tool()
async def list_chatgpt_models() -> dict:
    """
    Get list of available ChatGPT models with details.
    
    Returns:
        Dictionary containing:
        - items: List of ChatGPT models with names, descriptions, pricing
        
    Cost: Free (0 credits)
    """
    logger.info("=" * 80)
    logger.info("TOOL: list_chatgpt_models")
    logger.info("=" * 80)
    
    result = await make_request("/ai_optimization/llm_responses/chatgpt/models")
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [])
        
        logger.info(f"OUTPUT: models_returned={len(task_result)}")
        
        return {
            "items": task_result,
            "cost_credits": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    return result


@mcp.tool()
async def list_claude_models() -> dict:
    """
    Get list of available Claude models with details.
    
    Returns:
        Dictionary containing:
        - items: List of Claude models with names, descriptions, pricing
        
    Cost: Free (0 credits)
    """
    logger.info("=" * 80)
    logger.info("TOOL: list_claude_models")
    logger.info("=" * 80)
    
    result = await make_request("/ai_optimization/llm_responses/claude/models")
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [])
        
        logger.info(f"OUTPUT: models_returned={len(task_result)}")
        
        return {
            "items": task_result,
            "cost_credits": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    return result


@mcp.tool()
async def list_gemini_models() -> dict:
    """
    Get list of available Gemini models with details.
    
    Returns:
        Dictionary containing:
        - items: List of Gemini models with names, descriptions, pricing
        
    Cost: Free (0 credits)
    """
    logger.info("=" * 80)
    logger.info("TOOL: list_gemini_models")
    logger.info("=" * 80)
    
    result = await make_request("/ai_optimization/llm_responses/gemini/models")
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [])
        
        logger.info(f"OUTPUT: models_returned={len(task_result)}")
        
        return {
            "items": task_result,
            "cost_credits": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    return result


@mcp.tool()
async def list_perplexity_models() -> dict:
    """
    Get list of available Perplexity models with details.
    
    Returns:
        Dictionary containing:
        - items: List of Perplexity models with names, descriptions, pricing
        
    Cost: Free (0 credits)
    """
    logger.info("=" * 80)
    logger.info("TOOL: list_perplexity_models")
    logger.info("=" * 80)
    
    result = await make_request("/ai_optimization/llm_responses/perplexity/models")
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [])
        
        logger.info(f"OUTPUT: models_returned={len(task_result)}")
        
        return {
            "items": task_result,
            "cost_credits": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    return result


# ============================================================================
# TIER 4: BATCH OPERATIONS (Cost-efficient bulk queries)
# ============================================================================

@mcp.tool()
async def chatgpt_task_post(
    tasks: list[dict],
    tag: Optional[str] = None
) -> dict:
    """
    Submit batch ChatGPT queries for asynchronous processing (cost-efficient).
    
    **SOLVES: Bulk queries at lower cost**
    
    Submit multiple prompts at once and retrieve results later.
    Up to 50% cheaper than live requests.
    
    Args:
        tasks: List of task dictionaries with prompt, model, language_name, location_name
        tag: Optional tag to identify this batch (for tracking)
    
    Returns:
        Dictionary containing:
        - id: Batch job ID for retrieval
        - tasks_count: Number of tasks submitted
        
    Cost: ~50% of live request costs
    """
    logger.info("=" * 80)
    logger.info("TOOL: chatgpt_task_post")
    logger.info(f"INPUT: tasks_count={len(tasks)}, tag={tag}")
    logger.info("=" * 80)
    
    payload = tasks
    if tag:
        for task in payload:
            task["tag"] = tag
    
    result = await make_request(
        "/ai_optimization/llm_responses/chatgpt/task_post",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0]
        task_id = task_result.get("id")
        
        logger.info(f"OUTPUT: batch_id={task_id}, tasks_submitted={len(tasks)}")
        
        return {
            "id": task_id,
            "tasks_count": len(tasks),
            "status": "pending",
            "timestamp": datetime.now().isoformat()
        }
    
    return result


@mcp.tool()
async def chatgpt_tasks_ready() -> dict:
    """
    Check which ChatGPT batch jobs are ready for retrieval.
    
    Returns:
        Dictionary containing:
        - items: List of completed batch jobs with IDs and metadata
        
    Cost: Free (0 credits)
    """
    logger.info("=" * 80)
    logger.info("TOOL: chatgpt_tasks_ready")
    logger.info("=" * 80)
    
    result = await make_request("/ai_optimization/llm_responses/chatgpt/tasks_ready")
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [])
        
        logger.info(f"OUTPUT: ready_batches={len(task_result)}")
        
        return {
            "items": task_result,
            "cost_credits": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    return result


@mcp.tool()
async def chatgpt_task_get(task_id: str) -> dict:
    """
    Retrieve results from a completed ChatGPT batch job.
    
    Args:
        task_id: Batch job ID from chatgpt_task_post
    
    Returns:
        Dictionary containing:
        - items: List of ChatGPT responses with answers and citations
        
    Cost: Free (0 credits) - you paid when submitting the batch
    """
    logger.info("=" * 80)
    logger.info("TOOL: chatgpt_task_get")
    logger.info(f"INPUT: task_id={task_id}")
    logger.info("=" * 80)
    
    result = await make_request(
        f"/ai_optimization/llm_responses/chatgpt/task_get/{task_id}"
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        items = task_result.get("items", [])
        
        logger.info(f"OUTPUT: results_returned={len(items)}")
        
        return {
            "items": items,
            "cost_credits": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    return result


@mcp.tool()
async def claude_task_post(
    tasks: list[dict],
    tag: Optional[str] = None
) -> dict:
    """
    Submit batch Claude queries for asynchronous processing.
    
    Args:
        tasks: List of task dictionaries with prompt, model, language_name, location_name
        tag: Optional tag to identify this batch
    
    Returns:
        Dictionary containing:
        - id: Batch job ID for retrieval
        - tasks_count: Number of tasks submitted
        
    Cost: ~50% of live request costs
    """
    logger.info("=" * 80)
    logger.info("TOOL: claude_task_post")
    logger.info(f"INPUT: tasks_count={len(tasks)}, tag={tag}")
    logger.info("=" * 80)
    
    payload = tasks
    if tag:
        for task in payload:
            task["tag"] = tag
    
    result = await make_request(
        "/ai_optimization/llm_responses/claude/task_post",
        method="POST",
        data=payload
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0]
        task_id = task_result.get("id")
        
        logger.info(f"OUTPUT: batch_id={task_id}, tasks_submitted={len(tasks)}")
        
        return {
            "id": task_id,
            "tasks_count": len(tasks),
            "status": "pending",
            "timestamp": datetime.now().isoformat()
        }
    
    return result


@mcp.tool()
async def claude_tasks_ready() -> dict:
    """
    Check which Claude batch jobs are ready for retrieval.
    
    Returns:
        Dictionary containing:
        - items: List of completed batch jobs
        
    Cost: Free (0 credits)
    """
    logger.info("=" * 80)
    logger.info("TOOL: claude_tasks_ready")
    logger.info("=" * 80)
    
    result = await make_request("/ai_optimization/llm_responses/claude/tasks_ready")
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [])
        
        logger.info(f"OUTPUT: ready_batches={len(task_result)}")
        
        return {
            "items": task_result,
            "cost_credits": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    return result


@mcp.tool()
async def claude_task_get(task_id: str) -> dict:
    """
    Retrieve results from a completed Claude batch job.
    
    Args:
        task_id: Batch job ID from claude_task_post
    
    Returns:
        Dictionary containing:
        - items: List of Claude responses
        
    Cost: Free (0 credits)
    """
    logger.info("=" * 80)
    logger.info("TOOL: claude_task_get")
    logger.info(f"INPUT: task_id={task_id}")
    logger.info("=" * 80)
    
    result = await make_request(
        f"/ai_optimization/llm_responses/claude/task_get/{task_id}"
    )
    
    if result.get("tasks") and len(result["tasks"]) > 0:
        task_result = result["tasks"][0].get("result", [{}])[0]
        items = task_result.get("items", [])
        
        logger.info(f"OUTPUT: results_returned={len(items)}")
        
        return {
            "items": items,
            "cost_credits": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    return result


if __name__ == "__main__":
    mcp.run()