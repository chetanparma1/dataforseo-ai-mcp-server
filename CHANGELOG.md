# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2025-01-24

### Added
- Initial release with 21 DataForSEO AI tools
- Live AI model interactions (ChatGPT, Claude, Gemini, Perplexity)
- Brand mention tracking across all major LLMs
- Keyword search volume analysis (AI-specific)
- Domain and page performance tracking
- Historical metrics and trend analysis
- Async batch processing for cost efficiency
- Full MCP protocol support
- Type-safe Pydantic models
- Comprehensive error handling and logging

### Tools Included

**Tier 1: Core Tools (5 tools)**
- search_mentions - Track brand mentions across LLMs
- ai_keyword_search_volume - Get AI search volumes
- chatgpt_live - Live ChatGPT responses with citations
- claude_live - Live Claude responses with citations
- gemini_live - Live Gemini responses with citations

**Tier 2: Analysis Tools (5 tools)**
- top_domains - Competitor domain analysis
- aggregated_metrics - Historical performance tracking
- perplexity_live - Live Perplexity responses
- chatgpt_scraper_live - Deep HTML extraction from ChatGPT
- cross_aggregated_metrics - Multi-domain comparison

**Tier 3: Advanced Features (5 tools)**
- top_pages - Best performing pages per domain
- list_chatgpt_models - Available ChatGPT models
- list_claude_models - Available Claude models
- list_gemini_models - Available Gemini models
- list_perplexity_models - Available Perplexity models

**Tier 4: Batch Operations (6 tools)**
- chatgpt_task_post - Submit async ChatGPT tasks
- chatgpt_tasks_ready - Check completed ChatGPT batches
- chatgpt_task_get - Retrieve ChatGPT results
- claude_task_post - Submit async Claude tasks
- claude_tasks_ready - Check completed Claude batches
- claude_task_get - Retrieve Claude results

### Documentation
- Comprehensive README with installation guide
- Usage examples for all tools
- Troubleshooting guide
- Cost tracking information
- Contributing guidelines
- Code of conduct
- Security policy

[1.0.0]: https://github.com/chetanparma1/dataforseo-ai-mcp-server/releases/tag/v1.0.0
