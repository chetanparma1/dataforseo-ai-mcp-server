# DataForSEO AI Optimization MCP Server

Complete MCP server for tracking brand visibility in LLMs (ChatGPT, Claude, Gemini, Perplexity).


## 🚀 Features
### ✅ LLM Live Responses (4 tools)

Query any LLM with citations in real-time:

- **chatgpt_live** - ChatGPT (gpt-4o-mini, gpt-4o, gpt-4-turbo)
- **claude_live** - Claude (claude-3-5-haiku, claude-3-5-sonnet)
- **gemini_live** - Gemini (gemini-1.5-flash, gemini-1.5-pro)
- **perplexity_live** - Perplexity (sonar, sonar-pro)

### 🔒 LLM Mentions (6 tools)

- **search_mentions** - Find brand mentions across all LLMs
- **top_domains** - Competitor analysis by domain
- **top_pages** - Top-performing pages in LLM responses
- **aggregated_metrics** - Historical tracking over time
- **cross_aggregated_metrics** - Compare multiple domains side-by-side

**Total: 10 focused, production-ready tools**
## 📋 Prerequisites

- Python 3.10+
- DataForSEO account with API credentials
- Claude Desktop (for testing)

## 🔧 Installation

### Step 1: Clone or Create Directory
```bash
mkdir dataforseo-mcp-server
cd dataforseo-mcp-server
```

### Step 2: Create Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Credentials

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Edit `.env` and add your DataForSEO credentials:
```env
DATAFORSEO_LOGIN=your_email@example.com
DATAFORSEO_PASSWORD=your_api_password_here
```

Get credentials from: https://app.dataforseo.com/api-dashboard

### Step 5: Test Authentication
```bash
python test_auth.py
```

Expected output:
```
✅ SUCCESS! Your DataForSEO credentials are working!
```

## 🚀 Running the Server

### Test Locally
```bash
python server.py
```

### Configure Claude Desktop

**macOS:**

Edit: `~/Library/Application Support/Claude/claude_desktop_config.json`
```json
{
  "mcpServers": {
    "dataforseo-ai": {
      "command": "/FULL/PATH/TO/venv/bin/python",
      "args": ["/FULL/PATH/TO/server.py"],
      "env": {
        "DATAFORSEO_LOGIN": "your_email@example.com",
        "DATAFORSEO_PASSWORD": "your_api_password"
      }
    }
  }
}
```

**Windows:**

Edit: `%APPDATA%\Claude\claude_desktop_config.json`
```json
{
  "mcpServers": {
    "dataforseo-ai": {
      "command": "C:\\FULL\\PATH\\TO\\venv\\Scripts\\python.exe",
      "args": ["C:\\FULL\\PATH\\TO\\server.py"],
      "env": {
        "DATAFORSEO_LOGIN": "your_email@example.com",
        "DATAFORSEO_PASSWORD": "your_api_password"
      }
    }
  }
}
```

**Get full paths:**
```bash
# macOS/Linux
pwd  # Current directory
which python  # Python path (use venv/bin/python)

# Windows
cd  # Current directory
where python  # Python path (use venv\Scripts\python.exe)
```

### Restart Claude Desktop

1. Quit Claude Desktop completely
2. Reopen Claude Desktop
3. Look for 🔌 icon in bottom-right
4. Click it - you should see "dataforseo-ai" listed

## 📖 Usage Examples

### Example 1: Check Brand Mentions
```
Use search_mentions to check if "Semrush" is mentioned in LLMs
```

### Example 2: Get AI Search Volume
```
Get AI search volume for: "SEO tools", "keyword research", "link building"
```

### Example 3: Multi-LLM Comparison
```
Compare responses: Ask ChatGPT, Claude, and Gemini "What are the best SEO tools?"
```

### Example 4: Competitor Analysis
```
Use top_domains to see which competitors dominate for "project management software"
```

### Example 5: Historical Tracking
```
Use aggregated_metrics to track semrush.com mentions from 2025-01-01 to 2025-03-01
```

## 💰 Cost Tracking

All tools log their credit costs:
- **search_mentions**: 2 credits ($0.002)
- **ai_keyword_search_volume**: 1 credit per keyword ($0.001)
- **chatgpt_live**: 5-20 credits depending on model ($0.005-$0.020)
- **top_domains**: 2 credits ($0.002)
- **aggregated_metrics**: 2 credits ($0.002)
- **Model listings**: FREE (0 credits)

## 🐛 Troubleshooting

### Error: "DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD must be set"
- Check `.env` file exists
- Verify credentials are correct
- No extra spaces around `=` signs

### Error: "API Error: Authentication failed"
- Verify credentials at https://app.dataforseo.com/api-dashboard
- Check account has credits available
- Confirm using API password, not account password

### Error: "ModuleNotFoundError: No module named 'fastmcp'"
- Activate virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

### Claude Desktop doesn't show the server
- Check config file path is correct
- Use absolute paths (not relative paths like `~/`)
- Restart Claude Desktop completely
- Check logs in Claude Desktop settings

## 📊 Logging

All tools log:
- Input parameters
- Output summaries
- Credit costs
- Timestamps

Check terminal output when server runs for detailed logs.

## 🔐 Security

- Never commit `.env` to git
- `.gitignore` is pre-configured
- Credentials are loaded from environment variables only

## 📚 API Documentation

Full DataForSEO API docs: https://docs.dataforseo.com/v3/ai_optimization/overview/

## 🆘 Support

Issues? Check:
1. DataForSEO API status: https://status.dataforseo.com/
2. DataForSEO support: https://dataforseo.com/support
3. Your account credits: https://app.dataforseo.com/

## 📝 License

MIT License - Use freely in your projects

## 🙏 Credits

Built for the SEO community by (https://github.com/chetanparma1)

Powered by [DataForSEO](https://dataforseo.com) API

---

**Questions?** Open an issue

**Want to contribute?** PRs welcome! See [CONTRIBUTING.md](CONTRIBUTING.md)

**Like this project?** ⭐ Star it on GitHub!            