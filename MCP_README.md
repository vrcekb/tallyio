# MCP Servers Installation - USPEŠNO! 🎉

## 📦 Instalirani Serverji

### ✅ **Filesystem Server**
- **Lokacija**: `C:\mcp-servers\node_modules\@modelcontextprotocol\server-filesystem`
- **Funkcija**: Dostop do datotek in direktorijev
- **Konfiguracija**: Dostop do C:\ in E:\ZETA\Tallyio

### ✅ **GitHub Server**
- **Lokacija**: `C:\mcp-servers\node_modules\@modelcontextprotocol\server-github`
- **Funkcija**: GitHub API integration
- **Potrebuje**: GITHUB_PERSONAL_ACCESS_TOKEN

### ✅ **Brave Search Server**
- **Lokacija**: `C:\mcp-servers\node_modules\@modelcontextprotocol\server-brave-search`
- **Funkcija**: Web search capabilities
- **Potrebuje**: BRAVE_API_KEY

### ✅ **Context7 Server** 🆕
- **Lokacija**: `C:\mcp-servers\context7\dist\index.js`
- **Funkcija**: Up-to-date documentation and code examples
- **Posebnost**: Fetches latest docs from libraries and frameworks
- **Uporaba**: Dodaj `use context7` v prompt

### ✅ **Task Master AI** 🆕
- **Lokacija**: `C:\mcp-servers\claude-task-master\mcp-server\server.js`
- **Funkcija**: Advanced task management with multiple AI models
- **Posebnost**: Supports Anthropic, OpenAI, Perplexity, Google, Mistral
- **Uporaba**: Task creation, management, and AI-powered execution

### ✅ **Python MCP SDK**
- **Lokacija**: Python site-packages
- **Funkcija**: MCP protocol implementation

## 🚀 Kako Uporabiti

### 1. **Claude Desktop Konfiguracija**
Kopiraj `claude_desktop_config.json` v Claude Desktop config folder:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### 2. **Context7 Uporaba** 🆕
V svojih promptih dodaj `use context7` za najnovejše dokumentacije:

```txt
Create a basic Next.js project with app router. use context7
```

```txt
Create a script to delete the rows where the city is "" given PostgreSQL credentials. use context7
```

```txt
Help me implement TallyIO MEV scanning with latest Rust patterns. use context7
```

### 3. **Nastavi API Ključe**

#### GitHub Token:
1. Pojdi na https://github.com/settings/tokens
2. Ustvari nov "Personal access token"
3. Dodaj v `claude_desktop_config.json`:
```json
"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
```

#### Brave Search API:
1. Pojdi na https://api.search.brave.com/
2. Registriraj se za API key
3. Dodaj v `claude_desktop_config.json`:
```json
"BRAVE_API_KEY": "your_brave_api_key_here"
```

### 4. **Zagon Serverjev**

#### Batch Scripts:
- `start-filesystem.bat` - Zažene filesystem server
- `start-github.bat` - Zažene GitHub server
- `start-context7.bat` - Zažene Context7 server 🆕
- `start-taskmaster.bat` - Zažene Task Master AI server 🆕

#### Ročni Zagon:
```bash
# Context7 Server
cd C:\mcp-servers\context7
node dist\index.js

# Task Master AI Server
cd C:\mcp-servers\claude-task-master
node mcp-server\server.js

# Filesystem Server
cd C:\mcp-servers
node node_modules\@modelcontextprotocol\server-filesystem\dist\index.js C:\ E:\ZETA\Tallyio
```

### 5. **Preverjanje Delovanja**

Če je vse pravilno nastavljeno, boš v Claude Desktop videl:
- 📁 File operations (read, write, list directories)
- 🔍 Web search capabilities
- 🐙 GitHub repository access
- 📚 **Context7 documentation fetching** 🆕
- 🎯 **Task Master AI management** 🆕

## 🎯 Context7 Prednosti

- ✅ **Up-to-date dokumentacije** - Ne več zastarele kode
- ✅ **Version-specific examples** - Pravilne API calls
- ✅ **No hallucinations** - Resnične, delujoče funkcije
- ✅ **Direct integration** - Brez tab-switching

## 🎯 Task Master AI Prednosti

- ✅ **Multi-AI Support** - Anthropic, OpenAI, Perplexity, Google, Mistral
- ✅ **Advanced Task Management** - Create, track, and execute complex tasks
- ✅ **AI-Powered Execution** - Intelligent task breakdown and execution
- ✅ **Integration Ready** - Works seamlessly with Claude Desktop

## 📁 Struktura Datotek

```
C:\mcp-servers\
├── context7\                   # 🆕 Context7 MCP Server
│   ├── dist\
│   │   └── index.js
│   ├── src\
│   └── .env
├── claude-task-master\         # 🆕 Task Master AI
│   ├── mcp-server\
│   │   └── server.js
│   ├── src\
│   ├── bin\
│   └── .env
├── node_modules\
│   └── @modelcontextprotocol\
│       ├── server-filesystem\
│       ├── server-github\
│       └── server-brave-search\
├── claude_desktop_config.json
├── start-context7.bat          # 🆕
├── start-taskmaster.bat        # 🆕
├── start-filesystem.bat
├── start-github.bat
├── package.json
└── MCP_README.md
```

## 🎯 Naslednji Koraki

1. **Nastavi API ključe** v konfiguraciji
2. **Kopiraj config** v Claude Desktop
3. **Restartaj Claude Desktop**
4. **Testiraj Context7** z `use context7` v promptih 🆕
5. **Testiraj Task Master AI** za napredno upravljanje nalog 🆕

**MCP Serverji z Context7 in Task Master AI so pripravljeni za uporabo! 🚀**
