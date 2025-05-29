@echo off
echo Starting Context7 MCP Server...
cd /d C:\mcp-servers\context7
set DEFAULT_MINIMUM_TOKENS=10000
node dist\index.js
pause
