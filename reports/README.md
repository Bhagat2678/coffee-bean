# Developer & AI Reporting Guidelines

This directory contains progress reports, change logs, and design documentation organized by project contributors.

## 📋 Rule for AI Agents & AI Assistants
Any AI developer (assistant, agent, or copilot) working on this codebase **MUST** automatically document the changes it makes. 

### Instructions:
1. Log all edits, additions, and removals in a single Markdown file named with the current date: `YYYY-MM-DD.md`.
2. Save this file under the `reports/AI_Agents/` directory.
3. If multiple sessions occur on the same day, append updates to the existing daily log file.

### Log Format:
The daily log file should follow this structure:
```markdown
# Change Log: YYYY-MM-DD
**Agent Name:** <Name (e.g. Antigravity)>
**Task Summary:** <Brief summary of the goal>

## 🛠️ Modded / Created Files
*   `path/to/file` — Description of changes
*   `path/to/another/file` — Description of changes

## ✅ Completed Tasks
- [x] Task description

## 📝 Details & Notes
<Additional context, test results, or instructions for future developers>
```
