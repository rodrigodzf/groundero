# Groundero

Visual grounding search for papers with Zotero integration and RAG-based question answering.

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Zotero desktop app (running locally)
- Google AI Studio API key

### Recommended

- [Better BibTeX](https://retorque.re/zotero-better-bibtex/) plugin for Zotero
- All items should have pinned citation keys (stored in the Extra field) for best results, auto-generated keys will be used as fallback for UI display.

### Setup

#### Install

```bash
uv tool install groundero
```

#### Development Install

```bash
uv sync
```

2. **Setup and build database**:

```bash
groundero
```

## Usage

### Commands

- `groundero` - Interactive query mode
- `groundero setup` - Initial setup and database build
- `groundero update [--limit N]` - Update database from Zotero
- `groundero query "question"` - Ask a single question
- `groundero list` - List indexed documents
- `groundero help` - Show help

### Examples

```bash
groundero update

# Ask questions about your papers
groundero query "Why are PINNs hard to train"

# Interactive mode for multiple queries
groundero
```

## Configuration

Environment is configured during `groundero setup` - no manual `.env` file needed.

### Zotero Local API Setup

**Required**: Groundero currently only works with local Zotero. Make sure the local API is enabled:

1. Open Zotero
2. Go to: **Settings** → **Advanced** → **Config Editor**
3. Set `extensions.zotero.httpServer.localAPI.enabled` to `true`
4. Restart Zotero

Without this setting, Groundero won't be able to connect to your local Zotero library.