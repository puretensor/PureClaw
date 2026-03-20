FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client ffmpeg ripgrep curl git fonts-dejavu-core fonts-dejavu-extra \
    pandoc poppler-utils qpdf procps ca-certificates gnupg \
    && rm -rf /var/lib/apt/lists/*

# Node.js 22 LTS from NodeSource (needed for Gemini CLI runtime + doc generation)
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Node.js packages: doc generation + Gemini CLI
# Gemini: --ignore-scripts skips tree-sitter native build (blocked by Docker seccomp)
RUN npm install -g docx pptxgenjs \
    && npm install -g --ignore-scripts @google/gemini-cli

# Codex CLI — standalone Rust binary, no Node.js needed
RUN curl -fsSL https://github.com/openai/codex/releases/latest/download/codex-x86_64-unknown-linux-musl.tar.gz \
    | tar xz -C /usr/local/bin \
    && mv /usr/local/bin/codex-x86_64-unknown-linux-musl /usr/local/bin/codex \
    && chmod +x /usr/local/bin/codex

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Claude CLI binary (213MB ELF, only needs glibc)
# Staged into build context as claude-bin/claude before build
COPY claude-bin/claude /usr/local/bin/claude
RUN chmod +x /usr/local/bin/claude

# Document skills seed (pdf, docx, pptx, xlsx)
COPY .claude-skills-seed/ /app/.claude-skills-seed/

# Application code
COPY . .

# Create dirs for persistent data and output
RUN mkdir -p /data/state/observers /data/memory /output \
    /app/.ssh /app/.claude /app/.config/puretensor/gdrive_tokens && \
    useradd -m -u 1000 -d /app nexus && \
    chown -R nexus:nexus /app /data /output

USER nexus

ENV HOME=/app \
    PYTHONUNBUFFERED=1 \
    CLAUDE_BIN=/usr/local/bin/claude

EXPOSE 9876 9877

CMD ["python3", "nexus.py"]
