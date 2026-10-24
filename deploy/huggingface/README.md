---
title: Groundwire
emoji: 🛡️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: true
short_description: LLM gateway with guardrails, failover and human approval
---

# Groundwire: live demo

An open-source, OpenAI-compatible LLM gateway. This Space runs the real gateway
with a live dashboard: click a scenario and watch PII redaction, prompt-injection
blocking, fact-checking, failover, tool-call guardrails and human approval happen.

- Source and docs: https://github.com/apmg001/groundwire
- Answers come from Llama 3.1 8B on Groq when available, otherwise from a backup
  model (that switch is the gateway's failover at work).
- Agent scenarios use scripted models and fake tools: nothing real is refunded.
- The dashboard stores no prompt or answer text, only verdicts and counts.
