"""Agent guardrails: decide, in code, which tool calls a model may make.

* :mod:`~groundwire.agents.policy`: ordered rules, argument limits, injection
  lockdown, loop limit (OWASP LLM06 / LLM10).
* :mod:`~groundwire.agents.schema`: checks arguments against a tool's JSON Schema.
"""
