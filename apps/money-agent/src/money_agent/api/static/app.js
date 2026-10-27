// Groundwire Money chat page.
// Safety: every piece of server or user text is inserted with textContent (never
// innerHTML), so nothing a model or a statement row contains can run as code.
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const TOOL_WORDS = {
    get_spending_summary: "spending summary",
    list_transactions: "transactions",
    get_monthly_trend: "monthly trend",
    find_recurring_payments: "recurring payments",
    plan_savings_goal: "goal plan",
    remember: "memory",
    delete_memory: "memory",
    list_memories: "memory",
  };

  // ------------------------------------------------------------------ //
  // Session: a random id per browser (demo identity; see api/identity.py)
  // ------------------------------------------------------------------ //

  function storageGet(key) {
    try { return window.localStorage.getItem(key); } catch { return null; }
  }
  function storageSet(key, value) {
    try { window.localStorage.setItem(key, value); } catch { /* private mode: session only */ }
  }
  function newId() {
    const bytes = new Uint8Array(16);
    crypto.getRandomValues(bytes);
    return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  }
  const userId = storageGet("gm_user") || newId();
  storageSet("gm_user", userId);
  let conversationId = null;

  // ------------------------------------------------------------------ //
  // HTTP
  // ------------------------------------------------------------------ //

  class ApiError extends Error {
    constructor(message, code) { super(message); this.code = code; }
  }

  async function api(path, options = {}) {
    const headers = { "X-User-Id": userId, ...(options.headers || {}) };
    let response;
    try {
      response = await fetch(path, { ...options, headers });
    } catch {
      throw new ApiError("Can't reach the server. Check your connection and try again.", "network");
    }
    if (response.status === 204) return null;
    let body = null;
    try { body = await response.json(); } catch { /* non-JSON error page */ }
    if (!response.ok) {
      const error = (body && body.error) || {};
      throw new ApiError(error.message || "Something went wrong. Please try again.", error.code);
    }
    return body;
  }

  // ------------------------------------------------------------------ //
  // Rendering helpers (text only)
  // ------------------------------------------------------------------ //

  function el(tag, props = {}, children = []) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else node.setAttribute(key, value);
    }
    for (const child of children) if (child) node.append(child);
    return node;
  }

  // Minimal, safe formatting: paragraphs, "- " bullets and **bold**.
  function inline(text) {
    const parts = text.split(/(\*\*[^*]+\*\*)/g).filter(Boolean);
    return parts.map((part) => part.startsWith("**") && part.endsWith("**")
      ? el("strong", { text: part.slice(2, -2) })
      : document.createTextNode(part));
  }
  function formatted(text) {
    const box = document.createDocumentFragment();
    let list = null;
    for (const raw of text.split("\n")) {
      const line = raw.trim();
      const bullet = line.match(/^([-*•]|\d+\.)\s+(.*)$/);
      if (bullet) {
        if (!list) { list = el("ul"); box.append(list); }
        list.append(el("li", {}, inline(bullet[2])));
      } else {
        list = null;
        if (line) box.append(el("p", {}, inline(line)));
      }
    }
    return box;
  }

  function rupees(value) {
    if (value === null || value === undefined) return "–";
    const [whole, paise = "00"] = String(value).replace("-", "").split(".");
    const head = whole.slice(0, -3), tail = whole.slice(-3);
    const grouped = head ? head.replace(/\B(?=(\d{2})+(?!\d))/g, ",") + "," + tail : tail;
    return (String(value).startsWith("-") ? "-" : "") + "₹" + grouped + "." + paise.padEnd(2, "0");
  }

  function notify(message, isError = false) {
    const box = $("notice");
    box.textContent = message;
    box.classList.toggle("error", isError);
    box.hidden = false;
    clearTimeout(notify.timer);
    notify.timer = setTimeout(() => { box.hidden = true; }, 7000);
  }

  // ------------------------------------------------------------------ //
  // Chat
  // ------------------------------------------------------------------ //

  function addMessage(role, text, extra = {}) {
    const item = el("li", { class: `msg ${role}${extra.pending ? " pending" : ""}` },
      [el("div", { class: "bubble" }, [role === "assistant" ? formatted(text) : el("p", { text })])]);
    if (extra.reply) item.append(meta(extra.reply));
    $("messages").append(item);
    item.scrollIntoView({ behavior: "smooth", block: "end" });
    return item;
  }

  function meta(reply) {
    const row = el("div", { class: "meta" });
    const v = reply.verification || {};
    if (v.status === "verified") {
      row.append(el("span", { class: "badge verified", text: "✓ Figures verified against your data" }));
      if (v.corrected) row.append(el("span", { class: "badge corrected", text: "↻ Corrected before showing" }));
    } else if (v.status === "failed") {
      row.append(el("span", { class: "badge failed", text: "⚠ Couldn't verify these figures — double-check them" }));
    }
    const used = [...new Set((reply.tools || []).map((t) => TOOL_WORDS[t.name] || t.name))];
    if (used.length) row.append(el("span", { class: "used", text: "Used: " + used.join(", ") }));
    return row;
  }

  async function send(text) {
    const message = text.trim();
    if (!message) return;
    addMessage("user", message);
    const pending = addMessage("assistant", "Checking your data…", { pending: true });
    setBusy(true);
    try {
      const reply = await api("/v1/me/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, conversation_id: conversationId }),
      });
      conversationId = reply.conversation_id;
      pending.remove();
      addMessage("assistant", reply.answer, { reply });
      if ((reply.tools || []).some((t) => t.name === "remember" || t.name === "delete_memory")) {
        loadMemories();
      }
    } catch (error) {
      pending.remove();
      addMessage("assistant", error.message);
    } finally {
      setBusy(false);
    }
  }

  function setBusy(busy) {
    $("send").disabled = busy;
    $("input").disabled = busy;
    if (!busy) $("input").focus();
  }

  // ------------------------------------------------------------------ //
  // Panel
  // ------------------------------------------------------------------ //

  async function loadSummary() {
    let summary;
    try {
      const overview = await api("/v1/me/overview");
      if (!overview.has_data) return; // nothing uploaded yet: keep the empty state
      summary = await api("/v1/me/summary");
    } catch (error) {
      notify(error.message, true);
      return;
    }
    const t = summary.totals;
    $("period").textContent = summary.period;
    $("income").textContent = rupees(t.income);
    $("spending").textContent = rupees(t.spending);
    $("invested").textContent = rupees(t.invested);
    $("saved").textContent = rupees(t.saved);
    $("rate").textContent = t.savings_rate_percent === null ? "–" : t.savings_rate_percent + "%";
    const bars = $("categories");
    bars.replaceChildren(...summary.spending_by_category.slice(0, 7).map((c) => {
      const fill = el("div", { class: "fill" });
      fill.style.width = Math.min(100, Number(c.share_percent)) + "%";
      return el("li", {}, [el("span", { text: c.label }),
        el("span", { class: "amount", text: rupees(c.total) }),
        el("div", { class: "track" }, [fill])]);
    }));
    $("summary-empty").hidden = true;
    $("summary").hidden = false;
  }

  async function loadMemories() {
    let memories = [];
    try { memories = await api("/v1/me/memories"); } catch (error) { notify(error.message, true); }
    $("memories").replaceChildren(...memories.map((m) => {
      const remove = el("button", { type: "button", "aria-label": "Forget this", text: "✕" });
      remove.addEventListener("click", async () => {
        try { await api(`/v1/me/memories/${m.id}`, { method: "DELETE" }); loadMemories(); }
        catch (error) { notify(error.message, true); }
      });
      return el("li", {}, [el("div", {}, [el("span", { class: "kind", text: m.kind }),
        el("span", { text: m.content })]), remove]);
    }));
    $("memories-empty").hidden = memories.length > 0;
  }

  async function upload(file, label) {
    if (!file) return;
    const form = new FormData();
    form.append("file", file, label || file.name);
    notify("Reading your statement…");
    try {
      const report = await api("/v1/me/statements", { method: "POST", body: form });
      const skipped = report.problem_count ? `, ${report.problem_count} rows unreadable` : "";
      const dupes = report.duplicates ? `, ${report.duplicates} already imported` : "";
      notify(`Imported ${report.imported} transactions (${report.first_date} to ${report.last_date}${dupes}${skipped}).`);
      await loadSummary();
    } catch (error) {
      notify(error.message, true);
    }
  }

  // ------------------------------------------------------------------ //
  // Wiring
  // ------------------------------------------------------------------ //

  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = $("input").value;
    $("input").value = "";
    send(text);
  });
  $("input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
  for (const chip of document.querySelectorAll(".chip")) {
    chip.addEventListener("click", () => send(chip.textContent));
  }
  $("upload").addEventListener("change", (event) => {
    upload(event.target.files[0]);
    event.target.value = "";
  });
  $("sample").addEventListener("click", async () => {
    try {
      const response = await fetch("/static/sample_statement.csv");
      if (!response.ok) throw new Error();
      await upload(await response.blob(), "sample_statement.csv");
    } catch {
      notify("Couldn't load the sample data.", true);
    }
  });

  loadSummary();
  loadMemories();
})();
