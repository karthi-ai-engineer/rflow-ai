// Rflow's demo workspace: the windows, the four apps and the director. Everything Rflow touches is real: the text boxes
// take typing and pasting, selections copy, the call's voices play through the speakers (so live translation and
// Meeting mode hear them). Keys: F8 = the next step in the focused app (a chat message, a call line); F9 = the director.
(() => {
  const S = window.SCENARIO;
  const LINES = window.LINES || {};
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const isJa = (t) => /[぀-ヿ㐀-鿿]/.test(t);
  const person = (who) => (who === "me" ? S.me : S.people[who]);

  // ---------------------------------------------------------------- windows

  const wins = Object.fromEntries($$(".win").map((w) => [w.dataset.app, w]));
  const layout = Object.fromEntries(Object.entries(wins).map(([k, w]) => [k, w.getAttribute("style")]));
  let z = 10;

  function focus(app) {
    const w = wins[app];
    if (!w) return;
    w.hidden = false;
    w.classList.remove("minimised");
    w.style.zIndex = ++z;
    $$(".win").forEach((x) => x.classList.toggle("front", x === w));
    taskbar();
    const target = { notes: ".notes-text", docs: ".page", chat: "[data-compose]" }[app];
    if (target && !w.contains(document.activeElement)) $(target, w)?.focus({ preventScroll: true });
  }

  function front() {
    return $$(".win").filter((w) => !w.hidden && !w.classList.contains("minimised"))
      .sort((a, b) => (+b.style.zIndex || 0) - (+a.style.zIndex || 0))[0];
  }

  function taskbar() {
    const top = front();
    $$("[data-task]").forEach((b) => {
      const w = wins[b.dataset.task];
      b.classList.toggle("running", !w.hidden);
      b.classList.toggle("focused", w === top);
    });
  }

  $$(".win").forEach((w) => {
    w.addEventListener("pointerdown", () => { if (!w.classList.contains("front")) focus(w.dataset.app); }, true);
    const bar = $(".titlebar", w);
    bar.addEventListener("dblclick", (e) => { if (!e.target.closest(".ctl")) w.classList.toggle("max"); });
    bar.addEventListener("pointerdown", (e) => {
      if (e.target.closest(".ctl, .search") || w.classList.contains("max")) return;
      const x0 = e.clientX - w.offsetLeft, y0 = e.clientY - w.offsetTop;
      bar.setPointerCapture(e.pointerId);
      const move = (m) => {
        w.style.left = Math.min(Math.max(m.clientX - x0, -w.offsetWidth + 120), innerWidth - 120) + "px";
        w.style.top = Math.min(Math.max(m.clientY - y0, 0), innerHeight - 100) + "px";
      };
      const up = () => { bar.removeEventListener("pointermove", move); bar.removeEventListener("pointerup", up); };
      bar.addEventListener("pointermove", move);
      bar.addEventListener("pointerup", up);
    });
    $$(".ctl button", w).forEach((b) => b.addEventListener("click", () => {
      if (b.dataset.act === "min") w.classList.add("minimised");
      if (b.dataset.act === "max") w.classList.toggle("max");
      if (b.dataset.act === "close") { w.hidden = true; if (w.dataset.app === "call") call.leave(); }
      taskbar();
    }));
  });

  $$("[data-task]").forEach((b) => b.addEventListener("click", () => {
    const w = wins[b.dataset.task];
    if (w === front()) { w.classList.add("minimised"); taskbar(); } else focus(b.dataset.task);
  }));
  $$("[data-open]").forEach((b) => b.addEventListener("click", () => focus(b.dataset.open)));

  // ---------------------------------------------------------------- Notes

  const notes = $(".notes-text");
  $("[data-notes-title]").textContent = S.notes.title;
  function notesStatus() {
    const before = notes.value.slice(0, notes.selectionStart).split("\n");
    $("[data-pos]").textContent = `Ln ${before.length}, Col ${before.at(-1).length + 1}`;
    $("[data-chars]").textContent = `${notes.value.length.toLocaleString()} characters`;
  }
  ["input", "keyup", "click", "select"].forEach((e) => notes.addEventListener(e, notesStatus));

  // ---------------------------------------------------------------- Docs

  const page = $(".page");
  $("[data-doc-title]").textContent = S.doc.title;
  function docStatus() {
    const words = (page.innerText.match(/[\w’'-]+|[぀-ヿ㐀-鿿]/g) || []).length;
    $("[data-words]").textContent = `${words.toLocaleString()} words`;
  }
  page.addEventListener("input", docStatus);
  $$("[data-cmd]").forEach((b) => b.addEventListener("mousedown", (e) => {
    e.preventDefault(); // keep the selection in the page
    const cmd = b.dataset.cmd;
    if (cmd === "paste") return;
    page.focus();
    document.execCommand(cmd, false, cmd === "foreColor" ? "#dc2626" : cmd === "hiliteColor" ? "#fde047" : null);
    docStatus();
  }));
  $$("[data-style]").forEach((b) => b.addEventListener("mousedown", (e) => {
    e.preventDefault();
    page.focus();
    document.execCommand("formatBlock", false, b.dataset.style);
  }));
  // A paste keeps the structure (paragraphs, lists, bold...) but none of another app's fonts and colours.
  page.addEventListener("paste", (e) => {
    const html = e.clipboardData.getData("text/html");
    const text = e.clipboardData.getData("text/plain");
    e.preventDefault();
    if (html) document.execCommand("insertHTML", false, clean(html));
    else document.execCommand("insertText", false, text);
    docStatus();
  });
  function clean(html) {
    const keep = new Set(["P", "BR", "UL", "OL", "LI", "B", "STRONG", "I", "EM", "U", "H1", "H2", "H3"]);
    const doc = new DOMParser().parseFromString(html, "text/html");
    const walk = (node) => {
      for (const child of [...node.childNodes]) {
        if (child.nodeType === 1) {
          walk(child);
          if (!keep.has(child.tagName)) child.replaceWith(...child.childNodes);
          else [...child.attributes].forEach((a) => child.removeAttribute(a.name));
        } else if (child.nodeType === 8) child.remove();
      }
    };
    walk(doc.body);
    return doc.body.innerHTML;
  }

  // ---------------------------------------------------------------- Chat

  const messages = $("[data-messages]");
  const compose = $("[data-compose]");
  let nextChat = 0;
  $("[data-team]").innerHTML = `<i>${esc(S.team.split(" ").map((w) => w[0]).join(""))}</i>${esc(S.team)}`;
  $("[data-channels]").innerHTML = S.channels.map((c) => `<li class="${c === S.channel ? "on" : ""}"># ${esc(c)}</li>`).join("");
  $("[data-channel]").textContent = S.channel;

  function addMessage({ who, time, text }) {
    const p = person(who);
    const me = who === "me";
    const div = document.createElement("div");
    div.className = "msg" + (me ? " me" : "");
    div.innerHTML = `${me ? "" : `<div class="av" style="background:${p.color}">${esc(p.initials)}</div>`}
      <div class="bubble"><div class="meta">${me ? "" : `<b>${esc(p.name)}</b>`}${esc(time)}</div>
      <div class="text"${isJa(text) ? ' lang="ja"' : ""}>${esc(text)}</div></div>`;
    messages.append(div);
    messages.scrollTop = messages.scrollHeight;
  }

  // The scene's time: S.clock when the page opened, running on in real time.
  const opened = Date.now();
  function sceneDate() {
    const [h, m] = (S.clock || "10:06").split(":").map(Number);
    const d = new Date();
    d.setHours(h, m, 0, 0);
    return new Date(d.getTime() + (Date.now() - opened));
  }
  function now() {
    const d = sceneDate();
    return `${d.getHours()}:${String(d.getMinutes()).padStart(2, "0")}`;
  }

  function resetChat() {
    messages.innerHTML = `<div class="day">Today</div>`;
    S.chat.start.forEach(addMessage);
    nextChat = 0;
    compose.textContent = "";
    $("[data-typing]").hidden = true;
    director();
  }

  // The next scripted message: "… is typing" first, as a real chat does.
  function chatNext() {
    const m = S.chat.next[nextChat];
    if (!m) return;
    nextChat++;
    const typing = $("[data-typing]");
    typing.textContent = `${person(m.who).name} is typing`;
    typing.hidden = false;
    setTimeout(() => { typing.hidden = true; addMessage({ ...m, time: now() }); }, 1600);
    director();
  }

  function send() {
    const text = compose.innerText.replace(/\n+$/, "").trim();
    if (!text) return;
    addMessage({ who: "me", time: now(), text });
    compose.textContent = "";
  }
  compose.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
  });
  compose.addEventListener("paste", (e) => { // plain text only, as a chat box takes it
    e.preventDefault();
    document.execCommand("insertText", false, e.clipboardData.getData("text/plain"));
  });
  $("[data-send]").addEventListener("click", send);

  // ---------------------------------------------------------------- Call

  const call = (() => {
    const w = wins.call;
    const tiles = $("[data-tiles]");
    let joined = 0, timer = 0, index = 0, audio = null, playingAll = false, frame = 0;
    $("[data-call-title]").textContent = S.call.title;
    $("[data-lobby-title]").textContent = S.call.title;
    $("[data-lobby-who]").innerHTML = Object.values(S.people)
      .map((p) => `<span class="av" style="background:${p.color}">${esc(p.initials)}</span>`).join("");
    const everyone = [...Object.keys(S.people), "me"];
    tiles.innerHTML = everyone.map((who) => {
      const p = person(who);
      return `<div class="tile${who === "me" ? " me" : ""}" data-who="${who}"><div class="face" style="background:${p.color}">${esc(p.initials)}</div>
        <div class="name"><svg><use href="#i-mic${who === "me" ? "-off" : ""}"/></svg>${esc(who === "me" ? `${p.name} (You)` : p.name)}</div></div>`;
    }).join("");

    function show(inCall) {
      $("[data-lobby]").hidden = inCall;
      $("[data-stage]").hidden = !inCall;
      $("[data-callbar]").hidden = !inCall;
    }

    function join() {
      if (joined) return;
      focus("call");
      show(true);
      joined = Date.now();
      // Rflow's Meeting mode finds the call by its window's title (sst/meeting/apps.py, "Demo call"), only while joined.
      document.title = `${S.call.title} · Demo call`;
      timer = setInterval(() => {
        const s = Math.floor((Date.now() - joined) / 1000);
        $("[data-timer]").textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
      }, 500);
    }

    function stop() {
      playingAll = false;
      if (audio) { audio.pause(); audio = null; }
      cancelAnimationFrame(frame);
      $$(".tile", tiles).forEach((t) => { t.classList.remove("speaking"); t.style.setProperty("--level", 0); });
      director();
    }

    function leave() {
      stop();
      clearInterval(timer);
      joined = 0;
      index = 0;
      document.title = "Workspace";
      $("[data-timer]").textContent = "00:00";
      show(false);
      director();
    }

    // One line: its audio file plays, and its speaker's ring follows the voice's loudness (audio/lines.js).
    function play(i) {
      const line = S.call.lines[i];
      if (!line) { playingAll = false; director(); return; }
      if (!joined) join();
      stop();
      index = i + 1;
      const info = LINES[line.file];
      const tile = $(`.tile[data-who="${line.who}"]`, tiles);
      audio = new Audio(`audio/${line.file}`);
      const mine = audio;
      tile.classList.add("speaking");
      const tick = () => {
        if (audio !== mine) return;
        const env = info?.env || [];
        const level = env.length ? env[Math.min(env.length - 1, Math.floor(mine.currentTime * (info.fps || 20)))] : 0.5;
        tile.style.setProperty("--level", level.toFixed(2));
        frame = requestAnimationFrame(tick);
      };
      mine.addEventListener("ended", () => {
        if (audio !== mine) return;
        tile.classList.remove("speaking");
        tile.style.setProperty("--level", 0);
        audio = null;
        cancelAnimationFrame(frame);
        director();
        if (playingAll && S.call.lines[index]) setTimeout(() => { if (playingAll) play(index); }, S.call.gap * 1000);
        else playingAll = false;
      });
      mine.addEventListener("error", () => { tile.classList.remove("speaking"); director(`missing audio/${line.file}`); });
      mine.play().catch(() => director("the browser blocked playing: click the page once, then try again"));
      frame = requestAnimationFrame(tick);
      director();
    }

    const next = () => play(index);
    const all = () => { playingAll = true; play(index >= S.call.lines.length ? 0 : index); };
    $("[data-join]").addEventListener("click", join);
    $("[data-leave]").addEventListener("click", () => { leave(); w.hidden = true; taskbar(); });
    return { join, leave, next, all, stop, get index() { return index; }, get playing() { return !!audio; }, get joined() { return !!joined; } };
  })();

  // ---------------------------------------------------------------- the director (F9) and F8

  const panel = $("#director");
  function director(note) {
    $("[data-d-chat]").textContent = `${nextChat} of ${S.chat.next.length} sent`;
    $("[data-d-call]").textContent = note || (call.joined ? `line ${call.index} of ${S.call.lines.length}${call.playing ? " · playing" : ""}` : "not joined");
    $("[data-d-lines]").innerHTML = S.call.lines.map((l, i) => {
      const missing = !LINES[l.file];
      return `<div class="${i === call.index - 1 && call.playing ? "now" : ""}${missing ? " missing" : ""}">${i + 1}. ${esc(person(l.who).name)}: ${esc(l.en)}${missing ? " (no audio yet)" : ` (${LINES[l.file].seconds.toFixed(1)} s)`}</div>`;
    }).join("");
  }

  function reset() {
    call.leave();
    Object.entries(wins).forEach(([k, w]) => { w.setAttribute("style", layout[k]); w.className = `win ${k}`; w.hidden = k === "call"; });
    notes.value = "";
    notesStatus();
    page.innerHTML = S.doc.html;
    docStatus();
    resetChat();
    focus("notes");
  }

  panel.addEventListener("click", (e) => {
    const d = e.target.closest("button")?.dataset.d;
    if (!d) return;
    const [what, arg] = d.split(":");
    if (what === "reset") reset();
    if (what === "theme") document.documentElement.dataset.theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    if (what === "layout") Object.entries(wins).forEach(([k, w]) => { w.setAttribute("style", layout[k]); w.classList.remove("max", "minimised"); });
    if (what === "open") focus(arg);
    if (what === "maximise") front()?.classList.toggle("max");
    if (what === "chat" && arg === "next") chatNext();
    if (what === "call") ({ join: call.join, next: call.next, all: call.all, stop: call.stop })[arg]();
    taskbar();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "F9") { e.preventDefault(); panel.hidden = !panel.hidden; director(); }
    if (e.key === "F8") {
      e.preventDefault();
      const app = front()?.dataset.app;
      if (app === "chat") chatNext();
      if (app === "call") call.joined ? call.next() : call.join();
    }
  });

  // ---------------------------------------------------------------- the clock

  function clock() {
    const d = sceneDate();
    $("[data-time]").textContent = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    $("[data-date]").textContent = d.toLocaleDateString([], { year: "numeric", month: "numeric", day: "numeric" });
  }
  clock();
  setInterval(clock, 10_000);

  const params = new URLSearchParams(location.search);
  if (params.get("theme")) document.documentElement.dataset.theme = params.get("theme");
  reset();
  if (params.get("open")) params.get("open").split(",").forEach(focus);
  if (params.get("join")) call.join();
  if (params.get("director")) { panel.hidden = false; director(); }
})();
