/* The Rflow lens: what you say drifts in on the left, passes through the lens (the steps Rflow really takes) and comes
   out clean on the right, typed into a box; or, for a meeting and a screen, a little window of Rflow at work. Four examples,
   all true to Rflow. Builds itself inside #lens (or any
   [data-lens]); without JavaScript lens.css shows the first example still. Pauses off-screen and in hidden tabs. */
(function () {
  "use strict";
  // scene: a little window instead of words through the lens (SCENES below; say and clean are then only for the label).
  // say: the words; _filler, !misheard word, ^number, *name kept (sel: text selected or copied, not spoken; split: speech
  // without spaces, in the phrases it arrives in, joined with nothing). clean: the text typed, {highlighted}; out_lang: its
  // language. chips: colour token, then the label.
  var DICT = '<span>Hold <kbd>Ctrl</kbd>+<kbd>Win</kbd> and speak</span>';
  var EX = [
    {tab: "Meeting", mode: "Live translation", hint: "<span><kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>L</kbd> in any call</span>",
      scene: "meeting", say: "来週のリリースは木曜日に延期します。資料は金曜日までにお願いします。",
      clean: "We’re moving next week’s release to Thursday. Please send the materials by Friday.",
      lines: [["来週のリリースは木曜日に延期します。", "We’re moving next week’s release to Thursday."],
              ["資料は金曜日までにお願いします。", "Please send the materials by Friday."]],
      chips: ["ir:Japanese → English", "okc:Speak the translation: heard in English", "vi:Hidden from screen shares"]},
    {tab: "Screen", mode: "Screen translator", hint: "<span><kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>S</kbd> over any app</span>",
      scene: "screen", say: "申請は月末までに提出してください。領収書の画像を添付する必要があります。",
      clean: "Please submit it by the end of the month. You need to attach a photo of the receipt.",
      msgs: [["田", "vi", "田中 由衣", "Yui Tanaka", "10:12", "申請は月末までに提出してください。", "Please submit it by the end of the month."],
             ["佐", "ir", "佐藤 健二", "Kenji Sato", "10:14", "領収書の画像を添付する必要があります。", "You need to attach a photo of the receipt."],
             ["田", "vi", "田中 由衣", "Yui Tanaka", "10:15", "不明点はこのチャットで聞いてください。", "If anything is unclear, ask in this chat."]],
      chips: ["ir:Japanese → English", "okc:Same place, same colours", "vi:Any app, any window"]},
    {tab: "Reply", mode: "Text Transform", hint: "<span>Select, then say <b>“Japanese Teams message”</b></span>",
      cap: "You wrote", out: "Ready to send", done: "Replaced", wave: 1, sel: 1, out_lang: "ja",
      say: "Thursday works for me. I’ll send the approval request today.",
      clean: "木曜日で問題ありません。" +
        "{本日中}に承認依頼をお送りします。",
      chips: ["vi:English → Japanese", "okc:Polite, for Teams"]},
    {tab: "Dictation", mode: "Voice typing", hint: DICT, cap: "You say", out: "At the cursor", done: "Typed", wave: 1,
      say: "_um so tell *Priya the !post !grass migration is _uh done and _like ^twenty ^five ^percent of the users are on it",
      clean: "Tell {Priya} the {PostgreSQL} migration is done, and {25%} of the users are on it.",
      chips: ["vi:Filler removed", "ir:Your word: PostgreSQL", "wr:Number: 25%", "okc:Name kept: Priya"]}
  ];
  var CHECK = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';
  var KIND = {_: "f", "!": "x", "^": "n", "*": "k"};

  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html) e.innerHTML = html;
    return e;
  }
  function stack(cls, parts) {
    var s = el("div", "lens-stack " + cls);
    parts.forEach(function (p) { s.appendChild(typeof p === "string" ? el("span", "", p) : p); });
    return s;
  }
  // The clean text typed up to k characters: the rest is there but hidden, so the box never changes size.
  function typed(segs, k) {
    var shown = "", rest = "", n = 0;
    segs.forEach(function (s) {
      var a = Math.max(0, Math.min(s.t.length, k - n)), o = s.b ? "<b>" : "", c = s.b ? "</b>" : "";
      n += s.t.length;
      if (a) shown += o + s.t.slice(0, a) + c;
      if (a < s.t.length) rest += o + s.t.slice(a) + c;
    });
    return shown + '<i class="lens-caret"></i>' + (rest ? "<s>" + rest + "</s>" : "");
  }

  function esc(t) { return t.replace(/&/g, "&amp;").replace(/</g, "&lt;"); }
  // t shown up to k characters: the rest is there but hidden, so nothing moves while it's typed
  function upto(t, k, caret) {
    return esc(t.slice(0, k)) + (caret ? '<i class="lens-caret"></i>' : "") + "<s>" + esc(t.slice(k)) + "</s>";
  }
  function put(e, html) { if (e._h !== html) e.innerHTML = e._h = html; }
  function part(now, from, ms, n) { return Math.max(0, Math.min(n, Math.floor((now - from) / ms * n))); }
  var DOTS = '<span class="sc-dots"><i></i><i></i><i></i></span>';
  var SPEAKER = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z"/><path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11"/></svg>';
  var CURSOR = '<svg class="sc-cur" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3l14 8.5-6.2 1.3L9.6 19z"/></svg>';
  var EQ = '<span class="sc-eq"><i></i><i></i><i></i><i></i></span>';
  function avatar(text, k) { return '<span class="sc-av" style="--k:var(--' + k + ')">' + text + "</span>"; }
  // Each scene builds its window and returns play(now): the window at that moment, in ms (1e9: the end, for still).
  var SCENES = {
    // A call in Japanese. The first line: English on Rflow's bar, the voice still Japanese. Then Speak the translation is
    // turned on, and the next line is heard in English, the call's own sound lowered (as Rflow does).
    meeting: function (ex, w) {
      w.innerHTML = '<div class="sc-win"><div class="sc-top">' + DOTS + '<span>Weekly sync</span><b class="sc-rec"><i></i>Live</b></div>' +
        '<div class="sc-call"><div class="sc-tile sc-talk">' + avatar("YT", "vi") + '<em>Yui Tanaka</em><p class="sc-bub" lang="ja">' +
        ex.lines.map(function () { return "<span></span>"; }).join("") + '</p><span class="sc-heard">' + SPEAKER +
        '<b lang="ja">日本語</b><b>English</b>' + EQ + "</span></div>" +
        '<div class="sc-tile">' + avatar("KS", "ir") + "<em>Kenji Sato</em></div>" +
        '<div class="sc-tile">' + avatar("HI", "wr") + "<em>Haruto Ito</em></div>" +
        '<div class="sc-tile">' + avatar("You", "okc") + "<em>You</em></div></div>" +
        '<div class="sc-bar"><div class="sc-bar-head"><span class="sc-from"><i></i>Japanese → English</span>' +
        '<span class="sc-speak">' + SPEAKER + 'Speak the translation<i class="sc-switch"></i>' + CURSOR + "</span></div>" +
        ex.lines.map(function () { return '<p class="sc-l"></p>'; }).join("") + "</div></div>";
      var win = w.firstChild, tile = w.querySelector(".sc-talk"), bub = w.querySelector(".sc-bub"), said = bub.children;
      var shown = w.querySelectorAll(".sc-l"), START = [400, 3900], ON = 3300;
      return function (now) {
        var talk = false;
        ex.lines.forEach(function (l, j) {
          var at = now - START[j], ja = part(now, START[j], 1400, l[0].length), en = part(now, START[j] + 700, 1500, l[1].length);
          put(said[j], upto(l[0], ja));
          said[j].classList.toggle("on", at >= 0 && (j === ex.lines.length - 1 || now < START[j + 1]));
          put(shown[j], upto(l[1], en, en > 0 && en < l[1].length));
          shown[j].classList.toggle("old", j < ex.lines.length - 1 && now >= START[j + 1]);
          talk = talk || (at >= 0 && at < 1500);
        });
        var on = now >= ON, last = START[START.length - 1];
        bub.classList.toggle("on", now >= START[0]);
        tile.classList.toggle("is-talk", talk);
        win.classList.toggle("sc-aim", now >= ON - 700 && now < ON + 600);  // the cursor comes to the switch, clicks, goes
        win.classList.toggle("sc-on", on);
        // what you hear: Yui in Japanese, then, with the switch on, Gemini's voice in English
        win.classList.toggle("sc-voice", on ? now >= last + 800 && now < last + 2600 : talk);
        win.classList.toggle("sc-en", on && now >= last + 800);
      };
    },
    // A Japanese Teams chat: the Screen translator's frame is moved onto it, and each message turns English in place
    screen: function (ex, w) {
      var x = function (ja, en) { return '<span class="sc-x"><span lang="ja">' + ja + '</span><span class="sc-en">' + en + "</span></span>"; };
      w.innerHTML = '<div class="sc-win"><div class="sc-top">' + DOTS + '<span lang="ja">経費精算チーム</span></div><div class="sc-body">' +
        '<div class="sc-rail"><i></i><i></i><i></i><i></i></div><div class="sc-msgs">' +
        ex.msgs.map(function (m) {
          return '<div class="sc-msg">' + avatar(m[0], m[1]) + "<div><em>" + x(m[2], m[3]) + " · " + m[4] + "</em><p>" + x(m[5], m[6]) +
            "</p></div></div>";
        }).join("") +
        '<div class="sc-frame"><span class="sc-glass"><i class="sc-scan"></i></span><span class="sc-tag">Screen translator</span>' + CURSOR +
        "</div></div></div></div>";
      var win = w.firstChild, msgs = w.querySelectorAll(".sc-msg");
      return function (now) {
        win.classList.toggle("sc-go", now >= 700);
        win.classList.toggle("sc-scanning", now >= 1900 && now < 1e8);
        [].forEach.call(msgs, function (m, j) { m.classList.toggle("tr", now >= 2500 + j * 450); });
      };
    }
  };

  function Lens(root) {
    var label = root.getAttribute("aria-label") || "";
    var view = el("div", "lens-view"), tabs = el("div", "lens-tabs");
    view.setAttribute("role", "img");
    root.removeAttribute("role");
    root.removeAttribute("aria-label");
    tabs.setAttribute("role", "group");
    tabs.setAttribute("aria-label", "Examples");

    var X = EX.map(function (d) {  // this lens's own copy: a page may have more than one (lens-preview.html)
      var ex = Object.create(d), p = el("p");
      if (ex.lang) p.lang = ex.lang;
      ex.w = (ex.sel ? [ex.say] : ex.say.split(ex.split || " ")).map(function (t, i) {
        var k = ex.sel ? "s" : KIND[t[0]], w = el("span", "w" + (k ? " " + k : ""));
        w.textContent = k && !ex.sel ? t.slice(1) : t;
        // a little messy, the same way every time
        w.style.cssText = "--dx:" + (Math.sin(i * 1.3) * .8).toFixed(1) + "px;--dy:" + (Math.sin(i * 2.7) * 2).toFixed(1) +
          "px;--r:" + (Math.sin(i * 4.1) * .7).toFixed(2) + "deg";
        if (i && !ex.split) p.appendChild(document.createTextNode(" "));
        return p.appendChild(w);
      });
      ex.p = p;
      ex.segs = ex.clean.split(/(\{[^}]*\})/).filter(Boolean).map(function (s) {
        return s[0] === "{" ? {t: s.slice(1, -1), b: 1} : {t: s};
      });
      ex.len = ex.segs.reduce(function (n, s) { return n + s.t.length; }, 0);
      ex.text = el("p", "", typed(ex.segs, ex.len));
      if (ex.out_lang) ex.text.lang = ex.out_lang;
      ex.chipBox = el("div");
      if (ex.scene) {  // a scene keeps its chips under its window, and has no captions of the lens's
        ex.cap = ex.out = ex.done = "";
        ex.sc = el("div", "sc sc-" + ex.scene);
        ex.play = SCENES[ex.scene](ex, ex.sc);
        ex.chipBox.className = "sc-chips";
        ex.sc.appendChild(ex.chipBox);
      }
      ex.c = ex.chips.map(function (c) {
        var at = c.indexOf(":"), chip = el("span", "c");
        chip.textContent = c.slice(at + 1);
        chip.style.setProperty("--k", "var(--" + c.slice(0, at) + ")");
        return ex.chipBox.appendChild(chip);
      });
      if (ex.scene) ex.chipBox = el("div");  // the lens's own chip row: empty for a scene
      // the timeline, in ms: words in, the lens at work (P), typing (O to done), a pause, then the next example
      var n = ex.w.length, s = Math.min(120, 1500 / n), P = 300 + (n - 1) * s + (ex.sel ? 1500 : 700), type = Math.min(2200, ex.len * 26);
      var done = P + 800 + type + 150;
      ex.t = {s: s, P: P, O: P + 800, type: type, done: done, gap: Math.min(520, (done - P - 500) / ex.chips.length),
        end: Math.max(6800, done + 2400)};
      if (ex.scene === "meeting") ex.t = {s: s, P: 1400, O: 1e9, type: 1, done: 6600, gap: 2400, end: 10200};
      if (ex.scene === "screen") ex.t = {s: s, P: 2500, O: 1e9, type: 1, done: 4300, gap: 450, end: 8200};
      ex.btn = el("button", "", ex.tab + "<i></i>");
      ex.btn.type = "button";
      return ex;
    });

    var stacks = [
      stack("lens-mode", X.map(function (ex) { return "<span><i></i>" + ex.mode + "</span>"; })),
      stack("lens-hint", X.map(function (ex) { return ex.hint; })),
      stack("", X.map(function (ex) { return ex.cap; })),
      stack("lens-say", X.map(function (ex) { return ex.p; })),
      stack("", X.map(function (ex) { return ex.out; })),
      stack("lens-text", X.map(function (ex) { return ex.text; })),
      stack("lens-done", X.map(function (ex) { return CHECK + ex.done; })),
      stack("lens-chips", X.map(function (ex) { return ex.chipBox; }))
    ];
    var head = el("div", "lens-head"), stage = el("div", "lens-stage"), inp = el("div", "lens-in"), out = el("div", "lens-out");
    var capIn = el("div", "lens-cap"), capOut = el("div", "lens-cap"), box = el("div", "lens-box");
    head.append(stacks[0], stacks[1]);
    capIn.append(stacks[2], el("span", "lens-wave", "<i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>"));
    inp.append(capIn, stacks[3]);
    capOut.append(stacks[4]);
    box.append(stacks[5]);
    out.append(capOut, box, stacks[6]);
    stage.append(inp, el("div", "lens-mid", '<i class="lens-pipe"></i><div class="lens-orb"><i class="lens-ring"></i><div class="lens-core">' +
      '<span class="lens-bars"><i></i><i></i><i></i><i></i><i></i></span><span class="lens-dots"><i></i><i></i><i></i></span>' +
      '<span class="lens-check">' + CHECK + '</span></div></div><i class="lens-pipe"></i>'), out, stacks[7]);
    var scenes = stack("lens-scenes", X.map(function (ex) { return ex.sc || el("div"); })), body = el("div", "lens-body");
    stacks.push(scenes);
    body.append(stage, scenes);
    view.append(head, body);
    view.querySelectorAll(".lens-wave i, .lens-bars i, .sc-eq i").forEach(function (b, j) {
      b.style.setProperty("--h", (6 + Math.abs(Math.sin(j * 2.1)) * 12).toFixed(0) + "px");
      b.style.animationDelay = (j * -0.13).toFixed(2) + "s";
    });
    X.forEach(function (ex, j) {
      ex.btn.addEventListener("click", function () { show(j); run(); });
      tabs.appendChild(ex.btn);
    });
    root.textContent = "";
    root.append(view, tabs);

    var reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)");
    var still = root.dataset.motion === "reduce" || !!(reduce && reduce.matches), seen = true, raf = 0, last = 0;
    var i = 0, t = 0, k = -1, frozen = root.dataset.seek != null;

    function flag(name, on) { root.classList.toggle(name, !!on); }
    function show(n) {
      i = (n + X.length) % X.length;
      t = 0;
      k = -1;
      stacks.forEach(function (s) {
        [].forEach.call(s.children, function (c, j) { c.classList.toggle("on", j === i); });
      });
      X.forEach(function (ex, j) {
        ex.btn.setAttribute("aria-pressed", j === i);
        ex.btn.lastChild.style.transform = "";
        if (j !== i) ex.w.concat(ex.c).forEach(function (e) { e.classList.remove("in"); });
      });
      var ex = X[i];
      view.setAttribute("aria-label", label + " Example, " + ex.tab + ": \u201c" + ex.say.replace(/[_!^*|]/g, "") +
        "\u201d becomes \u201c" + ex.clean.replace(/[{}]/g, "") + "\u201d");
      flag("lens--still", still);
      draw(still ? 1e9 : 0);
    }
    function draw(now) {
      var ex = X[i], p = ex.t;
      flag("is-scene", ex.scene);
      if (ex.play) ex.play(now);
      flag("is-wave", ex.wave);
      flag("is-live", ex.wave && now < p.P);
      flag("is-proc", now >= p.P);
      flag("is-type", now >= p.O);
      flag("is-done", now >= p.done);
      flag("is-out", !still && now >= p.end - 500);
      ex.w.forEach(function (w, j) { w.classList.toggle("in", now >= 300 + j * p.s); });
      ex.c.forEach(function (c, j) { c.classList.toggle("in", now >= p.P + 300 + j * p.gap); });
      var nk = Math.max(0, Math.min(ex.len, Math.floor((now - p.O) / p.type * ex.len)));
      if (nk !== k) ex.text.innerHTML = typed(ex.segs, k = nk);
      ex.btn.lastChild.style.transform = still ? "" : "scaleX(" + Math.min(1, now / p.end).toFixed(3) + ")";
    }
    function frame(now) {
      raf = 0;
      t += Math.min(64, now - (last || now));  // after a pause, carry on where it was
      last = now;
      if (t >= X[i].t.end) show(i + 1);
      draw(t);
      run();
    }
    function run() {
      var go = seen && !document.hidden && !still && !frozen;
      flag("is-paused", !go);
      if (go && !raf) raf = requestAnimationFrame(frame);
      if (!go) {
        cancelAnimationFrame(raf);
        raf = last = 0;
      }
    }

    show(+root.dataset.start || 0);
    if (frozen) {  // a fixed moment, for checking (lens-preview.html)
      var at = +root.dataset.seek;
      while (at >= X[i].t.end) {
        at -= X[i].t.end;
        show(i + 1);
      }
      draw(t = at);
    }
    if (reduce && reduce.addEventListener) reduce.addEventListener("change", function () { still = reduce.matches; show(i); run(); });
    document.addEventListener("visibilitychange", run);
    if (window.IntersectionObserver) {
      new IntersectionObserver(function (es) { seen = es[es.length - 1].isIntersecting; run(); }).observe(root);
    }
    run();
  }

  function start() {
    [].forEach.call(document.querySelectorAll("#lens, [data-lens]"), function (root) {
      if (!root.firstChild) Lens(root);
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
