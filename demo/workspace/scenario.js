// The demo's story, in one place: a made-up company (Aozora Systems), made-up people, and what they write and say.
// Change the words here; the windows read them. The call's audio files are in audio/ (made by
// scripts/make_demo_voices.py, or recorded and dropped in under the same names); audio/lines.js holds their lengths
// and loudness, so the speaking rings follow the voice.
window.SCENARIO = {
  // The scene's clock: the taskbar and new messages start here and run on from it, so the times fit the story.
  clock: "10:06",
  me: { name: "Karthi", initials: "KM", color: "#4f6bed" },
  people: {
    yui: { name: "Yui Tanaka", initials: "YT", color: "#c2410c", role: "Product manager" },
    kenji: { name: "Kenji Sato", initials: "KS", color: "#0f766e", role: "QA lead" },
    haruto: { name: "Haruto Ito", initials: "HI", color: "#7c3aed", role: "Engineer" },
  },

  // Chat: the release channel, in Japanese. "start": shown from the beginning; "next": one per step (F8 in Chat).
  team: "Aozora Systems",
  channels: ["General", "release-4-2", "design", "support"],
  channel: "release-4-2",
  chat: {
    start: [
      { who: "kenji", time: "9:41", text: "おはようございます。昨夜の回帰テストの結果を共有します。不具合が2件見つかりました。" },
      { who: "haruto", time: "9:44", text: "確認しました。どちらもログイン画面の問題です。修正には2日ほどかかりそうです。" },
      { who: "yui", time: "9:52", text: "ありがとうございます。来週火曜日のリリースは厳しそうですね。木曜日に延期してもいいでしょうか？" },
    ],
    next: [
      { who: "yui", time: "9:58", text: "Karthiさん、お客様への連絡をお願いできますか？金曜日までに返事をもらえると助かります。" },
      { who: "kenji", time: "10:03", text: "了解です。修正が入ったら水曜日の夕方までに再テストします。" },
      { who: "yui", time: "10:05", text: "ありがとうございます！11時の定例で最終確認しましょう。" },
    ],
  },

  // Docs: a wordy draft in English, for Text Transform (concise, professional, bullet points...).
  doc: {
    title: "Customer update - Release 4.2",
    html: `<h1>Customer update: Release 4.2</h1>
<p>Hi everyone, I just wanted to quickly let you know that, after quite a long discussion with the QA team earlier today,
we have more or less decided that it would probably be best to move the release of version 4.2 from Tuesday to Thursday
next week, because there are still a couple of issues in the regression tests that really need to be fixed before we can
feel confident about shipping it to all of our customers.</p>
<p>The two issues are both on the login screen. Haruto thinks the fix will take about two days, and Kenji will run the
regression tests again on Wednesday evening once the fix is in, so that we have enough time to check everything properly
before Thursday.</p>
<p></p>`,
  },

  // The call: "Release 4.2 sync", in Japanese. Each line is one audio file; the English is only for the script.
  call: {
    title: "Release 4.2 sync",
    lines: [
      { who: "yui", file: "01-yui.wav", ja: "皆さん、おはようございます。今日はリリース4.2の日程について決めたいと思います。",
        en: "Good morning, everyone. Today I'd like to decide the schedule for release 4.2." },
      { who: "kenji", file: "02-kenji.wav", ja: "回帰テストで、ログイン画面の不具合が2件見つかりました。",
        en: "The regression tests found two bugs on the login screen." },
      { who: "haruto", file: "03-haruto.wav", ja: "修正には2日かかります。火曜日までには終わらせます。",
        en: "The fix will take two days. I'll finish it by Tuesday." },
      { who: "yui", file: "04-yui.wav", ja: "では、リリースを来週の木曜日に延期しましょう。",
        en: "Then let's move the release to next Thursday." },
      { who: "kenji", file: "05-kenji.wav", ja: "了解です。水曜日の夕方までに再テストを終わらせます。",
        en: "Understood. I'll finish retesting by Wednesday evening." },
      { who: "yui", file: "06-yui.wav", ja: "Karthiさん、お客様への連絡をお願いします。金曜日までにお願いできますか？",
        en: "Karthi, please contact the customer. Could you do it by Friday?" },
    ],
    gap: 1.4, // seconds between lines when they play one after another
  },

  // Notes: empty, for dictation.
  notes: { title: "Untitled" },
};
