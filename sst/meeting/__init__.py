"""Meeting mode: one switch per part, started and stopped with the meeting the user is in (Teams, Zoom, Meet, Webex).

    apps.py      the meeting's window, found by itself (MeetingWindow); alive() says when it closed
    recorder.py  the meeting's sound on this laptop: others.wav (what the laptop plays) and you.wav (the microphone)
    screen.py    the meeting window seen every few seconds, a frame kept when the screen changed (WindowFrames)
    minutes.py   after the meeting: transcript with speakers, minutes, decisions, action items (MinutesResult)
    session.py   MeetingSession: the parts together, their state ("recording" -> "writing minutes" -> "done"/"failed")
                 and what the meeting costs so far

Live translation is sst.live's own bar, started and stopped with the meeting. Like it, nothing here imports sst.pipeline,
sst.dictate or sst.audio (tests/test_meeting_session.py fails if anything does): a meeting is not dictation.
"""
