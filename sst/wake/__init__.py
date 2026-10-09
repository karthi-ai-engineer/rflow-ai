"""The wake word (opt-in): with the cursor in a text box, say the keyword, speak (pauses are fine), say "stop it", and
the text is typed there. The keyword again goes on; "<keyword>, make it a Teams message" runs that Text Transform
command. Only the user's voice starts it (the voice check, on by default). Ctrl+Win works as before.

    tokens.py      a typed keyword as the spotter's word pieces (bpe.model read in process)
    words.py       which keywords can be used, their spellings, cutting keyword and end phrase from the text

The measurements behind the numbers are the spike's (issue #162). Nothing here imports Qt or the app; sherpa-onnx is
loaded only when a model is.
"""
from sst.wake.words import END_PHRASES, check_keyword, prompts, spellings, strip_end, strip_keyword

__all__ = ["END_PHRASES", "check_keyword", "prompts", "spellings", "strip_end", "strip_keyword"]
