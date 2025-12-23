# groklexa
A grok (and local AI) friendly hands free web app for voice interactions with AI

Groklexa is a fully in-browser voice assistant that wakes up when you say its custom wake word (“Groklexa” or whatever you choose), letting you chat hands-free with Grok without ever pressing a button.
Features:

* Custom wake-word detection running clientside as a 2MB voice model in the browser (powered by openwakeword)
* Voice activity detection (VAD) for reliable start/stop
* Real-time speech-to-text → Grok inference → text-to-speech pipeline
* Animated visual feedback: a serene woman in headphones “wakes up” (head turn + subtle smile) when listening, drifts back to sleep when off
* Threaded conversation history with manual clear
* All processing configurable: transcription, inference, and synthesis can all be configured to be remote cloud or local
* 100% client-side wake word and VAD → no audio leaves your device until you speak after activation


Perfect for desktop always-on setups, kitchen counters, workshops, or just late-night philosophical rants when typing feels like too much work.
Open source, hackable, and deliberately fun.

<img width="1719" height="998" alt="image" src="https://github.com/user-attachments/assets/dc69893e-9a3d-4c71-8490-737120c7106d" />
