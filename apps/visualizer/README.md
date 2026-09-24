# Research visualizer

Planned inspection application, separate from a commercial visualizer experience.
The first version should play a clip with a shared timeline for beats, measured audio
features, target intensity, and model predictions. Semantic sound labels can be added
after an audio-event model exists. The UI is not implemented yet.

Start with exported files, not a running inference server. The fixture exporter writes
an `index.json` with relative paths, checksums, config, dataset identity, and per-clip
metadata. See `scripts/generate_test_audio.py` and `src/felt/audio/fixtures.py`.

The original feature array contains broad energy channels, not a detailed spectrogram.
Frequency-analysis views need a separately versioned spectral export. All rendered
tracks must honor their own timebase; envelope outputs start at the first frame end.
