# Research applications

Applications consume the research package's versioned artifacts. They must not be
required for Python installation, training, evaluation, or automated Python checks.

- `website/`: public research information, reports, and demos.
- `visualizer/`: an inspection tool for audio, beats, features, targets, and predictions.

These are documented application boundaries, not implemented applications yet.
Commercial applications, accounts, commerce, and custom ring engineering belong in
separate product repositories. Add a JavaScript workspace and lockfile only when the
first web application is implemented; no empty frontend build tooling is needed now.
