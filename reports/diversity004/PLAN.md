# Sound diversity study 004

Freeze the warmup-linear-v2 model, feature representation, procedural target,
30-epoch schedule and three initialization seeds 7, 19 and 42. Use dataset seed
271828 with 480/120/120 families and four variants per family, for 2,880 clips.
Require no canonical family overlap with studies 001 or 003.

The new synthetic-diverse-v2 generator varies tempo from 60 to 200 BPM, percussion
pitch and decay, swing, hit strength and overall gain. Four deterministic profiles
cover percussion alone, a harmonic tone bed, a noise bed, and sparse hits with
varied dynamics. Record profiles in the manifest and report each separately.
No copyrighted source recordings or external samples enter this dataset.

Before training, evaluate study 003 seed 7 on the new validation split to measure
transfer. Seed 7 is the first predetermined seed, not a winner selected by score.
Do not tune the generator or architecture after inspecting this result. Then train
all three seeds from scratch on the diverse dataset. Use three-epoch timing checks
and exact checkpoint resume, as in study 003. Select epochs on validation loss only.

After all training completes, evaluate the three original study 003 models and the
three new models on the same new test split, reporting all seeds and all profiles.
The test set is consumed once for this frozen comparison. Do not select a winning
seed or adjust hyperparameters from its scores. Compare overall/active/startup RMSE,
quiet-region intensity and event F1 against the same procedural and beat baselines.
Retain prior aggregate acceptance targets: active RMSE <= 0.08, event F1 >= 0.95,
quiet mean <= 0.02, startup RMSE <= 0.02. Report per-profile failures even if the
aggregate passes. A profile lacking quiet frames has no quiet metric, not zero.

This changes sound coverage, not the learned objective. The teacher still rewards
energy-following, including noise. Success does not establish musical selectivity,
pleasant haptics, realistic music synthesis, or hardware suitability. Real recordings
and physical output need a separate evaluation and are not included in this study.

The illustrative preview will use seed 7 and the first tonal-bed test clip in
manifest order. This selection is fixed before test evaluation; it illustrates the
validation failure category rather than choosing a favorable test example.
