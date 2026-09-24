# First physical playback evaluation

The training pipeline predicts a normalized intensity request at 100 Hz. It does
not yet predict actuator acceleration, comfort or preference. The next useful
product check is whether the exported patterns produce distinct, repeatable and
well-timed sensations on one selected device.

## Fixed first session

Use twelve short synthetic clips, three each from percussion, tonal-bed, noise-bed
and sparse-dynamics profiles. Freeze their manifest IDs before playback. Compare
beat-only pulses, the deterministic teacher, and the trained model. Play the same
source clip and supplied beat times for all three conditions. Randomize condition
order and hide the condition names in the participant view.

Keep the device, placement and playback intensity mapping fixed for the session.
Begin with low requested intensity and let the wearer stop immediately. Do not
silently normalize each pattern: this would erase intensity errors we need to see.
A phone held in a hand and a motor mounted to a ring are separate evaluation setups.
Record both the requested pattern and the device mapping version.

For each condition, record alignment, clarity of individual events, comfort and
whether sustained vibration obscures beats. Also record an explicit preferred
condition and the option that neither feels useful. These are exploratory judgments
from the wearer, not a validated perceptual score or a population-level claim.

Log clip ID, condition, trial order, source/pattern start times, mapping version,
device information, intensity setting and ratings. Measured playback timing is
separate from model inference time. Do not interpret a software timestamp as a
measurement of physical motor onset.

## Device adapter boundary

A native phone test app or an ESP32 host adapter should read the versioned pattern
artifact and schedule its 10 ms frames against a shared playback clock. The adapter
owns device-specific intensity limits and timing. Firmware, ring mechanical design
and the commercial product remain outside this public research repository.

The research repository owns fixtures, artifact schemas and anonymized evaluation
summaries. Product code should not import training internals. A later research
visualizer can display source audio, supplied beats and requested envelopes from
the same artifacts without controlling an actuator.

## What a result would change

If the procedural teacher and model feel equivalent, favor the simpler rule for the
initial product prototype. If neither feels useful, change the target objective
using recorded preferences before spending more compute on imitation. If playback
is inconsistent, fix device scheduling and calibration before judging the model.
Any preference-trained model requires a separate plan and newly held-out examples.
