---
title: Evidence packet and reporting for anomaly cases
tags: [evidence, reporting, extraction]
status: illustrative
---
# Evidence packet and reporting for anomaly cases

> Status: illustrative, project-authored playbook. Not an official KPTCL, CEA or SLDC procedure.

## What to capture
Every case should keep: the scenario or event identifier and timestamp; the anomaly score and
status; the section residuals with their z-scores; the top contributing features (SHAP); the
likely-cause split; the estimated kilometre, the 80% interval and the ranked candidate taps;
the model version and dataset hash; and the raw metered measurements used.

## Language
Reports describe a "suspected unauthorized extraction" or an "unexplained anomaly". They must not
state that theft is confirmed. Model outputs are indicators that justify inspection, not proof.

## Chain of custody
Field findings, photographs and survey data are attached to the same case record with who
collected them and when. Any change to a case after review is logged.

## Feedback loop
Record the field outcome (found, not found, sensor fault, other) against the case. These outcomes
measure the real false-positive rate and guide retraining and threshold tuning.
