---
title: Triage of suspected metering and sensor faults
tags: [sensor, metering, data-quality]
status: illustrative
---
# Triage of suspected metering and sensor faults

> Status: illustrative, project-authored playbook. Not an official KPTCL, CEA or SLDC procedure.

## When this playbook applies
Use it when the likely-cause split points to a sensor error, when a meter-consistency (KCL)
residual is large at one bus, when the load-ratio spread between metered loads is unusually wide,
or when channels are missing.

## Signatures that distinguish a sensor fault from extraction
A biased power meter breaks the balance at a single bus: the bus injection no longer equals the
sum of the line flows measured at that bus. Extraction keeps every bus balanced but leaves
unexplained power inside a section and changes the angle drop across it. A drifting sensor
shows a residual that grows steadily over successive scans. A missing channel appears as a
missing-data flag rather than a residual.

## Triage steps
First, identify the implicated channel from the top contributing features (KCL, voltage or angle
deviation at a named bus). Second, compare it with redundant measurements: the line-end flows at
the same substation, neighbouring buses, and historical values at a similar load. Third, check
the instrument transformer and meter health records, recent calibration dates and communication
link status. Fourth, if the channel is confirmed faulty, mark it as bad data so it is excluded,
and raise a maintenance request for recalibration or replacement.

## Do not dispatch a line patrol yet
Line patrols for suspected extraction should wait until the metering at both ends of the flagged
section has been verified. A sensor fault that is mistaken for extraction wastes crew time and
erodes trust in the alarms.

## Recording
Record the faulty channel, the fault type (bias, drift, stuck or missing), the time it was first
observed and the corrective action.
