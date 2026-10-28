---
title: Response to a suspected line or segment outage
tags: [outage, network, protection]
status: illustrative
---
# Response to a suspected line or segment outage

> Status: illustrative, project-authored playbook. Not an official KPTCL, CEA or SLDC procedure.

## When this playbook applies
Use it when the likely cause is "other event", typically a line or segment outage. Typical
signs are a very large power residual in one section, abrupt flow redistribution and missing
measurements downstream.

## Confirm the event
Check breaker status indications and the SCADA event log for the affected section. Review
protection relay records (trip flags, fault location estimates) and confirm with the control
centre whether any planned outage or switching was in progress.

## Do not treat it as extraction
An outage produces residuals much larger than typical extraction and usually islands or
de-energizes part of the corridor. Extraction localization outputs are not meaningful during an
outage and should be ignored until the network is restored.

## Restoration and follow-up
Restoration follows the utility's switching and restoration procedures under control-centre
authority. After restoration, re-run detection to confirm the residuals have returned to normal.
Record the event so the scenario can be excluded from extraction statistics.
