---
title: Safety, human approval and governance for advisor output
tags: [governance, safety, approval]
status: illustrative
---
# Safety, human approval and governance for advisor output

> Status: illustrative, project-authored playbook. Not an official KPTCL, CEA or SLDC procedure.

## Human approval is mandatory
All advisor and model outputs are recommendations. A qualified engineer or operator reviews them
and approves any field dispatch or operational change. The system has no connection to SCADA or
to real grid equipment and never executes switching, dispatch or shedding.

## Know the model's limits
Results come from a simulated, simplified corridor model whose parameters are assumed, not a
verified utility model. Detection and localization accuracy degrade for small extractions, under
higher sensor noise and with missing channels. Under major-only metering, adjacent candidate
locations can be hard to tell apart. Treat the interval and the ranked taps as a search area, not
a pinpoint.

## Escalation
Low-confidence or conflicting results go to a senior engineer before any dispatch. Repeated false
alarms at the same location are reviewed and may lead to threshold changes or retraining.

## Requests the advisor will not fulfil
The advisor does not issue commands to operate breakers, switches or generators, and it does not
override the utility's procedures. Such requests are answered with this governance guidance.
