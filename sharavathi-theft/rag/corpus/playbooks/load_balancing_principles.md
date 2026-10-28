---
title: Load-balancing and mitigation principles (recommendation-only)
tags: [mitigation, load-balancing, redispatch, voltage]
status: illustrative
---
# Load-balancing and mitigation principles (recommendation-only)

> Status: illustrative, project-authored playbook. Not an official KPTCL, CEA or SLDC procedure.
> Mitigation is future work in this project; nothing here is automated.

## Purpose
An unmetered extraction increases flows and losses upstream of the extraction point and
depresses voltage downstream. These principles describe what an operator might consider while an
investigation is in progress. Every action is a recommendation for human approval.

## Assess the impact first
Quantify the effect before acting: segment loading against thermal limits, the minimum bus
voltage, and the extra losses. In this project, impact can be studied by applying a candidate
change to the simulated network and re-running the AC power flow.

## Options, in order of least disruption
Voltage support comes first: adjusting reactive compensation such as switched capacitors near the
affected buses. Generation redispatch comes next: changing injections to relieve stressed
segments, which properly requires an optimal power flow with cost and limit data. Controlled load
management (shifting or shedding) is a last resort and must follow the utility's approved load
management procedures.

## Constraints
Any proposal must respect thermal limits, voltage limits and the stability margins of the long
corridor. Plans must be validated in simulation before an operator considers them. The system
never executes switching or dispatch itself.
