# metrobus-repairs-nlp-workshop
Workshop which demonstrates how to use nlp to categorize bus repairs for a city metro area.

Problem:  how can bus drivers notify maintenance that there may be a problem with their bus?

Solution:  at the end of their shift, when bus drivers check out, they can enter a textual description of what issue is occurring.  e.g. when I press on the brakes they squeal.  The NLP model will take in the textual description and classify it as:  brake, starter, engine, other, and send a report to maintenance.

Maintenance will pick up the report and then investigate the repair claim. 

Intent is for this workshop to eventually use all of the following services in order to demonstrate managed service usage in RHODS:
1. Grafana for data curation
2. JupyterHub to run experiments and create the NLP model
3. Seldon to deploy and manage the NLP model
4. Seldon to monitor and track performance of the model
