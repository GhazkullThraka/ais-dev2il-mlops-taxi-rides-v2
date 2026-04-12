# General 

I am holding a course on the topics DevOps and MLOps for students of AIS (Artificial Intelligence Sytems) in the second semester. The topics that are handled are 

- Motivation for DevOps / MLOps
- Principles of Flow, Feedback, Continuous Learning and Experimentation
- Version Control using Git
- Zero Configuration, Build Tooling
- Automated Testing
- CI, Github Build Pipelines
- Packaging with Docker
- IaC using Docker Compose (potentially K8S if we have time)
- Observability / Production Telemetry (Logs, Metrics, Traces)
- Monitoring

You help me creating contents and especially creating simplest contents that students can grasp. We only have 9 sessions a 150 minutes time to tackle these topics, therefore everything must be kept as simple as possible. The contents are taught in the form of 

- lectured content (slides)
- exercises that are directly done in pairing sessions in the sessions
- exercises that follow up on the session contexts and that are done as homework

Each session follows this principle outline
- Grasp the problem at hand by experiencing how it is if you don't have the practices in place (e.g. diagnose an app that misbehaves but does not have observability)
- Have a block with theory (lecture style)
- Have an example that students can work on on their own in pairing mode in the session
- Have additional examples in a "Level Up" section available for students that are faster and for homework
- If required, there can be multiple of such blocks (theory + self study) in a single session (e.g. per topic)
The students are not very far in their studies. They only had 1 semester of Python basics.

# Instructions for this Repository

This repository focuses on these MLOps topics
- Organising data using Parquet
  - Goals
    - Understand what Parquet is and why it is useful for data management in ML projects
    - Open a Parquet file in PyCharm and inspect its contents
    - Understand how to work with Parquet files in Python using pandas
    - Have the pre-processed taxi ride data from DVC available in the data folder
- Data Management via DVC (via dagshub)
  - Goals
    - Understand what DVC is and why it is useful for data management in ML projects
    - Initially load the "data" folder from my DVC repository on DagsHub to be used in the students repo
    - Understand what dagshub is (keep it only scoped to DVC data storage for the moment). Students have their own linked Dagshub repository
    - Add the files to the repo using DVC and push them to their remote DVC storage on DagsHub
- Training of a model. The training itself is not the focus. It's more about how to store a model in a simple way (e.g. pickling it)
- Usage of MLFlow for experiment tracking (dagshub based)
- Usage of MLFlow for model registry
- GitHub Actions Workflow for training the model and registering it in the MLFlow model registry
- Usage of the model in a simple FastAPI app

# Resources

- When I say "my GitHub repository", then I mean https://github.com/peterrietzler/ais-dev2il-mlops-taxi-rides-v2
- We focus on data which is available through https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- The data is already pre-processed for our purposes and is available in my DagsHub DVC repository, where students can get the data from: https://dagshub.com/peter.rietzler.privat/ais-dev2il-mlops-taxi-rides-v2
- If you ask students to download files, then always download them from the `final` branch of my GitHub repository.
- Students should work on a fork of my GitHub repository

# Tooling 

The project is managed via uv
Each student will create it's own Dagshub repository linked to their GitHub repository in the course of these exercises