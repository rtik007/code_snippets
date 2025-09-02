I’d like to share the key challenges our team is currently facing with the GenAI development environment in OSDS and outline possible next steps.

Key Challenges

Fragmented Environment Setup

Individual pods and the combined pod (AICoE_GenAI_3.10) are managed separately by the technology team.

This results in inconsistent package installation and maintenance, creating overhead and variation across projects.

Limited Access to Latest Python Ecosystem

The team cannot fully leverage widely used, modern Python packages from the data science and AI community.

This slows development and prevents adoption of best-in-class tools for model building, evaluation, and deployment.

Inconsistent Package Versioning

Different teams are working with different package versions.

This leads to compatibility issues, unexpected errors, and duplicated troubleshooting efforts.

Lack of Access to Critical AI Evaluation & Adversarial Testing Packages

Key libraries are missing, including:

Evidently AI (model monitoring & drift detection)

PyRiT, ART, Promptfoo, DeepTeam, Grok, Giskard (adversarial testing, attacks, jailbreaks, probes, AI QA)

Without these, we cannot properly evaluate models for fairness, robustness, compliance, or adversarial resilience.

No Standardized Process for Package Availability

There is no clear process to request new packages in Artifactory.

Multiple artifactories exist with inconsistent and outdated package lists.

Lack of Dedicated Environments for LLM App Development

Frameworks like Streamlit, Gradio, Shiny are not supported in isolated environments.

Unutilized Licensed Tools

Although licenses were procured, the team is still struggling with Python setup and package installation, leaving these tools unusable.

Unclear Migration Roadmap for Technology Assets

No clarity exists on the migration strategy for:

Development Platforms (OSDS)

Databases (e.g., Splunk)

Enterprise Data Lake (EDL).

Security & Compliance Uncertainty

While Python environment monitoring is in place, it is unclear whether security and compliance checks are adequate for model development platforms such as OSDS or Run.AI.

Impact

Significant developer time is lost on environment setup instead of core development.

Limited adoption of modern AI practices, especially around evaluation, adversarial resilience, and compliance testing.

Difficulty maintaining consistency across teams, resulting in inefficiencies and repeated work.

Next Steps Under Consideration

Centralized, containerized environments using Run.AI and GCP, with managed local setups via Cloud PC / VDI and JFROG Artifactory.

Establishing a single AI CoE GenAI pod for package governance and version control.

Collaborating with the technology team to test evaluation and adversarial testing packages in OSDS, preparing for Run.AI and GCP adoption.

Proposal

We propose a joint discussion to explore solution options and create a roadmap that:

Ensures governance and security,

Enables access to modern tools, and

Provides flexibility for project teams to innovate effectively.
