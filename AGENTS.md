
# Codex Agent Instructions

## 1. Project Overview

This repository contains a 3D Printer Predictive Maintenance System developed for a UVA engineering capstone project.

The system collects data from an Ultimaker S3 printer, an STM32 microcontroller, and three external sensors to train machine learning models that classify printer component health.

The primary models are:
- Random Forest
- XGBoost
- Hybrid-fusion 1D-CNN

Detailed project specifications are maintained in `DATA_PIPELINE.md`, `ML_MODELS.md`, and `PROJECT_ARCHITECTURE.md`. Refer to Section 2 for documentation responsibilities and usage rules.

## 2. Project Documentation

The following Markdown files contain the primary technical documentation for this project. Consult the relevant files before making implementation decisions or modifying code.

### DATA_PIPELINE.md

Defines the data collection and preprocessing pipeline, including:
- Ultimaker S3 API and external sensor data collection
- Sensor sampling frequencies and timestamp synchronization
- Raw data storage and organization
- Rolling-window segmentation
- Feature engineering and dataset construction
- Data preprocessing and quality considerations

Read this file before working on data ingestion, synchronization, preprocessing, feature engineering, or dataset generation.

### ML_MODELS.md

Defines the machine learning methodology, including:
- Random Forest and XGBoost implementations
- Hybrid-fusion 1D-CNN architecture
- Input features and model-specific data representations
- Output labels and component health classifications
- Training, validation, and testing procedures
- Hyperparameter tuning and evaluation metrics

Read this file before implementing, modifying, training, tuning, or evaluating machine learning models.

### PROJECT_ARCHITECTURE.md

Defines the overall system architecture, including:
- Project goals and system requirements
- Hardware components and their responsibilities
- Communication between the Ultimaker S3, STM32, sensors, and processing system
- Software components and their interactions
- End-to-end system workflow
- Integration between data collection, processing, and machine learning

Read this file before making architectural decisions, modifying component interfaces, or integrating system components.

### Documentation Rules

- Treat these files as the primary references for project-specific technical decisions.
- Read the relevant documentation before making changes; do not rely solely on assumptions or previous conversation context.
- Follow documented specifications unless explicitly instructed otherwise.
- If documentation conflicts with existing code, identify the discrepancy instead of silently choosing one.
- If multiple documentation files conflict, explain the inconsistency and request clarification before making consequential changes.
- Do not invent missing technical specifications.
- When an approved implementation change makes documentation outdated, update the relevant Markdown files accordingly.
- Keep documentation consistent with the actual implementation.

## 3. General Behavior

- Prioritize correctness, simplicity, maintainability, and reproducibility.
- Do not blindly agree with my suggestions. Evaluate them objectively and explain potential problems.
- Challenge decisions when there are meaningful technical concerns.
- Do not introduce unnecessary complexity or overengineering.
- Follow existing project conventions unless there is a compelling reason to change them.
- Do not make assumptions about unclear requirements, data formats, or system behavior.
- When necessary, ask clarifying questions before making consequential decisions.
- Explain important design decisions and their tradeoffs.

## 4. Code Modifications

- Inspect relevant existing code before making changes.
- Make minimal, targeted modifications.
- Do not rewrite working functionality unnecessarily.
- Preserve existing interfaces and functionality unless explicitly instructed otherwise.
- Avoid adding dependencies unless justified.
- Do not modify unrelated files.
- Do not delete existing data, models, or experiment results without explicit permission.
- Clearly summarize what changed and why.

## 5. Code Quality

- Write readable, modular Python code.
- Use descriptive names for variables, functions, and classes.
- Add type hints where practical.
- Use docstrings for nontrivial functions.
- Avoid excessive comments that merely repeat what the code does.
- Prefer straightforward implementations over unnecessarily sophisticated abstractions.
- Handle errors and edge cases appropriately.
- Avoid hardcoded paths, credentials, and machine-specific configurations.

## 6. Data Processing

- Preserve original raw data.
- Do not silently discard, interpolate, or modify measurements.
- Maintain timestamp integrity across data sources.
- Account for differences in sensor sampling frequencies.
- Make preprocessing steps explicit and reproducible.
- Avoid introducing data leakage during feature engineering, preprocessing, or dataset splitting.
- Do not assume that sensor streams are perfectly synchronized.

## 7. Machine Learning

- Prioritize generalization over achieving high training accuracy.
- Avoid data leakage between training, validation, and test datasets.
- Split datasets by independent print runs rather than randomly splitting overlapping windows.
- Fit preprocessing transformations using training data only.
- Use appropriate evaluation metrics, including F1-score, precision, recall, and confusion matrices.
- Account for class imbalance when evaluating models.
- Use reproducible random seeds where applicable.
- Do not modify experimental labels or their definitions without explicit approval.
- Distinguish between experimentally demonstrated performance and hypothetical expectations.
- Never fabricate model results or evaluation metrics.

## 8. Debugging and Testing

- Investigate the root cause of bugs rather than applying superficial fixes.
- Explain the likely cause of an issue before implementing major changes.
- Test modified functionality whenever practical.
- Add or update tests when appropriate.
- Do not claim that code works unless it has been tested.
- If testing cannot be completed, clearly explain why.
- Identify unresolved bugs, assumptions, and limitations.

## 9. Communication

- Be direct, concise, and technically precise.
- Explain complex concepts clearly without oversimplifying important details.
- For major architectural decisions, present the recommended approach and explain why.
- Identify meaningful risks, limitations, and alternatives.
- Distinguish facts from assumptions and recommendations.
- Avoid excessive praise, unnecessary reassurance, and blindly validating my ideas.

## 10. Final Response Format

After completing a coding task, summarize:
1. What was changed.
2. Why the changes were made.
3. What was tested and the results.
4. Any remaining issues or recommended next steps.

Keep summaries proportional to the complexity of the task.
