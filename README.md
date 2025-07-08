# Federated Learning for Collaborative Cyber-Attack Detection in Internet of Vehicles (IoV) Network

## Overview

This repository contains a comprehensive implementation and analysis of federated learning algorithms for collaborative cyber-attack detection in Internet of Vehicles (IoV) networks using the CICIoV2024 dataset.

## Problem Statement

Internet of Vehicles (IoV) networks face significant cybersecurity challenges due to their distributed nature and vulnerability to various attack types including Denial-of-Service (DoS) attacks and spoofing attacks targeting critical vehicle components (gas, RPM, speed, steering wheel). Traditional centralized intrusion detection systems are inadequate for IoV environments due to privacy concerns, bandwidth limitations, and the need for real-time collaborative threat detection across distributed vehicle fleets.

## Proposed Solution

We propose a novel federated learning (FL) framework for attack detection on the CAN bus with a Non-IID (non-identically and independently distributed) client setup based on attack specialization. This approach simulates realistic scenarios where different vehicles in a fleet might be targeted by different types of attacks, enabling collaborative learning without sharing sensitive raw data.

### Key Features:
- **Non-IID Client Configuration**: Specialized clients focusing on different attack types
- **Multi-Algorithm Support**: Implementation of four state-of-the-art FL algorithms
- **Privacy-Preserving**: No raw data sharing between clients
- **Real-World Simulation**: Reflects actual IoV deployment scenarios

## Methodology

### Dataset
- **Source**: CICIoV2024 dataset from the Canadian Institute for Cybersecurity
- **Attack Types**: Benign traffic, DoS attacks, and four spoofing attack variants
- **Data Format**: Decimal representation of CAN bus messages
- **Features**: CAN ID + 8 data bytes (9 features total)

### Client Specialization Strategy
1. **Client 1 (Benign Focus)**: The benign data of IoV traffic.
2. **Client 2 (DoS Specialist)**: Benign data portion + all DoS attack data
3. **Client 3 (Gas Spoofing Specialist)**: Benign data portion + all gas spoofing data
4. **Client 4 (RPM Spoofing Specialist)**: Benign data portion + all RPM spoofing data
5. **Client 5 (Speed Spoofing Specialist)**: Benign data portion + all speed spoofing data
6. **Client 6 (Steering Wheel Spoofing Specialist)**: Benign data portion + all steering wheel spoofing data

### Federated Learning Algorithms Evaluated
1. **FedAvg**: Standard federated averaging
2. **FedProx**: Federated learning with proximal regularization
3. **SCAFFOLD**: Variance reduction through control variates
4. **FedNova**: Normalized averaging for heterogeneous local updates

### Model Architecture
- **Type**: 4-layer Deep Neural Network (DNN)
- **Architecture**: 64 → 32 → 16 → 1 neurons
- **Regularization**: Batch normalization, dropout, L2 regularization
- **Activation**: ReLU (hidden layers), Sigmoid (output)
- **Optimization**: Adam optimizer with adaptive learning rate

### Experimental Setup
- **Rounds**: 8 federated learning rounds
- **Local Epochs**: 3-4 epochs per round
- **Evaluation Metrics**: F1-score (primary), Accuracy, Precision, Recall
- **Data Split**: 70% training, 30% testing with proper separation
- **Reproducibility**: Fixed random seeds for deterministic results

## Results

### Performance Rankings (F1-Score - Primary Metric)

Based on comprehensive experimental analysis with proper data separation and statistical validation:

| Rank | Algorithm | F1-Score | Accuracy | Precision | Recall |
|------|-----------|----------|----------|-----------|--------|
| 1    | FedAvg    | 0.8330   | 0.9992   | 0.8328    | 0.8333 |
| 2    | FedProx   | 0.8157   | 0.9836   | 0.8328    | 0.8019 |
| 3    | FedNova   | 0.8157   | 0.9836   | 0.8328    | 0.8019 |
| 4    | SCAFFOLD  | 0.8042   | 0.9746   | 0.8328    | 0.7842 |

### Key Findings

1. **FedAvg Superior Performance**: Achieved the highest F1-score (0.8330) and exceptional accuracy (0.9992)
2. **Effective Non-IID Learning**: All algorithms successfully learned from specialized, heterogeneous data
3. **Balanced Metrics**: High precision (0.8328) reduces false alarms while maintaining good recall for attack detection
4. **Practical Performance**: All algorithms achieved F1-scores above 0.80, suitable for real-world deployment
5. **Close Competition**: FedProx and FedNova tied for second place with identical performance metrics

### Cybersecurity Implications
- **F1-Score Optimization**: Balances precision and recall critical for IoV attack detection
- **False Alarm Reduction**: High precision minimizes disruption to vehicle operations
- **Attack Coverage**: High recall ensures comprehensive threat detection
- **Real-Time Capability**: Fast convergence enables rapid deployment in IoV networks

## Technical Implementation

### Dependencies
```python
tensorflow>=2.13.0
flwr[simulation]>=1.5.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Key Components
- **Data Preprocessing**: Standardization, class balancing, proper train/test separation
- **FL Client Implementation**: Algorithm-specific client classes with control variates
- **Aggregation Methods**: Weighted averaging with algorithm-specific enhancements
- **Evaluation Framework**: Comprehensive metrics calculation and statistical analysis

## Usage

1. **Data Preparation**: Place CICIoV2024 decimal CSV files in the `decimal/` directory
2. **Environment Setup**: Install required dependencies
3. **Experiment Execution**: Run the Jupyter notebook cells sequentially
4. **Results Analysis**: Review comprehensive performance metrics and visualizations

## Research Contributions

1. **Comprehensive FL Algorithm Comparison**: First systematic evaluation of 4 FL algorithms for IoV cybersecurity
2. **Non-IID Specialization Framework**: Novel client configuration reflecting real IoV attack patterns
3. **Rigorous Experimental Methodology**: Proper data separation preventing leakage with statistical validation
4. **Practical IoV Insights**: Actionable recommendations for federated learning deployment in vehicle networks
5. **Open Source Implementation**: Complete reproducible codebase for research community

## Future Work

- Extension to additional attack types and IoV protocols
- Integration with real-time vehicle communication systems
- Scalability analysis for large-scale IoV deployments
- Privacy-preserving techniques enhancement
- Edge computing optimization for resource-constrained vehicles

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Canadian Institute for Cybersecurity for the [CICIoV2024](https://www.unb.ca/cic/datasets/iov-dataset-2024.html) dataset
- Flower (flwr) federated learning framework
- TensorFlow and scikit-learn communities

---

**Keywords**: Federated Learning, Internet of Vehicles, Cybersecurity, Intrusion Detection, CAN Bus, Non-IID Data, SCAFFOLD, FedProx, Attack Detection
