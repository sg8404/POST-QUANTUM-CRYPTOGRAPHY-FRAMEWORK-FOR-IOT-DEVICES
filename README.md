# POST QUANTUM CRYPTOGRAPHY FRAMEWORK FOR IOT DEVICES


This repository presents a comprehensive **Post-Quantum Cryptographic Framework** designed to secure Internet of Things (IoT) networks against quantum computing threats. It integrates **McEliece cryptography**, **AES-256 encryption**, and **physical-layer authentication techniques** like RF Fingerprinting and Physical Unclonable Functions (PUFs) to ensure robust, scalable, and energy-efficient IoT security.

---

## Features

- **Hybrid Cryptographic Framework**: Combines McEliece (quantum-resistant key exchange) with AES-256 (fast and secure encryption).
- **Physical-Layer Security**:
  - RF Fingerprinting for device-specific authentication.
  - PUF-based authentication to counter hardware cloning attacks.
- **Energy Efficiency**: Optimized to meet low-power IoT constraints (≤ 10 mJ per operation).
- **Scalable Design**: Tested for networks with up to 10,000 nodes, ensuring adaptability for large-scale IoT deployments.
- **Robust Simulation Environment**: Includes tools for performance analysis across throughput, latency, energy usage, and authentication accuracy.

---

## Performance Highlights

- **Throughput**: Maintains data rates up to 1,000 nodes.
- **Latency**: Low latency of ≤ 20 ms, suitable for real-time IoT applications.
- **Authentication Accuracy**:
  - RF Fingerprinting: 96% classification accuracy.
  - PUF-based authentication: 98% accuracy.
  - False Acceptance Rate (FAR): < 0.05.
- **Energy Usage**: ≤ 10 mJ per operation, enabling use in resource-constrained IoT systems.

---

## Technologies Used

- **Programming Languages**: Python, MATLAB
- **Simulation Tools**: NS3, Mininet, MATLAB Simulink
- **Frameworks**: PyCrypto for cryptography, RF Fingerprinting models, PUF-based logic, Machine Learning Algorithms
- **Cryptographic Algorithms**:
  - McEliece for quantum-resistant key exchange
  - AES-256 for real-time encryption

