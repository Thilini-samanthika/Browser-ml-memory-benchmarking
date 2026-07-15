# Browser ML Memory Benchmarking

## Project Title

**An Empirical Performance Evaluation of Memory Heap Stability and Garbage Collection Overhead in Browser-Based Machine Learning: TensorFlow.js vs ONNX Runtime Web Under High-Spec Workstation Workloads**

---

## Project Overview

This research project investigates the memory management behavior and garbage collection overhead of browser-based machine learning frameworks. The study compares TensorFlow.js and ONNX Runtime Web by executing continuous machine learning inference workloads inside modern web browsers and measuring runtime performance under high-spec workstation environments.

---

## Objective

The main objective of this research is to evaluate and compare the memory heap stability, JavaScript garbage collection overhead, and runtime efficiency of TensorFlow.js and ONNX Runtime Web during continuous browser-based machine learning inference.

---

## Research Question

**Does ONNX Runtime Web provide better memory heap stability and lower garbage collection overhead than TensorFlow.js during continuous browser-based machine learning inference on high-performance hardware?**

---

## Technologies

### Frontend
- React
- TypeScript
- Vite

### Machine Learning
- TensorFlow.js
- ONNX Runtime Web

### Performance Profiling
- Chrome DevTools
- Firefox Developer Tools
- Performance API

### Development
- Node.js
- npm
- Git
- GitHub

---

## Proposed Folder Structure

```
browser-ml-memory-benchmarking/

├── docs/
│   ├── methodology/
│   └── diagrams/
│
├── datasets/
│
├── models/
│
├── results/
│
├── scripts/
│
├── src/
│   ├── tensorflow/
│   ├── onnx/
│   ├── profiler/
│   └── utils/
│
├── public/
│
├── README.md
├── package.json
└── requirements.txt
```

---

## Research Methodology

The project follows an experimental benchmarking methodology.

1. Load an identical machine learning model using TensorFlow.js and ONNX Runtime Web.
2. Execute continuous inference workloads.
3. Monitor JavaScript heap memory usage.
4. Measure garbage collection events.
5. Record execution time and memory statistics.
6. Compare the performance of both frameworks using statistical analysis.

---

## Expected Performance Metrics

- JavaScript Heap Usage
- Heap Growth Rate
- Garbage Collection Pause Time
- Execution Time
- Memory Allocation
- Peak Memory Usage

---

## Repository Status

Current Progress:

- [x] Research proposal completed
- [x] Public GitHub repository created
- [x] Initial project structure prepared
- [ ] React project setup
- [ ] TensorFlow.js integration
- [ ] ONNX Runtime Web integration
- [ ] Memory profiler implementation
- [ ] Benchmark automation
- [ ] Dataset generation
- [ ] Statistical analysis
- [ ] Final research paper

---

## Team Members

**Member 1**
- M. Thilini Samanthika

**Member 2**
- J.A.B.B. Jayakody

---

## License

This repository is created for academic research purposes as part of the **IT41043 – Intelligent Systems** module at Horizon Campus.