# UdaciMed: Efficient Medical Diagnostics with Hardware-Aware AI

## Background scenario

You are a Machine Learning Engineer at **UdaciMed**, a healthcare technology startup developing AI-powered diagnostic tools. Your team is preparing to deploy a new chest X-ray pneumonia detection model across diverse infrastructure, including cloud services, hospital workstations, and portable clinic devices.

To ensure only the most efficient models make it into the production pipeline, UdaciMed has a strict internal policy: performance is a feature. Before any model can be deployed, it must meet a strict performance service level agreement (SLA) with the universally compatible ONNX format on the standardized development machine _(as described in the [Project Instructions](#project-instructions) below)_.

**The challenge:** The current ResNet-18 model meets clinical accuracy standards but requires significant optimization to satisfy the strict performance demands of real-world medical environments.

**Your mission:** Optimize the model through hardware-aware architectural modifications and deployment acceleration to achieve:
- **<100MB memory footprint** for multi-tenant deployment
- **>2,000 samples/sec throughput** for high-volume screening
- **>98% sensitivity** for clinical safety (non-negotiable)
- **<3ms latency** for real-time diagnosis

## Project overview

In this project, you will develop a complete **hardware-aware optimization pipeline** for pneumonia detection using the [PneumoniaMNIST](https://medmnist.com/) dataset. Starting with a [ResNet-18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) baseline, you will analyze performance bottlenecks, apply architectural optimizations, implement hardware acceleration using [ONNX Runtime](https://onnxruntime.ai/), and analyze expected performance across multiple deployment scenarios. 

**Key pipeline stages:**
1. **Baseline Analysis** - Profile performance bottlenecks and identify optimization opportunities
2. **Architecture Optimization** - Implement 3+ optimization techniques within a modular optimization framework
3. **Hardware Acceleration & Deployment** - Apply hardware optimizations and define next steps for model deployment on production targets

Each pipeline stage corresponds to a notebook in the `notebooks/` folder. Complete the relevant _TODOs_ in each notebook to deploy a production-ready medical imaging model that demonstrates significant performance improvements while maintaining diagnostic accuracy.

### Learning objectives

By completing this project, you will:
- **Analyze performance trade-offs** between optimization strategies for specific hardware targets
- **Benchmark and profile** model performance across different deployment scenarios
- **Implement hardware-aware architectural optimizations** on deep learning models
- **Apply hardware acceleration** to optimize model inference
- **Deploy optimized models** using ONNX execution providers
- **Evaluate optimization strategies** for diverse deployment targets through critical analysis

## Project instructions

The `starter/` folder is the home for your project.

> **A note on technical requirements**
> 
> This project has been developed and tested on an NVIDIA T4 instance with 16GB VRAM, running Ubuntu 22.04 with CUDA 12.4, cuDNN 8.9.2, NVIDIA driver 550, Python 3.10, and Docker pre-installed. Baseline performance metrics and environment setup have been calibrated for this configuration and may require adjustments for different hardware setups.

Follow these steps to complete the project:

1. [Pre-requisite: Set up the project](#1-pre-requisite-set-up-the-project)
2. [Understand the project folder structure](#2-understand-the-project-folder-structure)
3. [Get started with the project](#3-get-started-with-the-project)

### 1. Pre-requisite: Set up the project

From the project home folder:

1. **(_Optional_) Create a virtual environment** (Python 3.10 recommended)
   ```bash
   python -m venv udacimed_env
   source udacimed_env/bin/activate  # On Windows: udacimed_env\Scripts\activate
   ```

2. **Install project scripts as editable local package:**
   ```bash
   pip install -e .
   ```

### 2. Understand the project folder structure

Below is a breakdown of the `starter/` folder.

```
.     
├── requirements.txt             
├── setup.py                
├── deployment/                                 # Production deployment configurations and artifacts
├── notebooks/                                  # Jupyter notebooks containing the main project workflow
│   ├── 01_baseline_analysis.ipynb        
│   ├── 02_architecture_optimization.ipynb 
│   └── 03_deployment_acceleration.ipynb  
├── utils/                                      # Utility modules supporting the optimization pipeline
│   ├── __init__.py                   
│   ├── architecture_optimization.py     
│   ├── data_loader.py                   
│   ├── evaluation.py                   
│   ├── model.py                        
│   ├── profiling.py                  
│   └── visualization.py                
└── results/                                    # Generated models, metrics, and benchmark results
```

### 3. Get started with the project

Your task is to optimize a pneumonia detection model for efficient, clinically-safe, production-ready deployment.

There are **three notebooks** to complete sequentially:

1. **[`notebooks/01_baseline_analysis.ipynb`](starter/notebooks/01_baseline_analysis.ipynb)**
   - Establish baseline model performance and identify optimization opportunities
   - Profile memory usage, computational complexity, and inference timing
   - Analyze architectural bottlenecks and deployment constraints

2. **[`notebooks/02_architecture_optimization.ipynb`](starter/notebooks/02_architecture_optimization.ipynb)**
   - Implement and evaluate architectural optimization techniques
   - Train the optimized model with preserved clinical performance
   - Validate optimization impact on deployment targets

3. **[`notebooks/03_deployment_acceleration.ipynb`](starter/notebooks/03_deployment_acceleration.ipynb)**
   - Convert models for production deployment with general hardware acceleration (ONNX format)
   - Benchmark performance against deployment targets
   - Provide insights on optimization strategies for GPU, CPU and edge/mobile

In each notebook, you will find **TODOs** for both implementation and analysis tasks. Note that `notebooks/02_architecture_optimization.ipynb` includes TODOs that require implementing functions in `utils/architecture_optimization.py`.

## Project submission

Your submission should include:

- **Completed notebooks** with all TODOs implemented and analysis questions thoroughly answered
- **Optimized model weights** saved in the `results/` directory with documented performance improvements
- **Performance benchmarks** demonstrating measurable progress toward deployment targets compared to baseline
- **Deployment configuration** with complete ONNX setup, model repository structure, and end-to-end testing results

### Evaluation criteria

Your project will be evaluated based on:

- **Technical implementation (25%)** - Quality and effectiveness of hardware-aware optimization techniques
- **Performance achievement (25%)** - Extent to which your optimized model meets UdaciMed's deployment requirements
- **Analysis quality (25%)** - Depth of performance analysis, insights, and strategic optimization decisions
- **Deployment readiness (25%)** - Completeness and robustness of final performance analysis and deployment recommendation

**Success indicators:**
- Achievement of ≥3 out of 4 optimization targets (memory, throughput, latency, sensitivity)
- Clear demonstration of optimization technique effectiveness through before/after comparisons
- Production-ready deployment configuration with documented testing procedures
- Thoughtful analysis of optimization trade-offs and deployment strategy recommendations

---

## Resources

### Technical Documentation
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [ONNX Model Optimization](https://onnxruntime.ai/docs/performance/)
- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [NVIDIA Triton Inference Server Documentation](https://github.com/triton-inference-server/server)
- [Intel OpenVino Developer Tools](https://www.intel.com/content/www/us/en/developer/tools/overview.html)
- [ExecuTorch Documentation](https://docs.pytorch.org/executorch/stable/index.html)
- [CoreML Documentation](https://developer.apple.com/documentation/coreml)
- [LiteRT Overview](https://ai.google.dev/edge/litert)

### Medical AI Context
- [MedMNIST Documentation](https://medmnist.com/)
- [ResNet Architecture Guide](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
- [Medical AI Deployment Best Practices](https://arxiv.org/abs/2109.09824)

## License
[License](../LICENSE.md)

---

**Ready to optimize medical AI for real-world deployment? Start with `notebook/01_baseline_analysis.ipynb` and begin your journey to production-ready healthcare AI!**