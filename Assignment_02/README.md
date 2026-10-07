# Assignment 2: Simulated Driving Agent Navigation via Genetic Algorithm

**Student:** Chaitanya Pahilwan  
**Track:** Tata Technologies TechPulse Applied AI/ML Laboratory  
**Topic:** Evolutionary Computation, Genetic Algorithms & Simulation

---

## 1. Project Overview & Objective
This assignment implements an autonomous simulated driving agent that learns to navigate a dynamic multi-lane roadway and avoid oncoming obstacles using a **Genetic Algorithm (GA)** without gradient-based learning or neural networks.

---

## 2. Simulation Environment Specification
- **Track Layout:** 5 parallel lanes (lane indices 0 through 4).
- **Episode Duration:** 40 discrete simulation timesteps.
- **Sensory Perception:** 3-timestep lookahead distance in the agent's current lane ($2^3 = 8$ discrete sensory states).
- **Control Actions:** `[-1, 0, +1]` corresponding to `[Move Left, Stay, Move Right]`.
- **State-Action Genome:** Lookup policy mapping `(lane_position, sensor_state) -> action`. Total genes = $5 \times 8 = 40$ genes.

---

## 3. Genetic Algorithm Operators & Hyperparameters
- **Population Size:** 80 candidate agent genomes.
- **Generations:** 40 evolutionary iterations.
- **Fitness Evaluation:** Average timesteps survived across 20 deterministic training seeds (`range(100, 120)`).
- **Elitism:** Top 12 genomes preserved directly per generation.
- **Recombination:** Single-point crossover between selected parents.
- **Mutation:** Uniform random gene mutation with a 3% mutation rate.

---

## 4. Experimental Results & Generalization Findings
- **Training Progression:** Average survival time improved from **8.10 timesteps** (Generation 0) to **33.30 timesteps** (Generation 39), demonstrating steady evolutionary adaptation.
- **Hold-out Generalization:** Evaluated on 5 completely unseen obstacle seeds (`[777, 888, 999, 1234, 5678]`), achieving an average survival of **9.20 timesteps**.
- **Engineering Insight:** A compact 40-gene policy table excels at mastering encountered obstacle streams; achieving robust generalization across arbitrary obstacle topologies requires expanded lookahead horizons and lateral sensor awareness.

---

## 5. Execution Instructions
```bash
cd Assignments/Assignment_02
jupyter notebook TTL_Assignment_02.ipynb
```
