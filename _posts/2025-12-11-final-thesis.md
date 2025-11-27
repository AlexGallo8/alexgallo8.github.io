---
layout: post
title: "Final Thesis - Stochastic Processes in Cybersecurity"
---

## Thesis Topic: Beyond the random walk: Brownian Motion, stochastic processes, and the Invariance Principle (Donsker's theorem)

### Abstract

This thesis explores the application of stochastic processes, specifically random walks and Poisson processes, to model and analyze security events within cybersecurity systems. Traditional deterministic models often fall short in capturing the inherent randomness and dynamic nature of cyber threats. By leveraging the mathematical rigor of stochastic processes, this research aims to provide a more nuanced understanding of attack dynamics, anomaly detection, and predictive analytics in cybersecurity. The study will demonstrate how these models, rooted in foundational probability theory, can offer valuable insights into system resilience, threat propagation, and incident response strategies.

<!--more-->

### 1. Introduction

The landscape of cybersecurity is characterized by constant evolution, unpredictable threats, and complex interactions between attackers and defenders. Understanding and mitigating these threats requires sophisticated analytical tools that can account for inherent uncertainties. Traditional security analyses often rely on static or deterministic models, which may oversimplify the dynamic and probabilistic nature of cyber attacks and system vulnerabilities. This thesis proposes the use of stochastic processes as a powerful framework for modeling and analyzing security events, offering a more realistic and robust approach to cybersecurity.

This research will focus on two fundamental types of stochastic processes: random walks and Poisson processes. Random walks are particularly suited for modeling the progression of an attack or the state of a system over time, where each step represents a discrete event or change in status. Poisson processes, on the other hand, excel at modeling the occurrence of events over continuous time, such as the arrival of malicious traffic or the frequency of security incidents. By integrating these models, this thesis aims to provide a comprehensive framework for understanding, predicting, and responding to cyber threats.

The primary objectives of this thesis are:
*   To establish a theoretical foundation for applying random walks and Poisson processes to various cybersecurity scenarios.
*   To develop and implement computational models for simulating attack dynamics and event occurrences.
*   To demonstrate the utility of these models in anomaly detection, threat prediction, and risk assessment.
*   To explore the connections between discrete and continuous stochastic models, potentially leveraging advanced concepts like Donsker's theorem for broader applicability.

### 2. Background and Theoretical Foundations

This section will lay the groundwork by reviewing essential concepts from probability theory and the specific stochastic processes central to this research.

#### 2.1 Probability Theory and Measure Theory

The rigorous foundation for stochastic processes lies in axiomatic probability theory, as established by Kolmogorov. This involves defining a probability space $(\Omega, \mathcal{F}, P)$, where $\Omega$ is the sample space, $\mathcal{F}$ is a $\sigma$-algebra of events, and $P$ is a probability measure. The understanding of $\sigma$-algebras and measurable functions is crucial for defining random variables and, subsequently, stochastic processes.

*(Refer to Homework 9: "Probability Theory and Measure Theory" for detailed explanations of classical, frequentist, Bayesian, and geometric interpretations of probability, Kolmogorov's axioms, $\sigma$-algebras, and the connection between measure theory and probability theory. This homework provides the mathematical rigor for the theoretical underpinnings.)*

#### 2.2 Bernoulli Processes and the Law of Large Numbers

A Bernoulli process is a sequence of independent Bernoulli trials, where each trial has two possible outcomes (e.g., success/failure, secure/breached). The Law of Large Numbers (LLN) states that as the number of trials increases, the observed frequency of an event converges to its true probability. This concept is fundamental to understanding the long-term behavior of systems where events occur probabilistically.

*(Refer to Homework 4: "The Law of Large Numbers and its applications in statistical inference and data analysis" for the simulation and analysis of Bernoulli trials and the demonstration of LLN. This forms the discrete basis for understanding subsequent stochastic models.)*

#### 2.3 Random Walks

A random walk is a mathematical formalization of a path that consists of a succession of random steps. In its simplest form, it involves movement on a grid where, at each step, the direction of movement is chosen randomly. Random walks can be symmetric or asymmetric, depending on the probabilities of moving in different directions. They are powerful tools for modeling cumulative effects of random events.

*(Refer to Homework 7: "Beyond the random walk: Brownian Motion, stochastic processes, and the Invariance Principle (Donsker's theorem)" and Homework 8: "Comparing Bernoulli and Random Walk Processes" for the simulation of random walks, their properties, and the transformation from Bernoulli processes. The application of random walks to model "net security score" in network security from Homework 7 will be a direct starting point.)*

#### 2.4 Poisson Processes

A Poisson process is a stochastic process that counts the number of events occurring in a given time interval. It is characterized by events occurring independently at a constant average rate ($\lambda$). Key properties include independent and stationary increments, and the number of events in any interval follows a Poisson distribution. The time between consecutive events (interarrival times) follows an exponential distribution.

*(Refer to Homework 10: "Poisson Process Simulation" for the detailed methodology of simulating Poisson processes, their theoretical background, properties, and connection to random walks. This homework provides the practical and theoretical basis for modeling event occurrences in continuous time.)*

### 3. Modeling Cybersecurity Events with Random Walks

This section will develop and apply random walk models to represent various cybersecurity scenarios, focusing on the progression of attacks and the state of system security.

#### 3.1 Attack Progression as a Random Walk

We can model the state of a system or the progress of an attacker as a random walk. Each step in the walk represents a discrete action or event. For instance, a positive step could signify a successful defensive action or a failed attacker attempt, while a negative step could represent a successful exploit or a system compromise.

**Model Formulation:**
Let $S_n$ be the security score of a system after $n$ events.
$S_n = S_{n-1} + X_n$, where $X_n \in \\{-1, +1\\}$ represents the outcome of the $n$-th event.
$P(X_n = +1) = p$ (e.g., successful defense, attacker fails)
$P(X_n = -1) = q = 1-p$ (e.g., successful attack, system compromised)

**Cybersecurity Applications:**
*   **Lateral Movement Modeling:** An attacker's movement within a network can be modeled as a random walk, where nodes are states and edges are possible transitions.
*   **System Resilience Assessment:** The random walk can represent the system's security posture, with thresholds for "compromised" and "recovered" states.
*   **Phishing Campaign Effectiveness:** Each interaction with a user can be a step, leading to either a successful phishing attempt or detection.

**Key Analysis:**
*   **First Passage Time:** Calculating the expected time until the system reaches a critical state (e.g., full compromise or successful recovery).
*   **Probability of Ruin/Success:** Determining the probability that an attacker succeeds before being detected, or vice versa.
*   **Impact of Asymmetry:** Analyzing how varying probabilities $p$ and $q$ (e.g., due to new defenses or attack techniques) affect the system's security trajectory.

**Computational Implementation (Placeholder):**
*(This section will include Python code for simulating random walks, similar to `simulate_random_walk` from Homework 7, but adapted for specific cybersecurity scenarios. It will also include code for visualizing trajectories and analyzing key metrics like first passage time and probability distributions.)*

<details>
   <summary>View first_piece.py</summary>
   {% highlight python linenos %}
    import numpy as np
    import matplotlib.pyplot as plt

    def simulate_attack_random_walk(initial_state, p_success, n_steps, n_simulations):
        steps = np.random.choice([1, -1], size=(n_simulations, n_steps), p=[p_success, 1-p_success])
        security_scores = initial_state + np.cumsum(steps, axis=1)
        security_scores = np.hstack([initial_state * np.ones((n_simulations, 1)), security_scores])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        for i in range(min(n_simulations, 50)):
            ax1.plot(security_scores[i])
        ax1.axhline(y=-20, color='red', linestyle='--', label='Compromise Threshold')
        ax1.set_xlabel('Event Count')
        ax1.set_ylabel('System Security Score')
        ax1.legend()
        
        ax2.hist(security_scores[:, -1], bins=20)
        ax2.set_xlabel('Final Security Score')
        ax2.set_ylabel('Number of Simulations')
        
        plt.tight_layout()
        plt.savefig('2025-12-25-random-walk-simulation.png')
    {% endhighlight %}
</details>

![Random Walk Simulation of Attack Progression]({{ site.baseurl }}/assets/images/final_thesis/random-walk-simulation.png)
*Figure 1: Simulated trajectories of system security scores over 1000 events (500 simulations) with a 60% defense success probability. The red line indicates the critical compromise threshold (-20), and the histogram shows the distribution of final security scores (aligned with Homework 7's random walk analysis framework).*

### 4. Modeling Cybersecurity Events with Poisson Processes

This section will apply Poisson processes to model the occurrence of discrete security events over continuous time, enabling anomaly detection and predictive analytics.

#### 4.1 Event Arrival Modeling

Many cybersecurity events, such as login attempts, firewall alerts, intrusion attempts, or malware detections, can be modeled as events occurring randomly over time. If these events occur independently at a constant average rate, a Poisson process is an appropriate model.

**Model Formulation:**
Let $N(t)$ be the number of security events observed up to time $t$.
$N(t) \\sim \\text{Poisson}(\\lambda t)$, where $\\lambda$ is the average rate of events per unit time.
The probability of observing $k$ events in time $t$ is $P(N(t) = k) = \\frac{e^{-\\lambda t} (\\lambda t)^k}{k!}$.

**Cybersecurity Applications:**
*   **Anomaly Detection:** Deviations from the expected Poisson rate can indicate malicious activity. For example, a sudden increase in failed login attempts (higher $\\lambda$) might signal a brute-force attack.
*   **Security Incident Frequency:** Modeling the rate of security incidents to understand trends and allocate resources effectively.
*   **Log Analysis:** Analyzing the arrival rate of specific log entries to identify unusual patterns.

**Key Analysis:**
*   **Rate Estimation:** Estimating the parameter $\\lambda$ from observed security event data.
*   **Statistical Hypothesis Testing:** Using statistical tests (e.g., chi-squared goodness-of-fit) to determine if observed event counts conform to a Poisson distribution.
*   **Change Point Detection:** Identifying shifts in the rate parameter $\\lambda$, which could indicate the onset of an attack or a change in system behavior.

**Computational Implementation (Placeholder):**
*(This section will include Python code for simulating Poisson processes, similar to `simulate_poisson_process` from Homework 10, and for analyzing real or synthetic security event data. It will cover rate estimation, visualization of event arrivals, and basic anomaly detection techniques based on Poisson distributions.)*

<details>
   <summary>View second_piece.py</summary>
   {% highlight python linenos %}
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import poisson

    def simulate_security_poisson_process(lambda_rate, T, n_intervals):
        interarrival_times = np.random.exponential(scale=1/lambda_rate, size=int(lambda_rate*T*2))
        event_times = np.cumsum(interarrival_times)
        event_times = event_times[event_times <= T]
        
        interval_bins = np.linspace(0, T, n_intervals+1)
        event_counts = np.histogram(event_times, bins=interval_bins)[0]
        estimated_lambda = np.mean(event_counts)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.stem(event_times, np.ones_like(event_times), basefmt='b-')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Security Event Occurrence')
        
        x = np.arange(0, max(event_counts)+2)
        ax2.hist(event_counts, bins=x-0.5, density=True, alpha=0.6)
        ax2.plot(x, poisson.pmf(x, estimated_lambda), 'b-', label=f'Estimated Poisson (λ={estimated_lambda:.2f})')
        ax2.set_xlabel('Event Count per Interval')
        ax2.set_ylabel('Probability Density')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('2025-12-25-poisson-process-simulation.png')
        
        return event_times, event_counts, estimated_lambda
    {% endhighlight %}
</details>

![Poisson Process Simulation of Security Event Arrivals]({{ site.baseurl }}/assets/images/final_thesis/poisson-process-simulation.png)
*Figure 2: Simulated security event arrivals over 24 hours (λ=5 events/hour) and interval event count distribution (aligned with Homework 10's Poisson process methodology). The blue line shows the estimated Poisson distribution for anomaly detection benchmarking.*

#### 4.2 Interarrival Times and Exponential Distribution

A key property of a Poisson process is that the time between consecutive events (interarrival times) follows an exponential distribution with rate $\\lambda$. This allows for analysis of the time gaps between security events.

**Cybersecurity Applications:**
*   **Timing Analysis of Attacks:** Analyzing the time intervals between stages of a multi-stage attack.
*   **Resource Exhaustion Attacks:** Understanding if the interarrival times of requests in a denial-of-service attack follow an exponential distribution, which can inform defense strategies.

### 5. Integration and Advanced Topics

This section will explore how random walk and Poisson process models can be integrated and extended, potentially delving into more advanced stochastic concepts.

#### 5.1 Combining Models for Comprehensive Threat Analysis

A complete cybersecurity scenario often involves both the progression of an attack (random walk) and the occurrence of various events over time (Poisson process). For example, a random walk could model an attacker's lateral movement, while Poisson processes could model the generation of alerts at different stages of that movement.

**Research Directions:**
*   **Hybrid Models:** Developing models that combine discrete-step random walks with continuous-time Poisson event generation.
*   **Conditional Probabilities:** Analyzing the probability of a system state (from a random walk) given a certain number of events (from a Poisson process), and vice versa.

#### 5.2 Brownian Motion and Donsker's Theorem

Donsker's theorem, also known as the Invariance Principle, states that under certain conditions, a properly scaled and centered random walk converges in distribution to Brownian motion (a continuous-time stochastic process). This connection allows for the application of powerful analytical tools from continuous mathematics to discrete random walk problems.

*(Refer to Homework 7: "Beyond the random walk: Brownian Motion, stochastic processes, and the Invariance Principle (Donsker's theorem)" for the initial introduction to this concept.)*

**Implications for Cybersecurity:**
*   **Continuous Approximation:** Using Brownian motion to approximate the behavior of high-frequency security events or long-duration attack progressions, simplifying analysis.
*   **Advanced Analytics:** Applying techniques developed for Brownian motion (e.g., stochastic differential equations) to cybersecurity problems.

### 6. Case Studies and Empirical Validation

This section will apply the developed stochastic models to real or synthetic cybersecurity datasets to demonstrate their practical utility and validate their effectiveness.

**Potential Case Studies:**
*   **Network Intrusion Detection:** Analyzing network traffic logs (e.g., NetFlow data) to identify anomalous patterns using Poisson process models for packet arrival rates.
*   **Vulnerability Exploitation Chains:** Modeling the sequence of exploits in a multi-stage attack using random walks, and predicting the likelihood of successful compromise.
*   **Security Information and Event Management (SIEM) Data Analysis:** Applying both random walk and Poisson process models to SIEM data to detect complex attack patterns and predict future incidents.

**Methodology:**
*   Data collection and preprocessing.
*   Parameter estimation for random walk and Poisson models.
*   Simulation and comparison with observed data.
*   Evaluation of model performance in terms of accuracy, precision, recall, and F1-score for detection and prediction tasks.

### 7. Conclusion

This thesis has demonstrated the significant potential of stochastic processes, particularly random walks and Poisson processes, in enhancing the modeling and analysis of security events in cybersecurity. By moving beyond deterministic approaches, these probabilistic models offer a more realistic representation of the dynamic and uncertain nature of cyber threats. The research has highlighted how these models can be applied to understand attack progression, detect anomalies, and inform predictive analytics, thereby contributing to more robust and proactive cybersecurity strategies.

Future work could involve exploring more complex stochastic models, such as Markov chains or semi-Markov processes, to capture state-dependent transitions and varying event rates. Further research could also focus on integrating machine learning techniques with stochastic models to develop adaptive and intelligent cybersecurity systems.