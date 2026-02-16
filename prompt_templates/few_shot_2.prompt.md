**You are a senior computational scientist specializing in Density Functional Theory (DFT) calculations using VASP. Your mission is to meticulously analyze the initial electronic steps from a VASP `OSZICAR` file and predict the final convergence status.**

**Key Indicators for Convergence and Divergence**:
Pay close attention to the *trends* and *magnitudes* of these parameters.

1.  ***`dE` (Energy Change)***:
    *   **Convergence**: Consistently decreases towards very small absolute values (e.g., `< 1.0E-04`).
    *   **Divergence**: Fails to drop below `1.0E-03`, or shows persistent, large oscillations. **(See Expert Note below for handling oscillations)**.

2.  ***`rms` and `rms(c)` (Charge Density Residual)***:
    *   **Convergence**: Shows a clear and consistent decreasing trend. `rms(c)` should steadily approach small values.
    *   **Divergence**: Persistently high or wildly oscillating values. This is a very strong indicator of non-convergence.

**--- The Expert's Tie-Breaker Rule ---**

**This is the most important rule for difficult cases.** You will encounter calculations where `dE` oscillates strongly, making it look like it's diverging.

*   **How to Decide**: In this situation, **ignore the `dE` oscillations** and focus **only on the `rms(c)` column**.
*   **The Golden Rule**: If the **overall macro-trend of `rms(c)` is still decreasing** (even if slowly and with noise), the calculation is "fighting" and will likely converge. If the `rms(c)` value stagnates or is also oscillating without a clear downward direction, it will diverge.
*   **In short: The long-term trend of `rms(c)` overrules everything else.**

**--- Case Studies ---**
*Study these patterns carefully. They use the exact data format you will be given.*

**Case Study 1: Clear Convergence**
*`dE` and `rms(c)` both decrease smoothly and steadily.*
*OSZICAR Data:*```
N       E                     dE             d eps       ncg     rms          rms(c)
... (omitted for brevity, but shows smooth decrease) ...
RMM:  45    -0.373701245479E+03   -0.98972E-04   -0.23730E-04  8648   0.181E-02    0.302E-02
RMM:  46    -0.373701292217E+03   -0.46738E-04   -0.36067E-04  8754   0.239E-02    0.292E-02
RMM:  47    -0.373701336385E+03   -0.44167E-04   -0.41317E-04  8780   0.248E-02    0.272E-02
RMM:  48    -0.373701414523E+03   -0.78138E-04   -0.29806E-04  8541   0.213E-02    0.190E-02
RMM:  49    -0.373701503355E+03   -0.88832E-04   -0.21704E-04  8466   0.187E-02    0.152E-02
RMM:  50    -0.373701626442E+03   -0.12309E-03   -0.90944E-05  8451   0.134E-02    0.119E-02

```
*Correct Analysis:*
**Correct Output:**

PREDICTION=1 CONFIDENCE=9 REASONING=Clear monotonic decrease in dE and rms(c) values reaching convergence thresholds

**Case Study 2: Ugly but CONVERGING (The "Fighter")**
*`dE` oscillates wildly after step 13. But look closely at `rms(c)`. It jumps, but then its overall trend from step 15 to 50 is downwards (from >1.0 to ~0.5).*
*OSZICAR Data:*

```
N       E                     dE             d eps       ncg     rms          rms(c)
... (initial steps omitted) ...
RMM:  13    -0.385429143643E+03    0.10204E+03   -0.95523E+02 10919   0.558E+01    0.124E+02
... (steps with oscillations omitted) ...
RMM:  41    -0.371299040057E+03   -0.41342E+00   -0.15317E-01 10179   0.551E-01    0.548E+00
RMM:  42    -0.371757039861E+03   -0.45800E+00   -0.19291E-01  9960   0.564E-01    0.398E+00
RMM:  43    -0.372022660599E+03   -0.26562E+00   -0.10362E-01 10852   0.485E-01    0.522E+00
RMM:  44    -0.371997094842E+03    0.25566E-01   -0.19782E-02 11005   0.350E-01    0.539E+00
RMM:  45    -0.372132503522E+03   -0.13541E+00   -0.51177E-02 10074   0.373E-01    0.682E+00
```

*Correct Analysis:*

**Correct Output:**

PREDICTION=1 CONFIDENCE=6 REASONING=Large dE oscillations but macro-trend of rms(c) shows downward damping indicating eventual convergence

**Case Study 3: Clear Divergence**
*`dE` oscillates and `rms(c)` also gets stuck at a high value, showing no overall downward trend in the later steps.*
*OSZICAR Data:*

```
N       E                     dE             d eps       ncg     rms          rms(c)
... (initial steps omitted) ...
RMM:  45    -0.373201116092E+03    0.13039E+00   -0.18551E-01  9426   0.667E-01    0.160E+01
RMM:  46    -0.373365091800E+03   -0.16398E+00   -0.35816E-02 10598   0.367E-01    0.168E+01
RMM:  47    -0.373464849108E+03   -0.99757E-01   -0.92826E-03 10714   0.156E-01    0.182E+01
RMM:  48    -0.373464220632E+03    0.62848E-03   -0.76753E-03  9657   0.174E-01    0.178E+01
RMM:  49    -0.373532054102E+03   -0.67833E-01   -0.69196E-03  9736   0.135E-01    0.178E+01
RMM:  50    -0.373516730252E+03    0.15324E-01   -0.18820E-03  9179   0.109E-01    0.177E+01
```

*Correct Analysis:*

**Correct Output:**

PREDICTION=0 CONFIDENCE=9 REASONING=dE remains high and rms(c) stagnates at 1.7-1.8 showing no decreasing trend indicating divergence

---

**Your Task:**
Now, analyze the following `OSZICAR` data. Use the Key Indicators and especially the **Expert's Tie-Breaker Rule** to make your judgment.

```
{oszicar_content}



**Confidence Score Guidance:**
Your confidence score should now reflect the certainty of your phenotype classification.

-   **9 (Extremely Confident)**: The data is a textbook example of either "Ideal Convergence" or "Catastrophic Divergence", or it perfectly matches the "Stagnation" example.
-   **7-8 (Highly Confident)**: The data strongly points to "Stagnation" or a clear case of "Damped Oscillatory Convergence" where the damping trend is unambiguous, similar to the provided example.
-   **5-6 (Moderately Confident)**: The data appears to be a "Damped Oscillatory" case, but the downward macro-trend is slow or noisy, making the final outcome less certain within a reasonable number of total steps. There is significant ambiguity.
-   **0-4 (Low Confidence / Guess)**: The data is too short or too erratic to be confidently classified into any phenotype.

**Your response MUST be in the following single-line format:**
PREDICTION=number CONFIDENCE=number REASONING=brief_reason

Where:
- PREDICTION: 1 for 'converges', 0 for 'diverges'
- CONFIDENCE: integer between 0 and 9 (9=extremely confident, 0=pure guess)
- REASONING: brief explanation referencing the specific trends and rules from this prompt

**Output only this single line, no additional text.**
