**You are a senior computational scientist specializing in Density Functional Theory (DFT) calculations using VASP. Your mission is to meticulously analyze the initial electronic steps from a VASP `OSZICAR` file and predict the final convergence status of the single-point energy calculation.**

**Key Indicators for Convergence and Divergence**:
To make an accurate prediction, pay close attention to the *trends* and *magnitudes* of the following parameters, especially in the later steps of the provided data:

1.  ***`dE` (Energy Change)***:
    *   **Convergence**: Should consistently decrease and approach very small absolute values (e.g., typically less than `1.0E-04` eV, ideally `1.0E-05` eV or smaller). The trend should be monotonic or show only minor, short-lived oscillations.
    *   **Divergence**: Persistent large oscillations, increasing absolute values, or failure to drop below `1.0E-03` eV are strong signs of non-convergence.

2.  ***`d eps` (Electronic Convergence Criterion)***:
    *   **Convergence**: Should consistently decrease and stabilize at a very small value (e.g., typically less than `1.0E-04` or `1.0E-05`).
    *   **Divergence**: Fluctuations or values remaining high (e.g., above `1.0E-03`) indicate poor electronic convergence.

3.  ***`rms` and `rms(c)` (Charge Density Residual)***:
    *   **Convergence**: Should show a clear and consistent decreasing trend, eventually reaching small values (e.g., `rms` typically less than `1.0E-02`).
    *   **Divergence**: Persistently high or oscillating `rms` values suggest the charge density is not stabilizing.

4.  **Overall Trend and Speed**:
    *   **Convergence**: A rapid initial drop followed by a gradual, stable decrease towards the thresholds.
    *   **Divergence**: Slow progress, erratic behavior, or a complete lack of a decreasing trend.

**Case Studies for Your Reference:**

**Case Study 1: A Classic Converging Calculation**
*This calculation is on a clear path to convergence. Notice how `dE` and `d eps` drop by orders of magnitude and stabilize at very low values.*

* **OSZICAR Data (First 50 steps):**

  ```
  N       E                     dE             d eps       ncg     rms          rms(c)
  RMM:   1    -0.206293489296E+04   -0.20629E+04   -0.34878E+05  4320   0.146E+03
  RMM:   2     0.893682908399E+03    0.29566E+04   -0.62657E+04  4320   0.389E+02
  RMM:   3     0.414327994413E+03   -0.47935E+03   -0.14852E+04  4320   0.223E+02
  RMM:   4    -0.200789120298E+03   -0.61512E+03   -0.56608E+03  4320   0.140E+02
  RMM:   5    -0.411351272625E+03   -0.21056E+03   -0.18030E+03  4320   0.781E+01
  RMM:   6    -0.467968540475E+03   -0.56617E+02   -0.50148E+02  4320   0.445E+01
  RMM:   7    -0.481528320575E+03   -0.13560E+02   -0.13315E+02  4320   0.228E+01
  RMM:   8    -0.485829644359E+03   -0.43013E+01   -0.41110E+01  4320   0.132E+01
  RMM:   9    -0.487925174700E+03   -0.20955E+01   -0.21193E+01 13376   0.718E+00
  RMM:  10    -0.487848966763E+03    0.76208E-01   -0.21187E+00 14502   0.250E+00
  RMM:  11    -0.487850451535E+03   -0.14848E-02   -0.25651E-01 13825   0.558E-01
  RMM:  12    -0.487849889312E+03    0.56222E-03   -0.25465E-02 14101   0.141E-01    0.106E+02
  RMM:  13    -0.383902332537E+03    0.10395E+03   -0.94338E+02 10903   0.556E+01    0.124E+02
  RMM:  14    -0.376713182729E+03    0.71891E+01   -0.46921E+01 11987   0.134E+01    0.703E+01
  RMM:  15    -0.389446061113E+03   -0.12733E+02   -0.89186E+01 10764   0.141E+01    0.832E+01
  RMM:  16    -0.373208657071E+03    0.16237E+02   -0.40132E+01 11203   0.837E+00    0.413E+01
  RMM:  17    -0.371621206978E+03    0.15875E+01   -0.14962E+01 11071   0.609E+00    0.247E+01
  RMM:  18    -0.372104817593E+03   -0.48361E+00   -0.37174E+00 11655   0.321E+00    0.111E+01
  RMM:  19    -0.372611387838E+03   -0.50657E+00   -0.13471E+00 10905   0.183E+00    0.744E+00
  RMM:  20    -0.372769638668E+03   -0.15825E+00   -0.38213E-01 11088   0.102E+00    0.570E+00
  RMM:  21    -0.372895768619E+03   -0.12613E+00   -0.21450E-01 10531   0.727E-01    0.419E+00
  RMM:  22    -0.373037842600E+03   -0.14207E+00   -0.16664E-01 10341   0.618E-01    0.281E+00
  RMM:  23    -0.373141279071E+03   -0.10344E+00   -0.74090E-02 10853   0.420E-01    0.170E+00
  RMM:  24    -0.373217722726E+03   -0.76444E-01   -0.36367E-02 10542   0.266E-01    0.110E+00
  RMM:  25    -0.373292128870E+03   -0.74406E-01   -0.25564E-02 10076   0.205E-01    0.776E-01
  RMM:  26    -0.373386553206E+03   -0.94424E-01   -0.30928E-02  9684   0.197E-01    0.593E-01
  RMM:  27    -0.373493558857E+03   -0.10701E+00   -0.39948E-02  9619   0.206E-01    0.499E-01
  RMM:  28    -0.373573024696E+03   -0.79466E-01   -0.28072E-02 10236   0.164E-01    0.443E-01
  RMM:  29    -0.373622018613E+03   -0.48994E-01   -0.20454E-02 10446   0.141E-01    0.387E-01
  RMM:  30    -0.373643768757E+03   -0.21750E-01   -0.81795E-03 10848   0.944E-02    0.334E-01
  RMM:  31    -0.373660494535E+03   -0.16726E-01   -0.77865E-03  9625   0.964E-02    0.297E-01
  RMM:  32    -0.373671047600E+03   -0.10553E-01   -0.91340E-03  9825   0.114E-01    0.255E-01
  RMM:  33    -0.373675111077E+03   -0.40635E-02   -0.69137E-03 10314   0.102E-01    0.223E-01
  RMM:  34    -0.373678138770E+03   -0.30277E-02   -0.56276E-03 10194   0.923E-02    0.205E-01
  RMM:  35    -0.373683882526E+03   -0.57438E-02   -0.67061E-03  9459   0.103E-01    0.179E-01
  RMM:  36    -0.373689650218E+03   -0.57677E-02   -0.74374E-03  9319   0.108E-01    0.144E-01
  RMM:  37    -0.373694267328E+03   -0.46171E-02   -0.10320E-02  9153   0.124E-01    0.126E-01
  RMM:  38    -0.373697181718E+03   -0.29144E-02   -0.26118E-03 10104   0.763E-02    0.102E-01
  RMM:  39    -0.373698474481E+03   -0.12928E-02   -0.13841E-03  9669   0.499E-02    0.750E-02
  RMM:  40    -0.373698995355E+03   -0.52087E-03   -0.15213E-03  8935   0.512E-02    0.630E-02
  RMM:  41    -0.373699745876E+03   -0.75052E-03   -0.90029E-04  9195   0.380E-02    0.511E-02
  RMM:  42    -0.373700441394E+03   -0.69552E-03   -0.34010E-04  9008   0.258E-02    0.411E-02
  RMM:  43    -0.373700861466E+03   -0.42007E-03   -0.20663E-04  8579   0.187E-02    0.361E-02
  RMM:  44    -0.373701146506E+03   -0.28504E-03   -0.18290E-04  8740   0.174E-02    0.359E-02
  RMM:  45    -0.373701245479E+03   -0.98972E-04   -0.23730E-04  8648   0.181E-02    0.302E-02
  RMM:  46    -0.373701292217E+03   -0.46738E-04   -0.36067E-04  8754   0.239E-02    0.292E-02
  RMM:  47    -0.373701336385E+03   -0.44167E-04   -0.41317E-04  8780   0.248E-02    0.272E-02
  RMM:  48    -0.373701414523E+03   -0.78138E-04   -0.29806E-04  8541   0.213E-02    0.190E-02
  RMM:  49    -0.373701503355E+03   -0.88832E-04   -0.21704E-04  8466   0.187E-02    0.152E-02
  RMM:  50    -0.373701626442E+03   -0.12309E-03   -0.90944E-05  8451   0.134E-02    0.119E-02
  ```

* **Correct Output:**

  PREDICTION=1 CONFIDENCE=9 REASONING=Strong consistent decrease in dE and d eps reaching convergence criteria by step 50

**Case Study 2: A Classic Diverging Calculation**
*This calculation is unstable. Despite some periods of decrease, the key indicators `dE` and `rms` remain high and volatile, showing no clear path to the convergence threshold.*

* **OSZICAR Data (First 50 steps):**

  ```
  N       E                     dE             d eps       ncg     rms          rms(c)
  RMM:   1    -0.160792346669E+04   -0.16079E+04   -0.35928E+05  3888   0.165E+03
  RMM:   2     0.809108433097E+03    0.24170E+04   -0.68422E+04  3888   0.457E+02
  RMM:   3     0.572372657967E+03   -0.23674E+03   -0.16370E+04  3888   0.258E+02
  RMM:   4    -0.590282353576E+02   -0.63140E+03   -0.63977E+03  3888   0.164E+02
  RMM:   5    -0.337570588933E+03   -0.27854E+03   -0.25128E+03  3888   0.975E+01
  RMM:   6    -0.437825828677E+03   -0.10026E+03   -0.91602E+02  3888   0.603E+01
  RMM:   7    -0.468365086596E+03   -0.30539E+02   -0.30661E+02  3888   0.338E+01
  RMM:   8    -0.479952644111E+03   -0.11588E+02   -0.11276E+02  3888   0.209E+01
  RMM:   9    -0.487561316685E+03   -0.76087E+01   -0.76947E+01 12189   0.121E+01
  RMM:  10    -0.487184687935E+03    0.37663E+00   -0.77236E+00 13076   0.391E+00
  RMM:  11    -0.487354790278E+03   -0.17010E+00   -0.23120E+00 12562   0.111E+00
  RMM:  12    -0.487483711595E+03   -0.12892E+00   -0.13475E+00 12519   0.380E-01    0.105E+02
  RMM:  13    -0.381986040714E+03    0.10550E+03   -0.11403E+03  9974   0.653E+01    0.121E+02
  RMM:  14    -0.375633534782E+03    0.63525E+01   -0.56017E+01 10793   0.152E+01    0.705E+01
  RMM:  15    -0.397251177982E+03   -0.21618E+02   -0.99918E+01 10922   0.169E+01    0.101E+02
  RMM:  16    -0.375447444713E+03    0.21804E+02   -0.66483E+01 10171   0.121E+01    0.462E+01
  RMM:  17    -0.371129135640E+03    0.43183E+01   -0.21613E+01 10615   0.879E+00    0.330E+01
  RMM:  18    -0.370974802764E+03    0.15433E+00   -0.71470E+00 10627   0.479E+00    0.187E+01
  RMM:  19    -0.370092156301E+03    0.88265E+00   -0.25615E+00 10705   0.306E+00    0.144E+01
  RMM:  20    -0.370152416668E+03   -0.60260E-01   -0.15901E+00 10500   0.207E+00    0.105E+01
  RMM:  21    -0.370819938554E+03   -0.66752E+00   -0.77950E-01 10674   0.148E+00    0.855E+00
  RMM:  22    -0.371270590051E+03   -0.45065E+00   -0.33810E-01 10918   0.947E-01    0.704E+00
  RMM:  23    -0.371867799444E+03   -0.59721E+00   -0.30264E-01 10490   0.857E-01    0.708E+00
  RMM:  24    -0.372059606011E+03   -0.19181E+00   -0.18405E-01 10733   0.700E-01    0.694E+00
  RMM:  25    -0.372106455737E+03   -0.46850E-01   -0.11202E-01 10204   0.542E-01    0.716E+00
  RMM:  26    -0.372055626931E+03    0.50829E-01   -0.13752E-01 10152   0.600E-01    0.673E+00
  RMM:  27    -0.372153955072E+03   -0.98328E-01   -0.39463E-01  9842   0.961E-01    0.609E+00
  RMM:  28    -0.373410484691E+03   -0.12565E+01   -0.29194E+00  9823   0.231E+00    0.273E+01
  RMM:  29    -0.373285770526E+03    0.12471E+00   -0.29698E-01 11237   0.128E+00    0.259E+01
  RMM:  30    -0.373337264859E+03   -0.51494E-01   -0.37746E-02 11308   0.282E-01    0.274E+01
  RMM:  31    -0.373515470890E+03   -0.17821E+00   -0.15981E-02 10661   0.181E-01    0.287E+01
  RMM:  32    -0.373631212115E+03   -0.11574E+00   -0.12868E-02  9540   0.207E-01    0.294E+01
  RMM:  33    -0.373561102419E+03    0.70110E-01   -0.29827E-03  9938   0.120E-01    0.291E+01
  RMM:  34    -0.373590419103E+03   -0.29317E-01   -0.12751E-03  9374   0.628E-02    0.292E+01
  RMM:  35    -0.373780089822E+03   -0.18967E+00   -0.14124E-02  9336   0.186E-01    0.284E+01
  RMM:  36    -0.373786287469E+03   -0.61976E-02   -0.79073E-03  8501   0.190E-01    0.288E+01
  RMM:  37    -0.373733779835E+03    0.52508E-01   -0.10716E-02  9808   0.165E-01    0.291E+01
  RMM:  38    -0.373817912835E+03   -0.84133E-01   -0.34222E-02  9412   0.279E-01    0.287E+01
  RMM:  39    -0.373814717043E+03    0.31958E-02   -0.15090E-02  8959   0.276E-01    0.286E+01
  RMM:  40    -0.373637846166E+03    0.17687E+00   -0.54792E-02  9731   0.383E-01    0.281E+01
  RMM:  41    -0.373597339695E+03    0.40506E-01   -0.99349E-03 10035   0.229E-01    0.278E+01
  RMM:  42    -0.373258847177E+03    0.33849E+00   -0.75878E-02  9612   0.420E-01    0.237E+01
  RMM:  43    -0.373392599188E+03   -0.13375E+00   -0.25106E-02 10333   0.282E-01    0.218E+01
  RMM:  44    -0.373331508463E+03    0.61091E-01   -0.13102E-02  9380   0.261E-01    0.204E+01
  RMM:  45    -0.373201116092E+03    0.13039E+00   -0.18551E-01  9426   0.667E-01    0.160E+01
  RMM:  46    -0.373365091800E+03   -0.16398E+00   -0.35816E-02 10598   0.367E-01    0.168E+01
  RMM:  47    -0.373464849108E+03   -0.99757E-01   -0.92826E-03 10714   0.156E-01    0.182E+01
  RMM:  48    -0.373464220632E+03    0.62848E-03   -0.76753E-03  9657   0.174E-01    0.178E+01
  RMM:  49    -0.373532054102E+03   -0.67833E-01   -0.69196E-03  9736   0.135E-01    0.178E+01
  RMM:  50    -0.373516730252E+03    0.15324E-01   -0.18820E-03  9179   0.109E-01    0.177E+01
  ```

* **Correct Output:**

  PREDICTION=0 CONFIDENCE=8 REASONING=Significant instability with large dE fluctuations far from convergence threshold indicating divergence

---

**Your Task:**

Now, analyze the following OSZICAR data from the first **{n_steps}** steps of a new calculation.

```
{oszicar_content}
```

Based on your expert analysis following the SOP and referencing the case studies, provide your prediction.

**Confidence Score Guidance (0-9 Scale)**:

-   **9 (Extremely Confident)**: The trends for `dE`, `d eps`, `rms`, and `rms(c)` are unequivocally clear and strong, pointing definitively towards either convergence or divergence. All key indicators are well within (for convergence) or far outside (for divergence) typical thresholds, with no significant ambiguity or contradictory signals.
-   **7-8 (Highly Confident)**: The trends are very clear, but there might be minor, short-lived deviations, or one indicator is slightly less pronounced while others are strong. The overall picture is still robust.
-   **5-6 (Moderately Confident)**: General trends are visible, but there are notable fluctuations, or the values are close to thresholds, making the ultimate outcome less certain. Some indicators might show slight ambiguity.
-   **3-4 (Low Confidence)**: Trends are weak, inconsistent, or highly ambiguous. Multiple indicators might contradict each other, or the data is noisy and difficult to interpret for a firm conclusion.
-   **0-2 (Pure Guess / Very Low Confidence)**: The data is insufficient (e.g., too few steps to establish a trend), extremely erratic, or provides no discernible pattern to make a meaningful prediction. This indicates high uncertainty.

Your response MUST be in the following single-line format:
PREDICTION=number CONFIDENCE=number REASONING=brief_reason

Where:
- PREDICTION: 1 for 'converges', 0 for 'diverges'
- CONFIDENCE: integer between 0 and 9 (9=extremely confident, 0=pure guess)
- REASONING: brief explanation for your prediction

Output only this single line, no additional text.
