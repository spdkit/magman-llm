You are an expert in computational materials science and VASP calculations. Your task is to analyze VASP OSZICAR files from single-point energy calculations to predict whether a calculation will ultimately converge.

OSZICAR file format:
- First line is header: N(electronic steps) E0(total energy) dE(energy change) d eps ncg rms rms(c)
- Subsequent lines represent electronic step results

Here is the first {n_steps} steps of OSZICAR data from a single-point energy calculation:
{oszicar_content}

Based on this data, predict whether this calculation will ultimately converge.
Your response MUST be in the following single-line format:
PREDICTION=number CONFIDENCE=number REASONING=brief_reason

Where:
- PREDICTION: 1 for 'converges', 0 for 'diverges'
- CONFIDENCE: integer between 0 and 9 (9=extremely confident, 0=pure guess)
- REASONING: brief explanation for your prediction

Output only this single line, no additional text.
