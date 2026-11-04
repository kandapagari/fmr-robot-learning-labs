# S0E1 lab · Small errors don't stay small

Open in Colab: https://colab.research.google.com/github/kandapagari/fmr-robot-learning-labs/blob/main/labs/s0e1-drift/s0e1_drift.ipynb

- **Video:** [Why a Robot Isn't a Chatbot With Arms](https://youtu.be/bN-Yv_gUcMM)
- **Book:** [Action Models for Robot Learning, §1.1](https://action-models-book.vercel.app/chapters/1/1/)
- **Tier 1:** CPU only, ~20 minutes.

FMR-6 draws a 200 mm line in 100 steps with a 0.12° heading bias per step: 0.004 mm off at step 1, 21.1 mm off at step 100. Then one line of feedback keeps it under 0.2 mm.

Files: `drift.py` (the policies, ~50 lines), `s0e1_drift.ipynb` (the lab). The numbers in the video are checked by `tests/test_s0e1_drift.py`.

The notebook runs on the Franka FR3 by default; set `ARM = "fmr6"` to use the arm from the video. The numbers are the same for both.
