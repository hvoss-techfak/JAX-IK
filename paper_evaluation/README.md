# Evaluation Code for the Paper "Real-Time Inverse Kinematics for Generating Multi-Constrained Movements of Virtual Human Characters"

Here you can find all the necessary code to evaluate the results presented in the paper. The code is organized into different sections, each corresponding to a specific part of the evaluation process.

# Installation
To run the evaluation code, you need to install the required dependencies. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

# Running the Evaluation
To run the evaluation, you can use the following command:

```bash
python timing.py
python timing_ablation.py
```
Please note that this will run all IK algorithms in all configurations 1010 times (10 warmup + 1000 evaluations) and save the results to json files. \
THIS WILL TAKE 10+ HOURS TO COMPLETE. \
At the end of the evaluation, the functions will automatically plot the results
