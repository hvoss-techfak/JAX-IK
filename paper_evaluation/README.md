# Evaluation Code for the Paper "Real-Time Inverse Kinematics for Generating Multi-Constrained Movements of Virtual Human Characters"

Here you can find all the necessary code to evaluate the results presented in the paper. The code is organized into different sections, each corresponding to a specific part of the evaluation process.

# Installation
To run the evaluation code, you need to install the required dependencies. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

# Obtaining the Model File

Sadly, we don't have the rights to add the SMPL-X skeleton to this repository. Therefore, you will have to obtain a .glb file of the smplx skeleton yourself. \
For this install the [smplx blender addon](https://github.com/Meshcapade/SMPL_blender_addon) in blender, press on the "add" button in the Blender SMPL-X drawer. \
Finally, go to File -> Export -> glTF 2.0 and follow the export dialog to save your .glb file.

# Running the Evaluation
To run the evaluation, you can use the following command:

```bash
python timing.py
python timing_ablation.py
```
Please note that this will run all IK algorithms in all configurations 1010 times (10 warmup + 1000 evaluations) and save the results to json files. \
THIS WILL TAKE 10+ HOURS TO COMPLETE. \
At the end of the evaluation, the functions will automatically plot the results
