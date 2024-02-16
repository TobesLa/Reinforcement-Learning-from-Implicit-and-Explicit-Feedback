# Reinforcement Learning from Implicit and Explicit Feedback

## Description

Welcome to my bachelor thesis project, which focuses on training a reinforcement learning agent from simulated human feedback in Python. This project allows users to train the agent within the Gymnasium environments "Frozenlake" and "Cliffwalking".

### Key Features

- Utilizes reinforcement learning techniques.
- Simulates human feedback for training.
- Incorporates multiple feedback variants that modify the way feedback is calculated.
- Compatible with Gymnasium environments: "Frozenlake" and "Cliffwalking".

## Installation

To run this project, follow the steps below:

1. Ensure you have Python installed on your system. If not, download and install it from [python.org](https://www.python.org/downloads/).

2. Create a virtual environment using Conda by executing the following command in your terminal:

    ```bash
    $ conda create --name <env> --file environments.txt
    ```

    Replace `<env>` with your preferred environment name.

3. Activate the virtual environment:

    ```bash
    $ conda activate <env>
    ```

4. Install the required dependencies:

    ```bash
    $ pip install -r requirements.txt
    ```

## Usage

### Training the Agent

To train the reinforcement learning agent and visualize its final policy with the gymnasiums gui, run the following command:
    
    $ python Result_Feedback_Agent.py

Consider modifying the file Result_Feedback_Agent.py to explore different feedback variants and both environments.

For rerunning any experiment run:

First change the directory to the Experiments directory:
    
    $ cd Experiments

Then run the experiment:

    $ python exp_<experimentname>.py

Replace `<experimentname>` with your preferred experiment name from the directory Experiments.

To visualize the result's run:

    $ python plot_<experimentname>.py

Replace `<experimentname>` with your preferred experiment name from the directory Experiments. 
The plots will be shown, and saved in the directory "Plots".
