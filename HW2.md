# Question 1: Imitation learning

1. Make necessary changes to the starter code to make it work for Behaviour cloning. Submit code as a zip file.

Code in: https://github.com/wooginawunan/reinforcementlearning/HW2/Q1

2. Attach images of frames/data that you collected in 1 page of your PDF submission.

<img align='left' width="150" height="100" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_2_files/img_0.png">
<img align='left' width="150" height="100" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_2_files/img_1.png">
<img align='left'  width="150" height="100" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_2_files/img_2.png">
<img align='left' width="150" height="100" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_2_files/img_3.png">
<img align='left' width="150" height="100" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_2_files/img_4.png">

For more images, please check https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_2_files

3. Attach images of frames/data that your behavior cloned policy produces in 1 page of your PDF submission.

I selected two series of snapshots, presenting how the car passes turns under my behavior cloned policy. 

<p align="center">
  <img width="460" height="300" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_3_files/case_1.png">
  <img width="460" height="300" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_3_files/case_2.png">
</p>

Please find all details in https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_3_files. 

4. Attach a curve that shows how important is the number of datapoints for behavior cloning in your PDF submission. In this curve, x-axis is the number of training examples and y-axis is the performance of the behavior cloning policy.

I collected 10,000 data samples in total and used 10% as validation set. Therefore the maximum training sample size is 9,000. To get the required curve, I trained models with 1,000, 3,000, 5,000, 7,000 and 9,000 samples as training data. The agent for each is selected based on its validation accuracy. Each agent has performed 7 episodes during testing. I used boxplot to present the collect rewards by each agent. We can see that using more datapoints is helpful for behavior cloning. Besides the boxplot, on the right, I showed the validation performance achieved by the models trained with each sample size. It shows consistent pattern with the collected rewards.      

<p align="center">
  <img width="460" height="300" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_4_files/training_samplesize.png">
  <img width="460" height="300" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_4_files/validation.png">
</p>

5. Additionally, attach screenshots, logs along with anything you feel is necessary as proof that your code worked in your PDF submission.

* Learning curve presenting training and validation performance of the model. 

<p align="center">
  <img width="460" height="300" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_5_files/training.png">
  <img width="460" height="300" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_5_files/val.png">
</p>

* Log recording five rounds of the behavior cloned policy. 

{"episode_rewards": [919.599999999984, 807.9459770114815, 768.3038327526003, 769.3158075601194, 799.2506493506305, 857.497173144859, 620.6547169811201], 
  "mean": 791.7954509715421, 
  "std": 85.61832234513746}

* Logs with real time testing for all trained models can be found in https://github.com/wooginawunan/reinforcementlearning/blob/main/Q1/results/

* One testing episode using agend trained with 7000 sample points is recorded and presented:

<p align="center">
  <video src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_5_files/Screen%20Recording%20-%20test-7000.mov" width="320" height="200" controls preload></video>
</p>

6. Write up any tricks you had to use to make behavior cloning work better.

I use ResNet-18 as the backbone achitecture for the network and applied the following tricks to improve the performance:

* weight decay is considered in the optimizer 
* A technique named modality dropout [https://arxiv.org/pdf/1501.00102.pdf] is utilized when learning from multiple images in the history.
* The ResNet-18s applied on history images are sharing weights with each other, which is also an common technique used in multi-modal learning.
* I considered lenght of history, learning rate as hyperparameters and conducted random search to find the optimal configuration. 
* Learning rate scheduler is added and learning rate is reduced with a rate of 0.1 at per 2000 steps.

The final best performed model is using learning of 0.1, weight decay of 0.001 and a history including 10 past+current images.

# Question 2: Value Iteration

### Section 2.2 

When s = 50, and a = 50, p(get reward |s = 50, a = 50) = p_h. 

which is higher than any other options with, for example,  p(get reward |s = 50, a = 49) = p_h * alpha, and alpha <= p_h. 

Similarly we can explain the oberseved patterns when s=25 and s=75.  

### Section 2.3 

Outputs for p= 0.25 and p = 0.55. Code in https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q2/question2.py

<p align="center">
  <img width="460" height="300" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q2/0.25.png">
  <img width="460" height="300" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q2/0.55.png">
</p>

### Section 2.4 

Action values: q_{k+1}(s, a) = \sum_{s', r} p(s', r | s, a) [r + max(q_k(s', a'))] 

