# Question 1: Imitation learning

1. Make necessary changes to the starter code to make it work for Behaviour cloning. Submit code as a zip file.

Code in: https://github.com/wooginawunan/reinforcementlearning/

2. Attach images of frames/data that you collected in 1 page of your PDF submission.

<img align='left' width="150" height="100" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_2_files/img_0.png">
<img align='left' width="150" height="100" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_2_files/img_1.png">
<img align='left'  width="150" height="100" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_2_files/img_2.png">
<img align='left' width="150" height="100" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_2_files/img_3.png">
<img align='left' width="150" height="100" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_2_files/img_4.png">

For more images, please check https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_2_files

3. Attach images of frames/data that your behavior cloned policy produces in 1 page of your PDF submission.

I selected two series of snapshots, presenting how the car passes turns under my behavior cloned policy. Please find all details in  

For more images, please check https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_3_files. 

4. Attach a curve that shows how important is the number of datapoints for behavior cloning in your PDF submission. In this curve, x-axis is the number of training examples and y-axis is the performance of the behavior cloning policy.

5. Additionally, attach screenshots, logs along with anything you feel is necessary as proof that your code worked in your PDF submission.

* Learning curve presenting training and validation performance of the model. 

<p align="center">
  <img width="460" height="300" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_5_files/training.png">
  <img width="460" height="300" src="https://github.com/wooginawunan/reinforcementlearning/blob/main/HW2/Q1_5_files/val.png">
</p>

* Log recording five rounds of the behavior cloned policy. 

"""{"episode_rewards": [532.3503311258146, 770.8677419354677, 579.477464788721, 849.2243243243041, 851.2108614232069, 316.10111731842414], "mean": 649.8719734859898, "std": 193.68720112070298}"""

6. Write up any tricks you had to use to make behavior cloning work better.

I use ResNet-18 as the backbone achitecture for the network and applied the following tricks to improve the performance:

* weight decay is considered in the optimizer 
* A technique named modality dropout [https://arxiv.org/pdf/1501.00102.pdf] is utilized when learning from multiple images in the history.
* The ResNet-18s applied on history images are sharing weights with each other, which is also an common technique used in multi-modal learning.
* I considered lenght of history, learning rate as hyperparameters and conducted random search to find the optimal configuration. 

# Question 2: Value Iteration

### Section 2.2 


### Section 2.3 


### Section 2.4 

Action values: 

\begin{equation} q_{k+1}(s, a) = \sum_{s', r} p(s', r | s, a) [r + max(q_k(s', a'))]
\end{equation}
