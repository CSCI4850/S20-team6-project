# Team LSTM
### Members
1. George Boktor 
2. Nibraas Khan
3. Ruj Haan
4. Michael McComas
5. Ramin Daneshi

## Demo (With Python 3)
For as long as humans have participated in the act of communication, concealing information in those communicative mediums or concealing any traces of communication has manifested into an art of its own. An important technique used to conceal any traces of communication called steganography has been used in many domains including government level communication and encoding harmful media in innocuous media such as in phishing emails. Our work aims to crack a popular steganography algorithm and running our code will highlight exactly how to do that. 

We will have one demo file, ```cycle_gan.ipynb``` to demonstrate quick and simple examples of our results. The notebook will highlight the capabilities of our Cycle Generative Adversarial Networks (CycleGANs) algorithm to crack the Least Significant Bit (LSB) Steganography technique.

Our demo uses Jupyter Notebook, so the most basic steps needed to run our code is:

1. Install Jupyter Notebook (https://jupyter.org/install)
2. Open ```cycle_gan.ipynb```

Below are more detailed instructions on how to work with our code including working with our packages, working with our datasets, manipulate our network, tuning hyper-parameters, and saving results. There is much more to be explored once you understand how to base CycleGAN algorithm works!

### Quick Start Guide
The following scripts and ipynb notebook executed in this order will allow you to install all the packages, install all the datasets, pre-process the data, and run the code (assuming you have Jupyter Notebook installed):

1. ```demo/setup```
2. ```demo/get_data.py```
3. ```demo/preprocess.py 10 5```
4. ```demo/cycle_gan.ipynb```

These steps will set up everything needed to run the code in ```cycle_gan.ipynb```.

### Detailed Steps

<details>
<summary>Installing Anaconda and Dependencies</summary>
<br>

#### Installing Anaconda and Dependences
    
</details>
    
<details>
<summary>Getting Datasets and Pre-processing</summary>
<br>

#### Getting Datasets and Pre-processing
    
</details>
    
<details>
<summary>Training, Testing, and Saving</summary>
<br>

#### Training

The code for training the CycleGAN model is found in ```demo/train.py```. The hyper-parameters, dataset, and saving mechanisms can be tweaked inside this file. 
    
To exceuate training run: 
    
```python3 demo/train.py [bit size]```
    
This command will run the code using the data from ```demo/get_data.py```, first 10 images from set 1 and 5 images from set 2 as training data from the specified bit size (0-8), and will save the results in results in ```demo/checkpoints/cycle_gan_train_[bit size]```.
    
</details>
    
<details>
<summary>Tuning Hyper-parameters and Manipulating the Model</summary>
<br>

#### Tuning Hyper-parameters and Manipulating the Model
    
</details>