# Team LSTM
### Members
1. George Boktor 
2. Nibraas Khan
3. Ruj Haan
4. Michael McComas
5. Ramin Daneshi

## Demo (With Python 3) (https://youtu.be/Zk5nWCd9CI0)
For as long as humans have participated in the act of communication, concealing information in those communicative mediums or concealing any traces of communication has manifested into an art of its own. An important technique used to conceal any traces of communication called steganography has been used in many domains including government level communication and encoding harmful media in innocuous media such as in phishing emails. Our work aims to crack a popular steganography algorithm and running our code will highlight exactly how to do that. 

We will have one demo file, ```demo/demo.ipynb``` to demonstrate quick and simple examples of our results. The notebook will highlight the capabilities of our Cycle Generative Adversarial Networks (CycleGANs) algorithm to crack the Least Significant Bit (LSB) Steganography technique.

Running the demo will show the effectiveness of our CycleGAN algorithm as shown here:

![Alt text](demo/results/cycle_gan_7.png?raw=true "Results of CycleGAN")

Our demo uses Jupyter Notebook, so the most basic steps needed to run our code is:

1. Install Jupyter Notebook (https://jupyter.org/install)
2. Open ```demo/demo.ipynb```

Below are more detailed instructions on how to work with our code including working with our packages, working with our datasets, manipulate our network, tuning hyper-parameters, and saving results. There is much more to be explored once you understand how to base CycleGAN algorithm works!

### Quick Start Guide
The following scripts and ipynb notebook executed in this order will allow you to install all the packages, install all the datasets, pre-process the data, and run the code (assuming you have Jupyter Notebook installed):

1. ```demo/setup -c```
2. ```demo/get_data```
3. ```demo/preprocess.py 15 10```
4. ```demo/demo.ipynb```

These steps will set up everything needed to run the code in ```demo/demo.ipynb``` for training and testing.

If you just want to see just the results of our experiments, follow these steps:

1. ```demo/setup```
2. ```demo/demo.ipynb```

### Detailed Steps

<details>
<summary>Installing Anaconda and Dependencies</summary>
<br>

#### Installing Anaconda and Dependences
1. To install Anaconda on a *Windows* machine, download the installer here: 
   https://www.anaconda.com/products/individual#windows
    
   Or

   To install Anaconda on a *Mac* machine, download the graphical installer here:   
   https://www.anaconda.com/products/individual#macos

2. After Anaconda is installed, initialize the Conda enviornment by executing this line in a command prompt:
```conda init```

3. Then, create a new enviornment, configured to Python 3, for this project:
```conda create -n cyclegan python=3```

4. Activate your newly created enviornment:
```conda activate cyclegan```

5. Install necessary scripts by running ```demo/setup```. Alternatively, run these install commands individually in your command line:
    - ```conda install -c anaconda tensorflow-gpu==1.14.0 -y```
    - ```conda install -c anaconda tensorflow-datasets -y```
    - ```conda install -c conda-forge glob2 -y```
    - ```conda install -c anaconda pillow -y```
    - ```conda install -c anaconda numpy -y```
    - ```conda install -c conda-forge matplotlib -y```
    - ```conda install -c conda-forge imageio -y```
    - ```conda install -c conda-forge tqdm -y```
    - ```conda install -c menpo pathlib -y```
    
6. The notebook also requires TensorFlow example:
    - ```pip install -q git+https://github.com/tensorflow/examples.git```    

</details>

<details>
<summary>Downloading Pre-trained Weights</summary>
<br>

#### Downloading Pre-trained Weights
    
We provide two ways of getting the pre-trained weights for our model. You can run the ```demo/set_up``` script with the ```-c``` flag or you can get them manually.
    
The steps for manual install are:

1. ```export fileid=1L0sh5pYQbsxFpDRRcTJ5FprBk5CemDc9``` -- for all checkpoints (10 GB)
    
   Or 
    
   ```export fileid=17kz2lVHgj0nYR-8hQoEeoU7mHWvoS-eY``` -- for the checkpoints needed to ```demo/demo.ipynb``` (2 GB)
2. ```export filename=checkpoints.zip```
3. ```wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt```
4. ```wget --load-cookies cookies.txt -O $filename 'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)```
5. ```rm -f confirm.txt cookies.txt```
6. ```unzip checkpoints.zip```                                                                                   
7. ```rm checkpoints.zip```
                                                                                                                                        
</details>
    
<details>
<summary>Getting Datasets and Pre-processing</summary>
<br>

#### Getting Datasets

Our script ```demo/get_data``` will grab data from the University of Berkeley Cycle Generative Networks datasets ```https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/```.

With the data as a zip file, the script will:

1. Unzip the file
2. Create four directories titled set1, set2, decodedArray, and encodedArray
3. Moves the files in the unzipped directory to the appropriate locations

To get a different dataset, replace the URL with the desired link in the ```demo/get_data``` script.

#### Pre-processing
    
Pre-processing in our case is taking the images from the ```demo/get_data``` script and passing them through the ```demo/sten.py``` to generate the encoded and decoded images. 
    
Calling the scripts as ```demo/preprocess.py 15 10``` will create all encoded and decoded images from all permutations of 15 of the first files from set1 and 10 of the first files from set2. Furthermore, it will create images for all bit sizes. 
    
The script ```demo/preprocess.py``` runs in parallel and can be modified to run on a single thread.
    
</details>
    
<details>
<summary>Training, Testing, Saving, and Manipulating the Model</summary>
<br>

#### Training and Saving

The code for training the CycleGAN model is found in ```demo/train.py```. The hyper-parameters, dataset, and saving mechanisms can be tweaked inside this file. 
    
To exceuate training run: 

```python3 demo/train.py [bit size]```
    
This command will run the code using the data from ```demo/get_data.py```, first 10 images from set 1 and first 5 images from set 2 as training data from the specified bit size (0-8), and will save the results in results in ```demo/checkpoints/cycle_gan_train_[bit size]```. All of these can be changed in the ```demo/train.py``` file.
    
#### Testing
    
After training is done, you can run: 
    
```python3 demo/test.py [bit size]```
    
The ```demo/test.py``` file assumes that you have trained in the bit size you are testing and the checkpoints have been saved in ```demo/checkpoints/cycle_gan_train_[bit size]```. The test file will generate an image, ```test.png``` that shows the results of the algorithm. The location of the checkpoints and the name of the file can be modified in ```demo/test.py```.
    
</details>
    
<details>
<summary>Tuning Hyper-parameters</summary>
<br>

#### Tuning Hyper-parameters 

To tune our hyper-parameters we used Bayesian Optimization. It tunes the hyper-parameters to get the best results for model performance.

The code for training the model using bayesian optimization can be found in ```demo/cycle_gan_bayes.py```. 
  
To execute Bayesian Optimization run: 

```python3 demo/cycle_gan_bayes.py [bit size] ```

The command will run the cycle gan model using bayesian optimization, for each iteration it will store the hyperparameters and the performance in ```demo/logs[bit size].json``` file. The search space of Bayesian Optimization can be changed in the file. 
    
</details>
