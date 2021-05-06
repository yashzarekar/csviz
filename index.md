# Transition from MATLAB to Python Controls Package

This page is a guide for installing the controls package for python and getting started with basic commands and functions in the controls toolbox.  

## Quick Links
- [1 - Installing Required Packages](#Installing-Required-Packages)  
    
- [2 - Using the Controls Toolbox](#Using-The-Controls-Toolbox)


<a name="Installing-Required-Packages"> </a> 
## 1 - Installing Required Packages

Jupyter notebook comes with important packages already installed. The user does not need to install NumPy, SciPy, or Matplotlib. For full functionality of the controls package, the **slycot** (Subroutine Library in Systems and Control Theory) package needs to be installed as a lot of functions defined in the controls package depend on slycot.

The controls and slycot packages can be installed using many methods. The two easiest methods are 

    1.1 - Using pip  
    1.2 - Using conda terminal (recommended for slycot)
    

### 1.1 - Pip

The controls package can be installed with a single line in a notebook code cell or Jupyter terminal (Jupyter notebook homepage > New > Terminal). Type and run the command 

>pip install control


**Note**: This method might not work for slycot because it requires many dependencies/compilers to be installed on the local machine. 


### 1.2 - Conda Terminal

In the Anaconda app, go to environments and click the arrow next to your desired environment ( **base (root)** if you do not have any virtual environments) and click "open terminal". Once the terminal is opened, type and run the command 
> conda install -c conda-forge control slycot

Type and enter "y" in the command prompt for slycot installation. See the picture below for steps to open the terminal: 


![image](https://user-images.githubusercontent.com/78013763/117369733-2f6f5280-ae7a-11eb-8f10-c432b331e14f.png)

<a name="Using-The-Controls-Toolbox"> </a> 
## 2 - Using the Controls Toolbox

The full documentation and guide can be found at [this link]( https://python-control.readthedocs.io/en/0.8.4/) (see Funtion reference bracket). Unlike MATLAB, the required directories/packages need to be imported in Python. The most common packages are numpy, scipy, matplotlib, and control. The controls package has a sub-directory for easy transition from MATLAB using the package **control.matlab**. The guide for this can be found at the link mentioned above. 

**Initialization** - The code below imports the required packages for running a program. 
```python
%matplotlib notebook 
#For interactive plots. This command creates plot that can be moved and shows x,y values when cursor is hovered on the plot. It can also be used for animations. %matplotlib inline just plots a pic of the graph.

import numpy as np
import scipy as sc
import control as co
import matplotlib.pyplot as plt

import warnings #Hide warning regarding imaginary plots - NOT RECOMMENDED for normal use
warnings.filterwarnings('ignore')
```

**Points to Remember**

- The array indexing in Python starts from **0** as opposed to 1 in MATLAB
- Array elements are accessed using square brackets ***[]*** as opposed to parentheses **()**
- Array elements need to be separated by a comma. E.g. array = np.array([1,2,3])
- Matrices are nested arrays and the first element in a 2-D Array would be an array itself. E.g: matrix = np.array([[1,2],[3,4]])
- While using a function defined in an imported package, use the package name or the assigned variable when calling the function. E.g: numpy.array or np.array, control.ss or co.ss as defined

### 2.1 System Creation 

To create a system, the functions **ss** and **tf** can be used to create a state space or a transfer function system, respectively. An example is illustrated below.
> **ss(A,B,C,D)** - Create a state space system  
> **tf(num,den)** - Create a transfer function system

```python
#Creating a transfer function system

num = [1,2] # Numerator co-efficients
den = [1,2,3] # Denominator co-efficients

sys_tf = co.tf(num,den)
print(sys_tf)
```

The output is the following:

![image](https://user-images.githubusercontent.com/78013763/117370557-68f48d80-ae7b-11eb-89d2-c1d9eaa73ef1.png)

```python
#Creating a state space system 

# Type system Matrices
A = np.array([[0,1],[-3,-2]])
B = np.array([[0],[1]])
C = np.array([2,1])
D = np.array([0])

sys_ss = co.ss(A,B,C,D)
print(sys_ss)
```
The output is the following:

![image](https://user-images.githubusercontent.com/78013763/117370935-f46e1e80-ae7b-11eb-831e-a78175282f3f.png)

### 2.2 System Conversion 

This section shows how to convert system from one type to another, i.e., from state-space to transfer function and vice-versa. Similar to MATLAB, the commands used are 
> **tf2ss(sys)** - Convert Transfer function to state-space  
> **ss2tf(sys)** - Convert state-space to transfer function

```python
# Transfer funtion to state space conversion of the above system
sys_ss2 = co.tf2ss(sys_tf)
print(sys_ss2)
```
The output is the following:
![image](https://user-images.githubusercontent.com/78013763/117371094-313a1580-ae7c-11eb-80b3-a70afd926961.png)

