# Parking Space Detector

This app will help us to detect free parking space in the allocated parking spots by using image Processing.

<!-- ## Dataset

The dataset was taken from [Princeton Modelnet](https://modelnet.cs.princeton.edu/).

[Modelnet10 Download ](http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip) 
![alt text](https://github.com/Shadin710/CAD-Object-Classification/blob/main/Images/A-sample-model-from-each-category-of-the-ModelNet10-dataset.png?raw=True) -->

## Language & OS
1. **Python 3.10.4**
2. **Windows 10 home**

## Modules
1. [Numpy](https://numpy.org/)
2. [Email](https://pypi.org/project/email/)
3. [CVZONE](https://github.com/cvzone/cvzone)
4. [Pickle](https://www.synopsys.com/blogs/software-security/python-pickling/#:~:text=Pickle%20in%20Python%20is%20primarily,transport%20data%20over%20the%20network.)
5. [OpenCV](https://opencv.org/)


## Installation of modules
Please create a virtual environment then install the modules
To create virtual environment open command tab in that folder and run the command
``` bash
py -3.X venv -m name_of_env
```
X is refering a python version
To activate the virtual environment

```bash
.\name_of_env\Scripts\activate
```


To install all these modules copy the requirement.txt file and run the below command
```bash
pip install -r requirment.txt
```
## Execute
First, we need to pick or annotate the parking space
```bash
python parking_space_picker.py
```
Second, Run the **main.py**
```bash
python main.py
```
for reference you can check out the example files which is located in the **video** folder



## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
