# Deep-Auto-White-Balance
Name: Santosh Vasa, Vikram Bharadwaj <br>
Date: 10/12/2022 <br>
Class: CS7180 - Advanced Perception <br>

### Operating System Used:
- Windows for Development - Windows 11
- Linux and Colab for Training 

### How to run? 
- Install requirements.txt <br> 
` pip install -r requirements.txt`
- Add the project folder to the PYTHONPATH:<br>
`export PYTHONPATH=${PYTHONPATH}:/path_to_project_folder`
- Update parameters.yaml with the data location. 
- link to datasets : <br>
  - http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
- Run python file train.py <br>
`python3 train.py`
- Time travel days:
  - 2 days
### RESULTS:

Areas where the network works well:
- Building outlines
- Water bodies
- Grass Outfields

Areas where the network does not work well:
- Roads occluded, Road classification
- Dark green trees mis classified as water
- Soft boundaries

Examples: <br>
![image info](out/result_1.png)
![image info](out/result_2.png)
![image info](out/result_3.png)
![image info](out/result_4.png)
![image info](out/result_5.png)
![image info](out/result_6.png)

