# NN-for-ILS
A neural network model that predicts the densities of ionic liquids under conditions of varying pressure and temperature.  The model was trained on 3,500 data points from the scientific literature.

<h1>Complete code</h1>
The full code includes both the network model, and preprocessing and uses the learned model. 
The whole thing was written for jupyter notepad, which is convenient and used for ML/DL/Data Science tasks.

However, a Python (.py) file can be used. The following changes and modifications are planned in jupyter notepad:
- Load and preprocess in a separate file;
- Download and train the model;
- Separate file for the finished model
=======================================================================================================================
<h1>Model predictions</h1>
Take ionic liquid as an example:
'C3C1Pyr_NTF2'
with the molar masses of the cation and anion, respectively: 113.121 and 280.146 g/mol
And a pressure range of 0.1 MPa - 100 Mpa in steps of 10
and a temperature range of 293.15-318.15.

To get predictions for such given conditions we have to use the predictions4 function and the whole form of the call will look like this:
name = 'C3C1Pyr_NTF2'
Mcat = 113.121
Man = 280.146

P = [0.1,10,20,30,40,50,60,70,80,90,100]
T = [293.15,298.15,303.15,308.15,313.15,318.15]
result = predictions4(Mcat,Man,0,P,T);

<h1>TO DO</h1>
In order to make the code easier to read, I want to make the following changes:

- Using Python files, not Jupyter notebook;

- All the ionic liquids I used in the form of a dictionary, for example:
  - {'C3C1Pyr NTF2': {'Mc', 'Ma', 'P', 'T'}}
