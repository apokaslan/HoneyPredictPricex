# Honey Predict Price 

## Necessary installations
```
pip install streamlit.
pip install pandas.
pip install joblib.
pip install pycaret[all].
```
## Run commend
and Go to the file path where the stream.py file is located with ‘cd’ and then type "__streamlit run stream.py__" in the terminal


## Command to change the model 
If you want to use a different model than __AutoML__, all you need to do is line 6 in the "stream.py" file 
select a different model from the __.pkl__ files instead of ```model = load_model(‘AutoMLbest_model’) ```
for example:``` model= load_model('OneLayerdNetwork')```


### Open App
To run local, go to the local host given to you in the terminal: __http://localhost:8501__

### Training
The models were trained with colab. Trained models were saved and streamlit apps were created with vscode
