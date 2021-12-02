# CMPT 353 final project - Fall Detection

### Required libraries

    - All of the required libraries needed to run the project successfully are listed in the requirements.txt file. 
    - Notably, these are the libraries you most likely need:
        - numpy
        - pandas
        - matplotlib
        - pykalman
        - scipy
        - statsmodels
        - seaborn
        - sklearn
        - tabulate
    - The project should be run in Python 3.
    
### Order of execution and results produced
    Way 1.
        1) Run 'clean_save.ipynb'
            - Results produced: filtered files for each scenario
                - Location saved: 'Data/Cleaned/X/''
                - X is the scenario
                
        2) Run 'transformation.ipynb'
            - Results produces: one transformed file for each scenario
                - Location saved: 'Data/Transformed/X'
                - X is the scenario
            
        3) Run 'statistics.ipynb'
            - Results produced: graphs of multiple inferential and statistical tests
            - Images only displayed in the notebook, not saved
            
        4) Run 'machine_learning.ipynb'
            - Results produced: images for ROC Curve and Confusion matrix for each model we created
            - Images only displayed in the notebook, not saved
            - Best models saved in location: "Models"
        
        5) Run 'predict.ipynb'
            - Imports Saved models from "Models" 
            - Predicts and print results for never seen data in "Data/testData"
        
    Way 2.   
        1) Run 'app.py'
            This will use already trained model and produce results after cleaning and transforming the data.
            - Input: takes in an input recording data file name from the command line
            - Results produced: prints out whether if it was a fall or not with the accuracy score
            - How to run it: 
                Linux   : python3 app.py filename.csv 
                Windows : python app.py filename.csv
            
        Note:
            - The input file should have similar format to files in Data/walkSit/walkSit1.csv
            - Simalar files could be generated using IOS app : “Sensor Data Recorder” - Nils Ackermann
            - Use Acceleration.csv generated from the application
