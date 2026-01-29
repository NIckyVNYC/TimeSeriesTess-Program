# TimeSeriesTess-Program
Time Series GANS on TESS Files
This is a python program I created that uses a GANS model from the open source AI program Orion ML to find anomolies in time series from TESS files.  The GANs model I used was trained on the time series of TESS Files that are example of exoplanets.   I trained the GANS models on the 7000 TESS files that are stars with exoplanets.  Then the GANs models can find times series in other TESS files that are anomolies compared to the times series in exoplanets.  This way, the program can go through millions of TESS files to find times series that are so anomaly that they could be examples of SETI, such as stars with Dyson Spheres around them. 

The main.py file is the main program.  the library.py has all the Python libraries that the program needs. The symFunc.py as various functions for the main program such as creating plots and changing the data to for GANs model.  
