
import numpy as np
import os
import pandas as pd
import csv
from numpy import ndarray
from datetime import datetime
#pd.set_option('display.max_rows', None)
import tensorflow as tf
import time as time_now

import matplotlib.pyplot as plt
import astropy
import astropy.units as u
import lightkurve as lk
from lightkurve import TessLightCurveFile
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.timeseries import TimeSeries
import pickle
from pathlib import Path
from orion.data import load_signal
global train_data
train_data = load_signal('S-1-train')
global train_data2
train_data2 = pd.DataFrame(data = train_data)
from astroquery.mast import Catalogs
from astroquery.simbad import Simbad
import traceback
import math
from matplotlib.backends.backend_pdf import PdfPages

from datetime import datetime
import funcsTess
import symFunc

data_fix = funcsTess.data_fix
data_fix2 = funcsTess.data_fix2
data_fix3 = funcsTess.data_fix3
sym_func_p = symFunc.sym_func_p
sym_func_t = symFunc.sym_func_t

f = open('test_file', 'w')
writer = csv.writer(f)

tess_array =[]

times_detect = 0
tess_data = []

n = 0

global times_fit

times_fit = 0
times_detect = 0
#this is a function that takes the Fit file and fixes the data that goes in to the GANS machine analysis for anomalies.
#it changes 3 things 1) gets rid of rows that has empty light values, 2) changes the light values to the values divided by the average value of that filed.  So the value if normal is 1.0
#changes the timestamps to be just 1, 2, 3 ... etc for each row.

'''
from orion import Orion
# load saved model
#joblib.load



#This is another version of Orion to load other than TADGAAN

#Try setting the interval between each value and the other to 1 (frequency of data based on the head you are showing)

hyperparameters = {
                        "mlprimitives.custom.timeseries_preprocessing.time_segments_aggregate#1": {
                            "interval": 1,

                        },

                        "mlprimitives.custom.timeseries_preprocessing.rolling_window_sequences#1": {
                            "window_size": 100,
                            "target_column": [0, 1, 2, 3]

                        },
                        "orion.primitives.tadgan.TadGAN#1": {
                            'epochs': 1,
                            'verbose': True,
                            'input_shape': [100, 4]


                        }
            }

'''

with open('/home/nvasilescunyc/tess/TADGAN3/trained_model.pickle' , 'rb') as f:
    orion_test = pickle.load(f)

'''
orion_test = Orion(
    pipeline='tadgan',
    hyperparameters=hyperparameters
    )

'''
#orion_test = orion = Orion( pipeline='lstm_dynamic_threshold', hyperparameters=hyperparameters)





#This for loop goes through the various sectors for the files I want the machine learning to train with.
for y in range(1, 2):
    #print("This is y", y)

    if y < 10:
        path = "/data/scratch/data/tess/lcur/spoc/raws/sector-0"
        print(path)
    else:
        path = "/data/scratch/data/tess/lcur/spoc/raws/sector-"
        print(path)
    count = 0
    for root, dirs, files in os.walk(f"{path}{y}"):

        for name in files:
            count = count + 1
            if count > 0:
              break

            #print(data_fix(name))
            start = time_now.time()
            #return_array = (data_fix(name))
            #analyze_array = return_array[['timestamp','value']]
            #print("analyze_array", analyze_array)
            file_address = root + "/" + name

            orion_test.fit(data_fix(file_address))
            train_time = time_now.time() - start
            times_fit = times_fit + 1
            print("this is times fit #", times_fit)
            print("This is name of file", name)
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            #print("this is the time of fit:", dt_string)


            # create an iterator object with write permission - model.pkl
            with open('time_list_train.csv', 'a') as h:

                        writer2 = csv.writer(h)
                        # write a row to the csv file
                        row = [times_fit, train_time, dt_string]
                        writer2.writerow(row)
                        h.close()


            with open('/home/nvasilescunyc/tess/TADGAN3/trained_model.pickle', 'wb') as f:
                pickle.dump(orion_test, f)
                orion_test.save('/home/nvasilescunyc/tess/TADGAN3/trained_model.pickle')

#This is a for loop that goes through the files to detect light curve anomolies.
with open('master_anomaly_list.csv', 'a') as g:
    writer2 = csv.writer(g)
    row = ["TIC", "SECTOR", "Tmag", "Vmag", "Plx", "Lumclass", "RV_Value", "Start", "End", "Severity", "Date", "Classification", "Symmetry"]
    writer2.writerow(row)
    g.close()

for y in range(6, 27):
    #print("This is y", y)
    with open('/home/nvasilescunyc/tess/TADGAN3/trained_model.pickle' , 'rb') as f:
        orion_test = pickle.load(f)

    if y < 10:
        path = "/data/scratch/data/tess/lcur/spoc/raws/sector-0"

    else:
        path = "/data/scratch/data/tess/lcur/spoc/raws/sector-"

    count = 0
    for root, dirs, files in os.walk(f"{path}{y}"):


        for name in files:

            x_zoom = []
            y_zoom = []

            #if count > 10:
               # break
            n = n + 1
            count = count + 1

            times_detect = times_detect + 1
            print("this is times detect #", times_detect)
            print("This is name of file", name)
            start2 = time_now.time()
            file_address = root + "/" + name
            print("this is start time", start2)
            #print("analyze_array", analyze_array)
            open_file = data_fix(file_address)

            anomalies = orion_test.detect(open_file)
            #print("file address:", file_address)
            #print("this is the sector and file", end_address)
            print(f"This is anomalies in file {file_address}: \n", anomalies)

            end_time = time_now.time()

            train_time2 = end_time - start2
            now2 = datetime.now()
            dt_string2 = now2.strftime("%d/%m/%Y %H:%M:%S")

            with open('time_list_detect.csv', 'a') as i:

                writer2 = csv.writer(i)
                # write a row to the csv file
                row = [count, times_fit, train_time2, dt_string2]
                writer2.writerow(row)
                i.close()

            if anomalies[anomalies['severity'] > 0.90].empty == False:
                print("success")


                cut_one_array = np.array(open_file)
                print("cut one array \n", cut_one_array)
                second_col = cut_one_array[:, [1]]
                #second_col_nozeros = second_col[~np.all(second_col == 0, axis=1)]
                first_col = cut_one_array[:, [0]]
                first_col_nozeros = first_col[~np.all(first_col == 0, axis=1)]
                bh_count = np.count_nonzero(second_col > 1.05, axis= 0)
                bh_count2 = np.count_nonzero(second_col < 0.95, axis= 0)
                print(f"this is count 1 and count 2:  {bh_count} {bh_count2}")
                exo_count = np.count_nonzero(second_col > 1.05, axis= 0)
                exo_count2 = np.count_nonzero(second_col < 0.95, axis= 0)
                print(f"this is count 1 and count 2:  {exo_count} {exo_count2}")


                if bh_count > 20 and bh_count2 < 20:
                    classify = "PULSE"
                elif exo_count < 20 and exo_count2 > 20:
                    classify = "TRANSIT"
                else:
                    classify = "NY-Classified"

                if classify == "PULSE" :
                    sym_class = sym_func_p(second_col, 10)

                elif classify == "TRANSIT" :
                    sym_class = sym_func_t(second_col, 10)

                else:
                    sym_class = "NA"






                return_array = (data_fix2(file_address))
                cut_one_array2 = return_array
                cut_one_array3 = (data_fix3(file_address))
                start = np.array(anomalies['start'])
                end = np.array(anomalies['end'])
                severity = np.array(anomalies['severity'])


                print(anomalies)
                print(start, end, severity)

                file_address = root + "/" + name
                file_address_split = file_address.split("-")
                print("file name before file address split", file_address_split)
                file_name = str(file_address_split[3])
                print("file name", file_name)
                file_name = file_name.lstrip('0')
                print("file name before sector strip", file_name)
                sector =  str(file_address_split[2])
                print("sector before strip", sector)
                sector = sector.lstrip('s0')
                print("this is sector", sector)


                #mast and simbad data download
                tic = "TIC" + str(file_name)

                try:

                    catalogData = Catalogs.query_object(tic,  catalog = "TIC")
                    mast_data_tmag = catalogData[:1]['Tmag']
                    mast_data_vmag = catalogData[:1]['Vmag']
                    mast_data_plx = catalogData[:1]['plx']
                    mast_data_lumc = catalogData[:1]['lumclass']

                except:
                    catalogData = ["fail"]
                    mast_data_tmag = ["fail"]
                    mast_data_vmag = ["fail"]
                    mast_data_plx = ["fail"]
                    mast_data_lumc = ["fail"]
                    pass

                try:
                    Simbad.add_votable_fields('typed_id', 'rv_value', 'sp')
                    result_table = Simbad.query_object(tic)
                    print("\n")
                    simb_data = result_table['RV_VALUE']

                except:
                    simb_data = ["fail"]
                    pass

                #plt.ylim(0, 3)

                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                #print("this is the time of detect:", dt_string)
                #mast and simbad data download



                '''
                tic = "TIC" + str(file_name)
                catalogData = Catalogs.query_object(tic,  catalog = "TIC")
                mast_data = catalogData[:1]['Tmag','Vmag', 'plx', 'lumclass']
                #Simbad.add_votable_fields('typed_id', 'rv_value', 'sp')
                result_table = Simbad.query_object(tic)
                print("\n")
                simb_data = result_table['RV_VALUE']
                #print(f"{tic} MAST: {mast_data} and SIMBAD: {simb_data}")
                '''


                print("number of files found", n)
                data_plot = cut_one_array2
                #data_plot = data_fix(name)
                data_plot = np.array(data_plot)
                data_plot2= cut_one_array3
                data_plot2 = np.array(data_plot2)


                x_coordinate = data_plot[:, [0]]
                y_coordinate = data_plot[:, [1]]
                x2_coordinate = data_plot2[:, [0]]
                y2_coordinate = data_plot2[:, [1]]
                back_coordinate = data_plot[:, [2]]
                cent1_coordinate = data_plot[:, [3]]
                cent2_coordinate = data_plot[:, [4]]
                high_y = (max(y_coordinate) * 1.15)
                high_y = high_y.astype(np.float)
                print("high_y", high_y)
                low_y = (min(y_coordinate) * 0.85)
                low_y = low_y.astype(np.float)
                print("low y", low_y)

                print("file name", file_name)
                print("this is sector", sector)
                #sector =  str(file_name2[2])

                #plot matplot
                now = datetime.now()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                #print("this is the time of detect:", dt_string)

                with PdfPages(f"{tic}_{y}_{classify}_{sym_class}.pdf") as pdf:
                    #plot 1 raw average light curve
                    plt.figure(figsize = (10, 5.2))
                    #plt.ylim(low_y, high_y)
                    plt.title(f"TIC {file_name} Sector {y}")
                    plt.text(2, 3, tic)
                    plt.plot(x_coordinate, y_coordinate, color="green")
                    plt.xlabel('Time')
                    plt.ylabel('Light Values (Unchanged)')
                    plt.figtext(0.40, 0.85, " ", horizontalalignment ="center", verticalalignment ="top", wrap = True, fontsize = 10, color ="black")
                    #plt.axis("tight")
                    #plt.axis('off')
                    #plt.show()
                    #pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

                    #plot 2 medium average light curve
                    plt.figure(figsize = (10, 5.2))
                    #plt.ylim(low_y, high_y)
                    plt.title(f"Target TIC {file_name} Sector {y} {sym_class}" )
                    plt.text(2, 3, tic)
                    plt.plot(x2_coordinate, y2_coordinate, color="blue")
                    plt.xlabel('Time')
                    plt.ylabel('Relative Flux (Electrons Per Second)')
                    plt.figtext(0.40, 0.85, " ", horizontalalignment ="center", verticalalignment ="top", wrap = True, fontsize = 10, color ="black")
                    #plt.axis("tight")
                    #plt.axis('off')
                    #plt.show()
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()





                    '''
                    #plot 2 normal Lightcurve
                    lc_1= TessLightCurveFile(file_address)
                    ax1 = lc_1.plot()
                    ax1.set_title("Regular LC Plot")
                    #plt.show()
                    #ax1.figure.savefig(f'{tic}-sec{y}_Lc_plot.png')
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()
                    '''

                    #tpf_target = lk.search_targetpixelfile(f"{file_name}", author="Tess", cadence='long', sector={sector}).download()

                    #plot 3 back light compared to light curve
                    plt.figure(figsize = (10, 5.2))
                    high_y = (max(y_coordinate) * 1.25)
                    high_y = high_y.astype(np.float)
                    low_y = (min(y_coordinate) * 0.75)
                    low_y = low_y.astype(np.float)
                    #plt.ylim(low_y, high_y)
                    plt.title(f"TIC {file_name} Sector {y} -- Background  curve")
                    #plt.text(2, 3, tic)
                    #plt.plot(x_coordinate, y_coordinate, color="green")
                    plt.plot(x_coordinate, back_coordinate, color="blue")
                    plt.xlabel('Time')
                    plt.ylabel('Flux (Electrons Per Second)')
                    plt.figtext(0.40, 0.85, " ", horizontalalignment ="center", verticalalignment ="top", wrap = True, fontsize = 10, color ="black")
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()

                    #plot 4 centroid1.

                    plt.figure(figsize = (10, 5.2))
                    high_y = (max(y_coordinate) * 1.25)
                    high_y = high_y.astype(np.float)
                    low_y = (min(y_coordinate) * 0.75)
                    low_y = low_y.astype(np.float)
                    #plt.ylim(low_y, high_y)
                    plt.title(f"TIC {file_name} Sector {y} -- Centroid 1 Curve")
                    plt.text(2, 3, tic)
                    #plt.plot(x_coordinate, y_coordinate, color="green")
                    plt.plot(x_coordinate, cent1_coordinate, color="magenta")
                    #plt.plot(x_coordinate, cent2_coordinate, color="cyan")
                    plt.xlabel('Time')
                    plt.ylabel('Flux (Electrons Per Second')
                    plt.figtext(0.40, 0.85, " ", horizontalalignment ="center", verticalalignment ="top", wrap = True, fontsize = 10, color ="black")
                    #plt.axis("tight")
                    #plt.axis('off')
                    #plt.show()
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()


                    #plot 5 centroid2.

                    plt.figure(figsize = (10, 5.2))
                    high_y = (max(y_coordinate) * 1.25)
                    high_y = high_y.astype(np.float)
                    low_y = (min(y_coordinate) * 0.75)
                    low_y = low_y.astype(np.float)
                    #plt.ylim(low_y, high_y)
                    plt.title(f"TIC {file_name} Sector {y} -- Centroid 2 Curve")
                    plt.text(2, 3, tic)
                    #plt.plot(x_coordinate, y_coordinate, color="green")
                    #plt.plot(x_coordinate, cent1_coordinate, color="magenta")
                    plt.plot(x_coordinate, cent2_coordinate, color="red")
                    plt.xlabel('Time')
                    plt.ylabel('Flux (Electrons Per Second)')
                    plt.figtext(0.40, 0.85, " ", horizontalalignment ="center", verticalalignment ="top", wrap = True, fontsize = 10, color ="black")
                    #plt.axis("tight")
                    #plt.axis('off')
                    #plt.show()
                    pdf.savefig()  # saves the current figure into a pdf page
                    plt.close()


                    time = 0


                    array_anomalies = np.array(anomalies)

                    for i in array_anomalies[0:]:

                        print("this is i in amomalies", i)
                        print("this is i severity", i[2])

                        if i[2]> 0.30:
                            time = time + 1
                            print("Time is", time)
                            sever_no = i[2]
                            sever_no = f"{sever_no:.3f}"

                            print("this is severity", sever_no)


                            if i[0] > 30:
                                start = int(i[0] - 30)
                            else:
                                start = int(i[0])

                            end = int(i[1])
                            print(array_anomalies)
                            print("This is start and end and severity", start, end, i[2])

                            for i in data_plot:
                                if (i[0] > start) and (i[0] < end):
                                    print("this is i row light value", i[1])
                                    x_zoom.append(i[0])
                                    y_zoom.append(i[1])
                            #print("This is zoom_array \n", zoom_array)


                            #print("x_zoom", x_zoom)
                            #print("y_zoom", y_zoom)

                            #plot 6 Zoom  light curve
                            plt.figure(figsize = (10, 5.2))
                            #plt.ylim(low_y, high_y)
                            plt.title(f"Zoom severity {sever_no}")
                            plt.plot(x_zoom, y_zoom, color="green")
                            plt.xlabel('Time')
                            plt.ylabel('Flux (RAW)')
                            plt.figtext(0.40, 0.85, " ", horizontalalignment ="center", verticalalignment ="top", wrap = True, fontsize = 10, color ="black")
                            #plt.axis("tight")
                            #plt.axis('off')
                            #plt.show()
                            pdf.savefig()  # saves the current figure into a pdf page
                            plt.close()

                            '''
                            with fits.open(file_address) as hdul:
                                data = hdul[1].data
                                #print("length before cut", len(data))
                                #print(data)
                                print("\n")
                                mask = data['TIME']

                                print("print length mask", len(mask))
                                print("this is end", end)
                                print("print mask end", mask[end])
                                print("mask 20", mask[end])

                                mask2 = mask < mask[end]
                                print("mask length after first cut", len(mask2))
                                count_mask = 0
                                for i in mask2:
                                    if i == True:
                                        count_mask = count_mask + 1
                                print("count mask2 is", count_mask)

                                print("mask", mask2)
                                newdata = data[mask2]
                                print("length after first cut newdata", len(newdata))
                                hdu = fits.BinTableHDU(data=newdata)
                                hdu.writeto(f'newtable[time].fits', overwrite=True)
                                print("\n\n\n")

                            with fits.open(f"newtable[time].fits") as hdul2:
                                data2 = hdul2[1].data
                                #print(data2)
                                mask3 = data2['TIME']
                                #print("mask length before second cut", len(mask3))
                                #print("this is start", start)
                                #print("This is mask start", mask[start])
                                mask4 = mask3 > mask[start]
                                newdata3 = data2[mask4]
                                hdu = fits.BinTableHDU(data=newdata3)
                                hdu.writeto(f'newtable[time]a.fits', overwrite=True)
                                #print("length after second cut", len(newdata3))
                                #print("\n\n\n")
                                #print(newdata3)

                            lc_1= TessLightCurveFile(f"newtable[time]a.fits")
                            ax1 = lc_1.plot()
                            ax1.set_title(f"Zoom severity {sever_no}")
                            #plt.show()
                            #ax1.figure.savefig(f'{tic}-sec{y}_Lc_plot.png')
                            pdf.savefig()  # saves the current figure into a pdf page
                            plt.close()
                            '''

                            plt.figure(figsize = (10, 5.2))
                            #plt.ylim(low_y, high_y)
                            plt.title(f"TIC {file_name} Sector {y} Severity from {start} to {end}")
                            plt.text(2, 3, tic)
                            plt.plot(x_coordinate, y_coordinate, color="green")
                            x_range = range(start, end)
                            x2 = list(x_range)
                            length_y = len(x2)
                            y2 = [y_coordinate[start]] * length_y
                            plt.plot(x2, y2, color="red")

                            #plt.arrow(x2[0], y2[0] * 1.02, 0, 0.03,  head_width = 1, width = 0.05)

                            plt.arrow(x2[0], y2[0] * 1.01,  0.0, -0.04, fc="k", ec="k", width = 100, head_width=300, head_length=100 )
                            #plt.annotate("Zoom", x2[0], y2[0] * 0.98)
                            '''
                            label_x = x2[0] - 20
                            label_y = y2[0]
                            arrow_x = x2[0]
                            arrow_y = y2[0] * 1.03
                            arrow_properties = dict(facecolor="black", width=0.5, headwidth=4, shrink=0.1)
                            plt.annotate("Zoom", xy=(arrow_x, arrow_y), xytext=(label_x, label_y),arrowprops=arrow_properties)
                            '''
                            plt.xlabel('Time')
                            plt.ylabel('Flux (RAW)')
                            plt.figtext(0.40, 0.85, " ", horizontalalignment ="center", verticalalignment ="top", wrap = True, fontsize = 10, color ="black")
                            #plt.axis("tight")
                            #plt.axis('off')
                            #plt.show()


                            pdf.savefig()  # saves the current figure into a pdf page
                            plt.close()




                with open('master_anomaly_list.csv', 'a') as g:

                        writer2 = csv.writer(g)
                        # write a row to the csv file

                        Tmag = str(mast_data_tmag)
                        Tmag = Tmag.replace("Tmag", "")
                        Tmag = Tmag.replace("-", '')
                        Vmag = str(mast_data_vmag)
                        Vmag = Vmag.replace("Vmag", "")
                        Vmag = Vmag.replace("-", '')
                        Plx = str(mast_data_plx)
                        Plx = Plx.replace("plx", "")
                        Plx = Plx.replace("-", '')
                        LumD = str(mast_data_lumc)
                        LumD = LumD.replace("lumclass", "")
                        LumD = LumD.replace("-", '')
                        rvD = str(simb_data)
                        rvD = rvD.replace("RV_VALUE", "")
                        rvD = rvD.replace("km", "")
                        rvD = rvD.replace("/", "")
                        rvD = rvD.replace("s", "")
                        rvD = rvD.replace("-", "")
                        row = [file_name, sector, Tmag, Vmag, Plx, LumD, rvD, start, end, severity, dt_string, classify, sym_class]
                        writer2.writerow(row)
                        g.close()



with open('/home/nvasilescunyc/tess/TADGAN/mypickle.pickle', 'wb') as f:
    pickle.dump(orion_test, f)
    #orion_test.save('/home/nvasilescunyc/tess/TADGAN3/mypickle.pickle')

#print("This is final round anomalies", anomalies)