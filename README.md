# FORCE 2020 Machine Learning Contest - Well logs

### NOTE: This repository is archived since 27 March 2025.

This GitHub repo relates to the Well Log competition of the [FORCE 2020 Machine Learning Contest](https://www.npd.no/en/force/events/machine-learning-contest-with-wells-and-seismic/), sponsored by Equinor.

The objective of the competition is to correctly predict lithology labels for provided well logs, provided NPD lithostratigraphy and well X, Y position. The training dataset contains 98 wells, whereas the test dataset includes 10 wells.

Firstly, the team went through an Exploratory Data Analysis (EDA) phase, aiming to have a better understanding of the dataset by looking at different types of plots (boxplots, cross-plots, etc) and data completeness. Secondly, a preprocessing phase aimed to perform some cleaning and feature engineering in order to reinforce the model. The modelling strategy consisted in using XGB and CatBoost classifiers to predict the lithology labels based on the different features available. The model had been applied to regional clusters in an attempt to capture local geological information. Some learnings can also be taken away concerning data visualization (well logs visualization, map view, QC, etc).

A general presentation can be found on SharePoint following this [link](https://statoilsrm.sharepoint.com/:p:/s/FORCEMLContest2020-WellLogs/EX-AgAL10o9OtpGtT2lY49MBK2RpncgYDVHg3SzFoD3yVQ?e=KFIxfq).

More information can be found on the [Xeek](https://xeek.ai/challenges/force-well-logs/overview) platform for this event.

The link to the model weights is: [link](https://statoilsrm-my.sharepoint.com/:f:/g/personal/hzan_equinor_com/EnRvm3FDU_NKmWkKryj5G08BYe8v4Wt5gUKJtIRyVhGYYA?e=VNUW4E)

#### Note1: the force_h3g.py script, the test dataset and the model weights should be in the same folder.
#### Note2: the default of the force_h3g.py script is to run prediction for file 'test.csv'. If you want to train the models, change the parameter 'train' from False to True.
