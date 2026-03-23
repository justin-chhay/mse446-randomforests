#!/usr/bin/env python3

###############################################################################################################################
# Settings
###############################################################################################################################

RAW_DATA="raw_data/recs2020_public_v7.csv"

CLEAN_X_TRAIN="clean_x_train.csv"
CLEAN_X_TEST ="clean_x_test.csv"
CLEAN_Y_TRAIN="clean_y_train.csv"
CLEAN_Y_TEST ="clean_y_test.csv"

#Columns we are trying to predict
COLUMNS_TO_PREDICT = [
    "DOLLAREL", "DOLLARNG", "DOLLARFO", "DOLLARLP", "SCALEB", "SCALEG", "SCALEE", "PAYHELP", "ENERGYASST", "COLDMA", "HOTMA"
]

#Explicitly identify categorical columns for efficiency and so the one-hot-encoding works correctly (it wasn't otherwise in testing)
CATEGORICAL_COLUMNS = [
    "REGIONC",
    "DIVISION",
    "state_postal",
    "state_name",
    "BA_climate",
    "IECC_climate_code",
    "UATYP10",
    "TYPEHUQ",
    "WALLTYPE",
    "ROOFTYPE",
    "WINFRAME",
    "FUELPOOL",
    "FUELTUB",
    "TYPERFR1",
    "LOCRFRI2",
    "RANGEFUEL",
    "COOKTOPFUEL",
    "OVENFUEL",
    "OUTGRILLFUEL",
    "DWCYCLE",
    "DRYRFUEL",
    "TVTYPE1",
    "TVUSE1",
    "TVTYPE2",
    "TVUSE2",
    "TVTYPE3",
    "TVUSE3",
    "INTERNET",
    "EQUIPM",
    "FUELHEAT",
    "EQUIPAUXTYPE",
    "FUELAUX",
    "USEHUMID",
    "ACEQUIPM_PUB",
    "ACEQUIPAUXTYPE_PUB",
    "USECFAN",
    "DEHUMTYPE",
    "USEDEHUM",
    "TYPETHERM",
    "HEATCNTL",
    "COOLCNTL",
    "H2OMAIN",
    "WHEATSIZ",
    "FUELH2O",
    "FUELH2O2",
    "ELPAY",
    "NGPAY",
    "LPGPAY",
    "FOPAY",
    "SMARTMETER",
    "EMPLOYHH",
    "HOUSEHOLDER_RACE",
    "WOODTYPE",
    #TODO others if any are missed
]

X_COLUMNS_TO_LABEL_ENCODE = [
    #All non-one-hot categorical variables in the data set appear to be numbered in the order that makes most sense,
    #so no need to label encode anything really!
    #TODO others if any are missed
]

X_COLUMNS_TO_ONE_HOT_ENCODE = [
    "REGIONC",
    "DIVISION",
    "state_postal",
    "state_name",
    "BA_climate",
    "IECC_climate_code",
    "UATYP10",
    "TYPEHUQ",
    "WALLTYPE",
    "ROOFTYPE",
    "WINFRAME",
    "FUELPOOL",
    "FUELTUB",
    "TYPERFR1",
    "LOCRFRI2",
    "RANGEFUEL",
    "COOKTOPFUEL",
    "OVENFUEL",
    "OUTGRILLFUEL",
    "DWCYCLE",
    "DRYRFUEL",
    "TVTYPE1",
    "TVUSE1",
    "TVTYPE2",
    "TVUSE2",
    "TVTYPE3",
    "TVUSE3",
    "INTERNET",
    "EQUIPM",
    "FUELHEAT",
    "EQUIPAUXTYPE",
    "FUELAUX",
    "USEHUMID",
    "ACEQUIPM_PUB",
    "ACEQUIPAUXTYPE_PUB",
    "USECFAN",
    "DEHUMTYPE",
    "USEDEHUM",
    "TYPETHERM",
    "HEATCNTL",
    "COOLCNTL",
    "H2OMAIN",
    "WHEATSIZ",
    "FUELH2O",
    "FUELH2O2",
    "ELPAY",
    "NGPAY",
    "LPGPAY",
    "FOPAY",
    "SMARTMETER",
    "EMPLOYHH",
    "HOUSEHOLDER_RACE",
    "WOODTYPE",
    #TODO others if any are missed
]

#It was very useful to look at the codebook RECS provides to determine the sort of data prep we need to do
#We are predicting: DOLLAREL, DOLLARNG, DOLLARFO, DOLLARLP, SCALEB, SCALEG, SCALEE, PAYHELP, ENERGYASST, COLDMA, HOTMA
#We do not need to impute any data because the RECS has already done it for us
#However we do need to drop those imputation flag variables, final analysis weights, as well as all other columns in the
#"ENERGY ASSISTANCE" and "End-use Model" sections as that would kind of be cheating
COLUMNS_TO_DROP = [
    #Imputation flags
    "ZACEQUIPAGE", "ZADQINSUL", "ZAGECDRYER", "ZAGECWASH", "ZAGEDW",
    "ZAGEFRZR", "ZAGERFRI1", "ZAGERFRI2", "ZAIRCOND", "ZAMTMICRO",
    "ZATHOME", "ZATTCCOOL", "ZATTCHEAT", "ZATTIC", "ZATTICFAN",
    "ZATTICFIN", "ZBACKUP", "ZBASECOOL", "ZBASEFIN", "ZBASEHEAT",
    "ZBASEOTH", "ZBEDROOMS", "ZBLENDER", "ZCABLESAT", "ZCELLAR",
    "ZCELLPHONE", "ZCOLDMA", "ZCOMBODVR", "ZCONCRETE", "ZCOOKTOP",
    "ZCOOKTOPFUEL", "ZCOOKTOPINDT", "ZCOOKTOPUSE", "ZCOOLAPT", "ZCOOLCNTL",
    "ZCRAWL", "ZCROCKPOT", "ZCWASHER", "ZDEHUMTYPE", "ZDESKTOP",
    "ZDISHWASH", "ZDOOR1SUM", "ZDRAFTY", "ZDRYER", "ZDRYRFUEL",
    "ZDRYRUSE", "ZDVD", "ZDWASHUSE", "ZDWCYCLE", "ZEDUCATION",
    "ZELPAY", "ZELPERIPH", "ZEMPLOYHH", "ZENERGYASST", "ZENERGYASST16",
    "ZENERGYASST17", "ZENERGYASST18", "ZENERGYASST19", "ZENERGYASST20",
    "ZENERGYASSTOTH", "ZEQUIPAGE", "ZEQUIPAUXTYPE", "ZEQUIPM", "ZFOPAY",
    "ZFREEZER", "ZFUELAUX", "ZFUELH2O", "ZFUELH2O2", "ZFUELHEAT",
    "ZFUELPOOL", "ZFUELTUB", "ZGARGCOOL", "ZGARGHEAT", "ZH2OAPT",
    "ZH2OMAIN", "ZHEATAPT", "ZHEATCNTL", "ZHEATHOME", "ZHHAGE",
    "ZHHSEX", "ZHIGHCEIL", "ZHOTMA", "ZHOUSEFAN", "ZHUMIDTYPE",
    "ZICE", "ZINTERNET", "ZINTSTREAM", "ZINTYPEBROAD", "ZINTYPECELL",
    "ZINTYPEOTH", "ZKOWNRENT", "ZLGTIN1TO4", "ZLGTIN4TO8", "ZLGTINCAN",
    "ZLGTINCFL", "ZLGTINLED", "ZLGTINMORE8", "ZLGTOUTANY", "ZLGTOUTCAN",
    "ZLGTOUTCFL", "ZLGTOUTLED", "ZLGTOUTNITE", "ZLOCRFRI2", "ZLPGPAY",
    "ZMICRO", "ZMONEYPY", "ZMONPOOL", "ZMONTUB", "ZMORETHAN1H2O",
    "ZNCOMBATH", "ZNGPAY", "ZNHAFBATH", "ZNHSLDMEM", "ZNOACBROKE",
    "ZNOACDAYS", "ZNOACEL", "ZNOACHELP", "ZNOHEATBROKE", "ZNOHEATBULK",
    "ZNOHEATDAYS", "ZNOHEATEL", "ZNOHEATHELP", "ZNOHEATNG", "ZNUMADULT1",
    "ZNUMADULT2", "ZNUMCFAN", "ZNUMCHILD", "ZNUMDLHP", "ZNUMDLHPAC",
    "ZNUMFIREPLC", "ZNUMFLOORFAN", "ZNUMFREEZ", "ZNUMFRIG", "ZNUMLAPTOP",
    "ZNUMMEAL", "ZNUMPORTAC", "ZNUMPORTDEHUM", "ZNUMPORTEL", "ZNUMPORTHUM",
    "ZNUMSMPHONE", "ZNUMTABLET", "ZNUMWWAC", "ZONLNEDUC", "ZORIGWIN",
    "ZOTHROOMS", "ZOUTGRILLFUEL", "ZOUTLET", "ZOVEN", "ZOVENFUEL",
    "ZOVENUSE", "ZPAYHELP", "ZPLAYSTA", "ZPOOLPUMP", "ZPOWEROUT",
    "ZPRKGPLC1", "ZPRSSCOOK", "ZRANGE", "ZRANGEFUEL", "ZRANGEINDT",
    "ZRCOOKUSE", "ZRECBATH", "ZRICECOOK", "ZROOFTYPE", "ZROVENUSE",
    "ZSCALEB", "ZSCALEE", "ZSCALEG", "ZSDESCENT", "ZSEPDVR",
    "ZSIZEOFGARAGE", "ZSIZFREEZ", "ZSIZRFRI1", "ZSIZRFRI2", "ZSMARTSPK",
    "ZSQFTEST", "ZSQFTINCA", "ZSQFTINCB", "ZSQFTINCG", "ZSQFTRANGE",
    "ZSSLIGHT", "ZSSOTHER", "ZSSSECURE", "ZSSTEMP", "ZSSTV",
    "ZSTORIES", "ZSWIMPOOL", "ZTELLDAYS", "ZTELLWORK", "ZTEMPGONE",
    "ZTEMPGONEAC", "ZTEMPHOME", "ZTEMPHOMEAC", "ZTEMPNITE", "ZTEMPNITEAC",
    "ZTLDESKTOP", "ZTLLAPTOP", "ZTLMONITOR", "ZTLOTHER", "ZTLTABLET",
    "ZTOAST", "ZTOASTOVN", "ZTOPFRONT", "ZTREESHAD", "ZTVAUDIOSYS",
    "ZTVCOLOR", "ZTVONWD1", "ZTVONWD2", "ZTVONWD3", "ZTVONWE1",
    "ZTVONWE2", "ZTVONWE3", "ZTVSIZE1", "ZTVSIZE2", "ZTVSIZE3",
    "ZTVTYPE1", "ZTVTYPE2", "ZTVTYPE3", "ZTVUSE1", "ZTVUSE2",
    "ZTVUSE3", "ZTYPEGLASS", "ZTYPERFR1", "ZTYPERFR2", "ZTYPETHERM",
    "ZUGASHERE", "ZUPRTFRZR", "ZUSECFAN", "ZUSECOFFEE", "ZUSEDEHUM",
    "ZUSEEQUIPAUX", "ZUSEHUMID", "ZVCR", "ZWALLTYPE", "ZWASHLOAD",
    "ZWASHTEMP", "ZWHEATAGE", "ZWHEATBKT", "ZWHEATSIZ", "ZWHYPOWEROUT",
    "ZWINDOWS", "ZWINECHILL", "ZWINFRAME", "ZYEARMADERANGE", "ZTOTROOMS",
    "ZDNTHEAT", "ZTYPEHUQ", "ZSTUDIO", "ZOUTGRILL", "ZHOUSEHOLDER_RACE",
    "ZACEQUIPM_PUB", "ZACEQUIPAUXTYPE_PUB", "ZELAMOUNT", "ZNGAMOUNT",
    "ZLPAMOUNT", "ZFOAMOUNT", "ZWDAMOUNT",

    #Final analysis weights
    "NWEIGHT", "NWEIGHT1", "NWEIGHT2", "NWEIGHT3", "NWEIGHT4", "NWEIGHT5", 
    "NWEIGHT6", "NWEIGHT7", "NWEIGHT8", "NWEIGHT9", "NWEIGHT10", "NWEIGHT11", 
    "NWEIGHT12", "NWEIGHT13", "NWEIGHT14", "NWEIGHT15", "NWEIGHT16", "NWEIGHT17", 
    "NWEIGHT18", "NWEIGHT19", "NWEIGHT20", "NWEIGHT21", "NWEIGHT22", "NWEIGHT23", 
    "NWEIGHT24", "NWEIGHT25", "NWEIGHT26", "NWEIGHT27", "NWEIGHT28", "NWEIGHT29", 
    "NWEIGHT30", "NWEIGHT31", "NWEIGHT32", "NWEIGHT33", "NWEIGHT34", "NWEIGHT35", 
    "NWEIGHT36", "NWEIGHT37", "NWEIGHT38", "NWEIGHT39", "NWEIGHT40", "NWEIGHT41", 
    "NWEIGHT42", "NWEIGHT43", "NWEIGHT44", "NWEIGHT45", "NWEIGHT46", "NWEIGHT47", 
    "NWEIGHT48", "NWEIGHT49", "NWEIGHT50", "NWEIGHT51", "NWEIGHT52", "NWEIGHT53", 
    "NWEIGHT54", "NWEIGHT55", "NWEIGHT56", "NWEIGHT57", "NWEIGHT58", "NWEIGHT59", 
    "NWEIGHT60",

    #"ENERGY ASSISTANCE" section, other than variables we are predicting
    #"SCALEB",
    #"SCALEG",
    #"SCALEE",
    #"PAYHELP",
    "NOHEATBROKE",
    "NOHEATEL",
    "NOHEATNG",
    "NOHEATBULK",
    "NOHEATDAYS",
    "NOHEATHELP",
    #"COLDMA",
    "NOACBROKE",
    "NOACEL",
    "NOACDAYS",
    "NOACHELP",
    #"HOTMA",
    #"ENERGYASST", 
    "ENERGYASST20", 
    "ENERGYASST19", 
    "ENERGYASST18", 
    "ENERGYASST17", 
    "ENERGYASST16", 
    "ENERGYASSTOTH", 
    "ZCOLDMA", 
    "ZENERGYASST", 
    "ZENERGYASST16", 
    "ZENERGYASST17", 
    "ZENERGYASST18", 
    "ZENERGYASST19", 
    "ZENERGYASST20", 
    "ZENERGYASSTOTH", 
    "ZHOTMA", 
    "ZNOACBROKE", 
    "ZNOACDAYS", 
    "ZNOACEL", 
    "ZNOACHELP", 
    "ZNOHEATBROKE", 
    "ZNOHEATBULK", 
    "ZNOHEATDAYS", 
    "ZNOHEATEL", 
    "ZNOHEATHELP", 
    "ZNOHEATNG", 
    "ZPAYHELP", 
    "ZSCALEB", 
    "ZSCALEE", 
    "ZSCALEG",

    #"End-use Model" section, other than variables we are predicting
    "KWH",
    "BTUEL",
    #"DOLLAREL",
    "ELXBTU",
    "PERIODEL",
    "ZELAMOUNT",
    "KWHSPH",
    "KWHCOL",
    "KWHWTH",
    "KWHRFG",
    "KWHRFG1",
    "KWHRFG2",
    "KWHFRZ",
    "KWHCOK",
    "KWHMICRO",
    "KWHCW",
    "KWHCDR",
    "KWHDWH",
    "KWHLGT",
    "KWHTVREL",
    "KWHTV1",
    "KWHTV2",
    "KWHTV3",
    "KWHAHUHEAT",
    "KWHAHUCOL",
    "KWHCFAN",
    "KWHDHUM",
    "KWHHUM",
    "KWHPLPMP",
    "KWHHTBPMP",
    "KWHHTBHEAT",
    "KWHEVCHRG",
    "KWHNEC",
    "KWHOTH",
    "BTUELSPH",
    "BTUELCOL",
    "BTUELWTH",
    "BTUELRFG",
    "BTUELRFG1",
    "BTUELRFG2",
    "BTUELFRZ",
    "BTUELCOK",
    "BTUELMICRO",
    "BTUELCW",
    "BTUELCDR",
    "BTUELDWH",
    "BTUELLGT",
    "BTUELTVREL",
    "BTUELTV1",
    "BTUELTV2",
    "BTUELTV3",
    "BTUELAHUHEAT",
    "BTUELAHUCOL",
    "BTUELCFAN",
    "BTUELDHUM",
    "BTUELHUM",
    "BTUELPLPMP",
    "BTUELHTBPMP",
    "BTUELHTBHEAT",
    "BTUELEVCHRG",
    "BTUELNEC",
    "BTUELOTH",
    "DOLELSPH",
    "DOLELCOL",
    "DOLELWTH",
    "DOLELRFG",
    "DOLELRFG1",
    "DOLELRFG2",
    "DOLELFRZ",
    "DOLELCOK",
    "DOLELMICRO",
    "DOLELCW",
    "DOLELCDR",
    "DOLELDWH",
    "DOLELLGT",
    "DOLELTVREL",
    "DOLELTV1",
    "DOLELTV2",
    "DOLELTV3",
    "DOLELAHUHEAT",
    "DOLELAHUCOL",
    "DOLELCFAN",
    "DOLELDHUM",
    "DOLELHUM",
    "DOLELPLPMP",
    "DOLELHTBPMP",
    "DOLELHTBHEAT",
    "DOLELEVCHRG",
    "DOLELNEC",
    "DOLELOTH",
    "CUFEETNG",
    "BTUNG",
    #"DOLLARNG",
    "NGXBTU",
    "PERIODNG",
    "ZNGAMOUNT",
    "BTUNGSPH",
    "BTUNGWTH",
    "BTUNGCOK",
    "BTUNGCDR",
    "BTUNGPLHEAT",
    "BTUNGHTBHEAT",
    "BTUNGNEC",
    "BTUNGOTH",
    "CUFEETNGSPH",
    "CUFEETNGWTH",
    "CUFEETNGCOK",
    "CUFEETNGCDR",
    "CUFEETNGPLHEAT",
    "CUFEETNGHTBHEAT",
    "CUFEETNGNEC",
    "CUFEETNGOTH",
    "DOLNGSPH",
    "DOLNGWTH",
    "DOLNGCOK",
    "DOLNGCDR",
    "DOLNGPLHEAT",
    "DOLNGHTBHEAT",
    "DOLNGNEC",
    "DOLNGOTH",
    "GALLONLP",
    "BTULP",
    #"DOLLARLP",
    "LPXBTU",
    "PERIODLP",
    "ZLPAMOUNT",
    "BTULPSPH",
    "BTULPWTH",
    "BTULPCOK",
    "BTULPCDR",
    "BTULPNEC",
    "BTULPOTH",
    "GALLONLPSPH",
    "GALLONLPWTH",
    "GALLONLPCOK",
    "GALLONLPCDR",
    "GALLONLPNEC",
    "GALLONLPOTH",
    "DOLLPSPH",
    "DOLLPWTH",
    "DOLLPCOK",
    "DOLLPCDR",
    "DOLLPNEC",
    "DOLLPOTH",
    "GALLONFO",
    "BTUFO",
    #"DOLLARFO",
    "FOXBTU",
    "PERIODFO",
    "ZFOAMOUNT",
    "BTUFOSPH",
    "BTUFOWTH",
    "BTUFONEC",
    "BTUFOOTH",
    "GALLONFOSPH",
    "GALLONFOWTH",
    "GALLONFONEC",
    "GALLONFOOTH",
    "DOLFOSPH",
    "DOLFOWTH",
    "DOLFONEC",
    "DOLFOOTH",
    "BTUWD",
    "ZWDAMOUNT",
    "TOTALBTUSPH",
    "TOTALDOLSPH",
    "TOTALBTUWTH",
    "TOTALDOLWTH",
    "TOTALBTUOTH",
    "TOTALDOLOTH",
    "TOTALBTU",
    "TOTALDOL",

    #SNEAKY source of leakage in ENERGY BILLS section
    #which would contain information about being unable to pay electric bills!
    "WHYPOWEROUT",

    #Drop some other misc useless columns
    "DOEID",
]

###############################################################################################################################
# Imports
###############################################################################################################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

###############################################################################################################################
# Data cleaning code
###############################################################################################################################

#Read in the raw data
#https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
raw_data = pd.read_csv(RAW_DATA)
print(raw_data.shape)

#Drop all of the columns we need to get rid of
#https://stackoverflow.com/questions/13411544/delete-a-column-from-a-pandas-dataframe
clean_data = raw_data.drop(columns=COLUMNS_TO_DROP)
print(clean_data.shape)

#Drop all rows with missing data; we have enough data and there aren't so many missing values that this will cause an issue
clean_data = clean_data.dropna()#From class :)
print(clean_data.shape)

#Convert all categorical columns to actual categories for efficiency as we did in Module 4 (also helps one-hot stuff)
clean_data.info()
for category_column in CATEGORICAL_COLUMNS:
    clean_data[category_column] = clean_data[category_column].astype("category")
clean_data.info()

#Seperate out X and y columns
X = clean_data.drop(columns=COLUMNS_TO_PREDICT)
y = clean_data[COLUMNS_TO_PREDICT]
print(X.shape)
print(y.shape)

#Do a test-train split using a random_state of 123456 for reproducibility (just do the usual 80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123456)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Deal with encoding of categorical variables while avoiding leakage

#Process columns to label encode
for label_encode_column in X_COLUMNS_TO_LABEL_ENCODE:
    #BE CAREFUL: ONLY FIT ON THE TRAINING DATA TO AVOID LEAKAGE AS DISCUSSED IN CLASS
    label_encoder = LabelEncoder()
    X_train[label_encode_column] = label_encoder.fit_transform(X_train[label_encode_column])
    #Then just apply the same transform (no training) to the test data
    X_test[label_encode_column] = label_encoder.transform(X_test[label_encode_column])

#Similar process for one-hot
for one_hot_encode_column in X_COLUMNS_TO_ONE_HOT_ENCODE:
    #AGAIN AVOID LEAKAGE (use handle_unknown as we learned in class in case test data contains something we didn't expect)
    #one_hot_encoder = OneHotEncoder(handle_unknown="ignore")
    #dummy_X_train = one_hot_encoder.fit_transform(X_train[[one_hot_encode_column]])
    #Was running into weird incompatibilities with sklearn and pandas: "'csr_matrix' object has no attribute 'index'"
    #So just using pandas get_dummies() to do one hot encoding and this reindex() function I discovered to recombine things
    #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reindex.html

    dummy_X_train = pd.get_dummies(X_train[[one_hot_encode_column]])
    X_train = X_train.drop(columns=[one_hot_encode_column])
    X_train = X_train.join(dummy_X_train)

    #Repeat for test data without fitting
    #dummy_X_test = one_hot_encoder.transform(X_test[[one_hot_encode_column]])
    dummy_X_test = pd.get_dummies(X_test[[one_hot_encode_column]])
    dummy_X_test = dummy_X_test.reindex(columns=dummy_X_train.columns, fill_value=False)
    X_test = X_test.drop(columns=[one_hot_encode_column])
    X_test = X_test.join(dummy_X_test)

#Write everything out to disk
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
X_train.to_csv(CLEAN_X_TRAIN, index=False)
X_test.to_csv(CLEAN_X_TEST, index=False)
y_train.to_csv(CLEAN_Y_TRAIN, index=False)
y_test.to_csv(CLEAN_Y_TEST, index=False)
