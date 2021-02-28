# Youtube-dl options
ydl_opts = {
    'quiet': True,
    'skip_download': True,
    'forceid': True,
    'forcetitle': True,
    'forceurl': True,
    'forcejson': True,
    'ignoreerrors': True,
    'download': False,
}

# Paths
newVideosPath = './data/newVideos.json'
predResultsPath = './data/predResults.json'
modelLRPath = './models/LogisticRegression.pkl.z'
modelLGBMPath = './models/LightGBM.pkl.z'
modelRFPath = './models/RandomForest.pkl.z'
TFidVecPath = './models/Vectorizer.pkl.z'


# Query filters for collecting data
baseQuery = "ytsearchdate100"
queryFilters = ["machine+learning", "data+science", "kaggle"]


