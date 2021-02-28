from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import csr_matrix
from utils import functions
from variables import modelObjects

import json
import youtube_dl
import settings
import pandas as pd


def newVideos():

    # Collect the data using youtube dl and store on json
    for filter in settings.queryFilters:
        with youtube_dl.YoutubeDL(settings.ydl_opts) as ydl:
            infoSearched = ydl.extract_info("{}:{}".format(settings.baseQuery, filter))
            with open(settings.newVideosPath, "w") as output:
                for entry in infoSearched['entries']:
                    if entry is not None:
                        data = {"title": entry.get('title'), "description": entry.get('description'), "upload_date": entry.get('upload_date'), "uploader": entry.get('uploader'),
                                "uploader_id": entry.get('uploader_id'), "uploader_url": entry.get('uploader_url'), "channel_id": entry.get('channel_id'), "channel_url": entry.get('channel_url'),
                                "duration": entry.get('duration'), "view_count": entry.get('view_count'), "average_rating": entry.get('average_rating'), "webpage_url": entry.get('webpage_url'),
                                "is_live": entry.get('is_live'), "like_count": entry.get('like_count'), "dislike_count": entry.get('dislike_count'), "channel": entry.get('channel'),
                                "extractor": entry.get('extractor'), "n_entries": entry.get('n_entries'), "playlist": entry.get('playlist'), "playlist_id": entry.get('playlist_id'),
                                "thumbnail": entry.get('thumbnail'), "fps": entry.get('fps'), "height": entry.get('height'),
                                "url": entry.get('url'), "width": entry.get('width'), "query": filter}
                        output.write("{}\n".format(json.dumps(data)))

def predictions():
    
    # Treat the data
    data = pd.read_json(settings.newVideosPath, lines=True)
    data = functions.treat(data)
    
    # Clean the data
    cleanedData = functions.clean(data)

    # Create features
    features = functions.createFeatures(cleanedData)

    # Extract data from text
    dataWText = functions.dataFromText(cleanedData, features, modelObjects.tFidVec)

    # Predict probabilities from models #

    # Random Forest
    probRF = functions.predictFromModel(dataWText,modelObjects.modelRF)

    # Light GBM
    probLGBM = functions.predictFromModel(dataWText,modelObjects.modelLGBM)
    
    # Logistic Regression
    scaledData = csr_matrix(dataWText.copy())
    scaler = MaxAbsScaler()
    scaledData = scaler.fit_transform(scaledData)

    probLR = functions.predictFromModel(scaledData,modelObjects.modelLR)

    # Ensembling the models

    prob = (probRF + probLGBM + probLR)/3

    return prob