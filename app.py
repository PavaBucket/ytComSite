from flask import Flask
from variables import modelObjects
from utils import MLProcess
import json
import settings
import os

app = Flask(__name__)


def getPredictions():
    
    # initializing videos list
    videos = []

    # get videos data with predictions
    if not os.path.exists(settings.newVideosPath):
        MLProcess.newVideos()

    if not os.path.exists(settings.predResultsPath):
        MLProcess.predictions()
    

    with open("novos_videos.json", 'r') as dataFile:
        for line in dataFile:
            videos.append(json.loads(line))

    # make predictions
    predictions = []


    return predictions


# Home page
@app.route('/')
def main():

    # Initializing model objects
    modelObjects.init()

    predictions = getPredictions()

    return """<!doctype html>
    <html lang="pt-br">

    <head>

        <!-- Meta tags -->
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta name="author" content="Victor Campos">

        <title>Home</title>

    </head>

    <body>

        <table>
            {}
        </table>
    
    </body>

    </html> """.format(predictions)


# Code to run on server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

