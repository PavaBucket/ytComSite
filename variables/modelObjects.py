import joblib as jb
import settings

def init():

    global modelRF
    global modelLGBM
    global modelLR
    global tFidVec

    # Get model objects from files
    modelRF = jb.load(settings.modelRFPath)
    modelLGBM = jb.load(settings.modelLGBMPath)
    modelLR = jb.load(settings.modelLRPath)
    tFidVec = jb.load(settings.TFidVecPath)