from textwrap import indent
from tracemalloc import start
from uuid import uuid4
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializer import *
from django.conf import settings
from django.core.files.storage import default_storage
import fitz
from os import path as ospath, listdir
from .prediction import Predictor
import pandas as pd
import json


# Create your views here.
def save_files(files, dir):
    file_names = []
    for file in files:
        file_name = default_storage.save(f"{dir}/{file.name}", file)
        file_names.append(file_name)

    return file_names


def get_text(files_names):
    media_root = settings.MEDIA_ROOT
    
    for fname in files_names:
        file_url = ospath.join(media_root, fname)
        doc = fitz.open(file_url)
        yield "".join([str(page.get_text()) for page in doc])
        

def predict_per(file_names):
    actual_personality = {
                    'I': "Introvert",
                    'E': "Extrovert",
                    'N': "Intution",
                    'S': "Sensing",
                    "T": "Thinking",
                    'F': "Feeling",
                    "J": "Judging",
                    "P": "Perceiving"
                }
    __pd_structure = {"name" :pd.Series([], dtype=pd.StringDtype())}
    __pd_structure.update({value: pd.Series([])  for value in actual_personality.values()})
    __pd_structure.update({"personality_cat": pd.Series([], dtype=pd.StringDtype())})
    data_result = pd.DataFrame().from_dict(__pd_structure)
    for resume in get_text(file_names):
        dataFrame = Predictor(file_names, resume)
        data_result = pd.concat([data_result, dataFrame.data_result])

    return json.loads(data_result.to_json(orient="split"))
    

@api_view(['POST'])
def predictor(request):
    try:
        if request.method == "POST":
            root_dir = "tempDir"
            per_prob = ""
            if not request.session.has_key("pc-id"):
                uuid = str(uuid4())
                request.session["pc-id"] = uuid
                request.session.set_expiry(0)
                files = dict(request.FILES)["files"]
                dir = ospath.join(root_dir, f"{uuid}")
                file_names = save_files(files, dir)
                per_prob = predict_per(file_names)
            elif request.session.has_key("pc-id"):
                session_value = request.session["pc-id"]
                dir = ospath.join(root_dir, session_value)
                media_root = ospath.join(settings.MEDIA_ROOT, dir)
                file_names =  [ospath.join(dir, f) for f in listdir(media_root)]
                per_prob = predict_per(file_names)

        return Response(per_prob, status=status.HTTP_200_OK)
    except Exception as e:
        print("=====> Error", e)
        return Response({"working": False}, status=status.HTTP_509_BANDWIDTH_LIMIT_EXCEEDED)
