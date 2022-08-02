from uuid import uuid4

# django rest framework

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response


from django.conf import settings
from django.core.files.storage import default_storage
import fitz
from os import path as ospath, listdir, system
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


def predict_per(file_names, dir):
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
    __pd_structure = {"name": pd.Series([], dtype=pd.StringDtype())}
    __pd_structure.update({value: pd.Series([])
                          for value in actual_personality.values()})
    __pd_structure.update(
        {"Type": pd.Series([], dtype=pd.StringDtype())})

    data_result = pd.DataFrame().from_dict(__pd_structure)
    for resume, file_name in zip(get_text(file_names), file_names):
        dataFrame = Predictor(file_name, resume)
        data_result = pd.concat([data_result, dataFrame.data_result])

    return json.loads(data_result.to_json(orient="split"))


ROOT_DIR = "temp"


@api_view(['POST'])
def predictor(request):
    try:
        if request.method == "POST":
            per_prob = ""
            if request.session.has_key("pc-id"):
                clearSessionDeleteFiles(request)

            request.session.flush()
            uuid = str(uuid4())
            request.session["pc-id"] = uuid
            request.session.set_expiry(0)
            files = dict(request.FILES)["files"]
            dir = ospath.join(ROOT_DIR, f"{uuid}")
            file_names = save_files(files, dir)
            per_prob = predict_per(file_names, dir)

        return Response(per_prob, status=status.HTTP_200_OK)
    except Exception as e:
        if request.session.has_key("pc-id"):
            clearSessionDeleteFiles(request)

        print("=====> Error", e)
        return Response({"working": False}, status=status.HTTP_509_BANDWIDTH_LIMIT_EXCEEDED)


def clearSessionDeleteFiles(request):
    res = {
        "error": True
    }
    if request.session.has_key("pc-id"):
        pc_id = request.session["pc-id"]
        dir = ospath.join(settings.MEDIA_ROOT, ROOT_DIR, pc_id)
        system(f"rm -rf {dir}")
        request.session.flush()
        res["error"] = False
    return res


@api_view(["POST"])
def clearSession(request):
    try:
        if request.method == "POST":
            res = clearSessionDeleteFiles(request)

    except Exception as e:
        return Response({**res, "msg": "Server Side Error"}, status=status.HTTP_403_FORBIDDEN)

    return Response({**res, "msg": "Cleared"}, status=status.HTTP_200_OK)
