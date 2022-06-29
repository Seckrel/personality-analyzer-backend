from rest_framework import serializers

class ListSerializer(serializers.ListField):
    child = serializers.FileField()

class FileSerializer(serializers.DictField):
    child = ListSerializer()