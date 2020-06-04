from rest_framework import serializers
from .models import TermModel

class TermModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = TermModel
        fields = ( 'id_doc' ,'term_ngram' , 'count' , 'tf' )
