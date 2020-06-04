from django.db import models

class TermModel(models.Model):

    id_doc =  models.IntegerField()
    term_ngram = models.CharField(max_length=255)
    count = models.IntegerField()
    tf = models.FloatField()
