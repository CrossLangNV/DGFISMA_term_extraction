For building: Run dbuild.sh from the project folder.

# Packages

Add DGFISMA RDF lib.

    pip install git+https://github.com/CrossLangNV/DGFISMA_RDF@development


# App

With API running at localhost:80:

Docs:
    [http://localhost:80/docs](http://localhost:80/docs)

Using e.g. postman you can send Post request to [localhost:80/similar_terms/align](localhost:80/similar_terms/align)

### Quick rebuild and running:
    
    docker build -t similartermsimage -f similar_terms\Dockerfile . & docker stop similartermscontainer & docker rm similartermscontainer & docker run --name similartermscontainer -p 80:80 similartermsimage

