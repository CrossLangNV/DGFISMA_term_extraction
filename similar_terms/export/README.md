For building: Run dbuild.sh from the project folder.

# App

With API running at localhost:80:

Docs:
    [http://localhost:12090/docs](http://localhost:12090/docs)

Using e.g. postman you can send Post request to [localhost:12090/export_sim_terms_eurovoc/](localhost:12090/export_sim_terms_eurovoc/)

### Quick rebuild and running:
    
    docker build -t glossarymapeurovocimage -f similar_terms/export/Dockerfile . & docker stop glossarymapeurovoccontainer & docker rm glossarymapeurovoccontainer & docker run --name glossarymapeurovoccontainer -p 12090:80 glossarymapeurovocimage

